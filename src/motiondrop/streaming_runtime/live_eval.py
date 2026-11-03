"""Evaluation harness for the live multi-task runtime.

Suites (each run is resumable and writes one JSON line per stream):

* ``proactive``: StreamingBench Proactive Output. A watch task is created at
  the question time and must fire within +-2 s of the ground truth with the
  expected text (official scoring). Measures long-running conditional tasks.
* ``instant``: StreamingBench Real-Time Visual Understanding. One stream per
  video; four-choice questions are asked at their timestamps.
* ``ovo-backward``: OVO-Bench backward tasks (EPM/ASI/HLD); the question is
  asked at the end of each clip. Stresses KV eviction and the memory log.
* ``realtime``: wall-clock paced playback with periodic instant questions;
  measures question latency, commentary lag, preemption and dropped ticks.

Interference conditions are composed with flags: ``--commentary`` adds a
standing commentary task to every stream, ``--shared-history`` writes task
outputs into the trunk (the legacy design), ``--max-context-tokens`` enables
eviction, ``--memory-interval`` enables background note-taking.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from motiondrop.paths import ovobench_annotations, ovobench_videos, results_dir, streamingbench_root
from motiondrop.streaming_runtime.config import RuntimeConfig
from motiondrop.streaming_runtime.live import LiveConfig, LiveScheduler
from motiondrop.streaming_runtime.run import sampled_indices

SUITES = ("proactive", "instant", "ovo-backward", "realtime")
DEFAULT_COMMENTARY = "Commentate what is happening in the video."
# Official StreamingBench proactive protocol (src/benchmark/StreamingBenchProactive.py).
PROACTIVE_TEMPLATE = (
    "You are an advanced image question-answering AI assistant. You have been provided with "
    "images and a question related to the images. Your task is to carefully analyze the "
    "images and provide the answer to the question. You need to carefully confirm whether "
    "the images content meet the conditions of the question, and then output the correct "
    "content.\n\nQuestion: {}\n\nThe answer is:\n"
)
REALTIME_QUESTIONS = (
    "What is happening right now?",
    "Describe the most recent action in one sentence.",
    "What can you see on the screen at this moment?",
)


@dataclass
class StreamEvent:
    seconds: float
    kind: str  # "task" or "question"
    payload: dict


@dataclass
class StreamPlan:
    plan_id: str
    video: Path
    start_seconds: float
    end_seconds: float
    events: list[StreamEvent] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motiondrop live-eval", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--name", required=True, help="Condition name used for output files.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, default=results_dir() / "live-eval")
    parser.add_argument("--streamingbench-root", type=Path, default=streamingbench_root())
    parser.add_argument("--ovobench-json", type=Path, default=ovobench_annotations())
    parser.add_argument("--ovobench-videos", type=Path, default=ovobench_videos())
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--window-frames", type=int, default=2)
    parser.add_argument("--fixed-compression", type=float, default=0.9)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--position-mode", choices=("contiguous", "absolute"), default="contiguous")
    parser.add_argument("--no-time-markers", action="store_true")
    parser.add_argument("--memory-interval", type=float, default=0.0)
    parser.add_argument("--no-memory", action="store_true", help="Do not show notes to questions.")
    parser.add_argument("--memory-scope", choices=("evicted", "all"), default="evicted")
    parser.add_argument("--commentary", action="store_true")
    parser.add_argument("--commentary-text", default=DEFAULT_COMMENTARY)
    parser.add_argument(
        "--extra-standing",
        type=int,
        default=0,
        help="Additional standing tasks (realtime stress test).",
    )
    parser.add_argument("--standing-interval", type=float, default=4.0)
    parser.add_argument("--shared-history", action="store_true")
    parser.add_argument("--watch-threshold", type=float, default=0.5)
    parser.add_argument("--lead-seconds", type=float, default=20.0)
    parser.add_argument("--question-interval", type=float, default=15.0)
    parser.add_argument("--realtime-seconds", type=float, default=90.0)
    parser.add_argument("--limit", type=int, help="Maximum number of streams.")
    parser.add_argument("--only", help="Comma-separated stream IDs to run (exact match).")
    parser.add_argument("--per-video", type=int, help="Maximum questions per video.")
    parser.add_argument("--max-gpu-memory-gib", type=int)
    return parser


def runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        window_frames=args.window_frames,
        fixed_compression=args.fixed_compression,
        time_markers=not args.no_time_markers,
        max_context_tokens=args.max_context_tokens or None,
        position_mode=args.position_mode,
    )


def live_config(args: argparse.Namespace, tick_policy: str) -> LiveConfig:
    return LiveConfig(
        tick_policy=tick_policy,
        standing_interval_s=args.standing_interval,
        watch_threshold=args.watch_threshold,
        memory_interval_s=args.memory_interval,
        use_memory=not args.no_memory,
        memory_scope=args.memory_scope,
        commit_outputs=args.shared_history,
    )


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def _commentary_events(args: argparse.Namespace, start: float) -> list[StreamEvent]:
    events = []
    if args.commentary:
        events.append(
            StreamEvent(start, "task", {"kind": "standing", "instruction": args.commentary_text})
        )
    extras = (
        "Describe the crowd and the atmosphere.",
        "Keep track of the score.",
        "Describe what the camera is showing.",
    )
    for index in range(args.extra_standing):
        events.append(
            StreamEvent(
                start, "task", {"kind": "standing", "instruction": extras[index % len(extras)]}
            )
        )
    return events


def proactive_plans(args: argparse.Namespace) -> list[StreamPlan]:
    from motiondrop.benchmarks.streamingbench_data import parse_timestamp

    root = args.streamingbench_root
    csv_path = root / "StreamingBench" / "Proactive_Output.csv"
    plans: list[StreamPlan] = []
    per_video: dict[int, int] = defaultdict(int)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            question_id = row["question_id"].strip()
            sample = int(question_id.rsplit("_", 2)[-2])
            video = root / "videos" / "proactive_output" / f"sample_{sample}" / "video.mp4"
            if not video.is_file():
                continue
            if args.per_video is not None and per_video[sample] >= args.per_video:
                continue
            per_video[sample] += 1
            asked = parse_timestamp(row["time_stamp"])
            truth = parse_timestamp(row["ground_truth_time_stamp"])
            expected = row["ground_truth_output"].strip()
            question = row["question"].strip()
            start = max(0.0, asked - args.lead_seconds)
            check = (
                f'{question} Is it the right time to output "{expected}"? '
                "You can only answer yes or no."
            )
            events = _commentary_events(args, start)
            events.append(
                StreamEvent(
                    asked,
                    "task",
                    {
                        "kind": "watch",
                        "instruction": question,
                        "once": True,
                        "check_prompt": PROACTIVE_TEMPLATE.format(check),
                        "output_prompt": PROACTIVE_TEMPLATE.format(question),
                        "probe": True,
                    },
                )
            )
            plans.append(
                StreamPlan(
                    question_id,
                    video,
                    start,
                    truth + 4.0,
                    events,
                    {"asked": asked, "truth": truth, "expected": expected, "question": question},
                )
            )
    return plans[: args.limit] if args.limit else plans


def instant_plans(args: argparse.Namespace) -> list[StreamPlan]:
    from motiondrop.benchmarks.streamingbench_data import build_choice_prompt, load_choice_manifest

    annotations, _ = load_choice_manifest(
        args.streamingbench_root, ["realtime_visual_understanding"]
    )
    by_video: dict[str, list[dict]] = defaultdict(list)
    for item in annotations:
        by_video[item["video_path"]].append(item)
    plans = []
    for video, items in by_video.items():
        items = sorted(items, key=lambda item: item["cutoff_seconds"])
        if args.per_video is not None:
            items = items[: args.per_video]
        events = _commentary_events(args, 0.0)
        for item in items:
            events.append(
                StreamEvent(
                    item["cutoff_seconds"],
                    "question",
                    {
                        "id": item["question_id"],
                        "prompt": build_choice_prompt(item["question"], item["options"]),
                        "choices": item["options"],
                        "truth": item["ground_truth"],
                        "task_type": item["task_type"],
                    },
                )
            )
        plans.append(
            StreamPlan(
                f"sample_{items[0]['sample_number']}",
                Path(video),
                0.0,
                max(item["cutoff_seconds"] for item in items),
                events,
            )
        )
    plans.sort(key=lambda plan: int(plan.plan_id.split("_")[1]))
    return plans[: args.limit] if args.limit else plans


def ovo_backward_plans(args: argparse.Namespace) -> list[StreamPlan]:
    from motiondrop.benchmarks.ovobench_data import BACKWARD_TASKS, build_choice_prompt
    from motiondrop.video import probe_video

    annotations = json.loads(args.ovobench_json.read_text(encoding="utf-8"))
    plans = []
    for item in annotations:
        if item["task"] not in BACKWARD_TASKS:
            continue
        video = args.ovobench_videos / f"{item['id']}.mp4"
        if not video.is_file():
            continue
        total, fps = probe_video(video)
        end = (total - 1) / fps
        events = _commentary_events(args, 0.0)
        events.append(
            StreamEvent(
                end,
                "question",
                {
                    "id": str(item["id"]),
                    "prompt": build_choice_prompt(item["task"], item["question"], item["options"]),
                    "choices": item["options"],
                    "truth": chr(65 + int(item["gt"])),
                    "task_type": item["task"],
                },
            )
        )
        plans.append(
            StreamPlan(f"ovo_{item['id']}", video, 0.0, end, events, {"task": item["task"]})
        )
    return plans[: args.limit] if args.limit else plans


def realtime_plans(args: argparse.Namespace) -> list[StreamPlan]:
    root = args.streamingbench_root / "videos" / "proactive_output"
    videos = sorted(
        root.glob("sample_*/video.mp4"), key=lambda path: int(path.parent.name.split("_")[1])
    )
    # Sports broadcasts (snooker, basketball, tennis) as in the target scenario.
    videos = [video for video in videos if 3 <= int(video.parent.name.split("_")[1]) <= 24]
    plans = []
    for video in videos[:: max(1, len(videos) // 6)][:6]:
        events = _commentary_events(args, 0.0)
        moment = args.question_interval
        index = 0
        while moment < args.realtime_seconds:
            events.append(
                StreamEvent(
                    moment, "question", {"id": f"q{index}", "prompt": REALTIME_QUESTIONS[index % 3]}
                )
            )
            moment += args.question_interval
            index += 1
        plans.append(StreamPlan(video.parent.name, video, 0.0, args.realtime_seconds, events))
    return plans[: args.limit] if args.limit else plans


PLAN_BUILDERS = {
    "proactive": proactive_plans,
    "instant": instant_plans,
    "ovo-backward": ovo_backward_plans,
    "realtime": realtime_plans,
}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _windows(plan: StreamPlan, args: argparse.Namespace):
    from motiondrop.video import probe_video

    total, fps = probe_video(plan.video)
    last = min(total - 1, int(plan.end_seconds * fps))
    first = int(plan.start_seconds * fps)
    indices = [
        index + first
        for index in sampled_indices(
            last - first + 1, fps, target_fps=args.fps, query_at_seconds=None, maximum=None
        )
    ]
    chunks = [
        indices[start : start + args.window_frames]
        for start in range(0, len(indices), args.window_frames)
    ]
    return chunks, fps


def _add_task(scheduler: LiveScheduler, event: StreamEvent) -> str:
    payload = dict(event.payload)
    payload.pop("probe", None)
    kind = payload.pop("kind")
    instruction = payload.pop("instruction")
    return scheduler.add_task(kind, instruction, event.seconds, **payload)["id"]


def run_plan(model, processor, args: argparse.Namespace, plan: StreamPlan) -> dict:
    from motiondrop.streaming_runtime.llava import LlavaStreamingBackend
    from motiondrop.video import SequentialFrameReader

    realtime = args.suite == "realtime"
    backend = LlavaStreamingBackend(model, processor, config=runtime_config(args))
    scheduler = LiveScheduler(
        backend, config=live_config(args, "realtime" if realtime else "lockstep")
    )
    chunks, fps = _windows(plan, args)
    reader = SequentialFrameReader(plan.video, start_frame=chunks[0][0])
    events = sorted(plan.events, key=lambda event: event.seconds)
    task_ids: dict[str, str] = {}
    answers: list[dict] = []
    pending: list[threading.Thread] = []
    started = time.perf_counter()
    scheduler.start()

    def handle(event: StreamEvent) -> None:
        if event.kind == "task":
            task_id = _add_task(scheduler, event)
            if event.payload.get("probe"):
                task_ids["probe"] = task_id
            return
        payload = event.payload
        if not realtime:
            scheduler.wait_processed(scheduler.submitted_windows)

        def ask() -> None:
            result = scheduler.ask(
                payload["prompt"],
                choices=payload.get("choices"),
                timeout=600,
                record=False,
            )
            answers.append(
                {"id": payload["id"], "seconds": event.seconds, **payload, **result, "prompt": None}
            )

        if realtime:
            thread = threading.Thread(target=ask)
            thread.start()
            pending.append(thread)
        else:
            ask()

    try:
        cursor = 0
        for chunk in chunks:
            window_start = chunk[0] / fps
            while cursor < len(events) and events[cursor].seconds < window_start:
                handle(events[cursor])
                cursor += 1
            frames, decoded = reader.read(chunk)
            if realtime:
                due = started + (decoded[-1] / fps - plan.start_seconds)
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            scheduler.submit(frames, decoded, [index / fps for index in decoded])
        if not realtime:
            scheduler.wait_processed(scheduler.submitted_windows)
        while cursor < len(events):
            handle(events[cursor])
            cursor += 1
        for thread in pending:
            thread.join(timeout=600)
        scheduler.close(timeout=600)
    finally:
        reader.close()
        if scheduler.errors or not scheduler.stopping:
            scheduler.abort()
    if scheduler.errors:
        raise RuntimeError(scheduler.errors[0]["traceback"])
    tasks = {
        task.id: {
            **task.public(),
            "outputs": [
                {
                    k: v
                    for k, v in output.items()
                    if k in ("seconds", "text", "lag_seconds", "p_yes")
                }
                for output in task.outputs
            ],
            "check_trace": [[round(t, 2), round(p, 4)] for t, p in task.check_trace],
        }
        for task in scheduler.tasks.values()
    }
    window_log = scheduler.window_log
    return {
        "plan_id": plan.plan_id,
        "video": str(plan.video),
        "start_seconds": plan.start_seconds,
        "end_seconds": plan.end_seconds,
        "meta": plan.meta,
        "answers": sorted(answers, key=lambda item: item["seconds"]),
        "tasks": tasks,
        "probe_task": task_ids.get("probe"),
        "windows": len(window_log),
        "mean_encode_ms": 1000 * statistics.fmean(w["encode_seconds"] for w in window_log)
        if window_log
        else None,
        "mean_prefill_ms": 1000 * statistics.fmean(w["prefill_seconds"] for w in window_log)
        if window_log
        else None,
        "max_arrival_to_prefilled_s": max(
            (w["arrival_to_prefilled_seconds"] for w in window_log), default=None
        ),
        "trunk": backend.trunk_stats(),
        "gpu_busy_seconds": scheduler.gpu_busy_seconds,
        "wall_seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _percentile(values, fraction: float) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(fraction * (len(values) - 1))))]


def summarize(suite: str, records: list[dict]) -> dict:
    summary: dict = {"suite": suite, "streams": len(records)}
    standing = [
        task
        for record in records
        for task in record["tasks"].values()
        if task["kind"] == "standing"
    ]
    if standing:
        outputs = [output for task in standing for output in task["outputs"]]
        summary["standing"] = {
            "tasks": len(standing),
            "runs": sum(task["runs"] for task in standing),
            "outputs": len(outputs),
            "silent": sum(task["silent"] for task in standing),
            "duplicates": sum(task["duplicates"] for task in standing),
            "preempted": sum(task["preempted"] for task in standing),
            "dropped": sum(task["dropped"] for task in standing),
            "mean_lag_seconds": _mean(output["lag_seconds"] for output in outputs),
            "p95_lag_seconds": _percentile([output["lag_seconds"] for output in outputs], 0.95),
        }
    if suite == "proactive":
        time_ok = answer_ok = fired = 0
        offsets = []
        for record in records:
            probe = record["tasks"].get(record["probe_task"] or "", {})
            outputs = probe.get("outputs", [])
            if not outputs:
                continue
            fired += 1
            first = outputs[0]
            offset = first["seconds"] - record["meta"]["truth"]
            offsets.append(offset)
            if -2.0 <= offset <= 2.0:
                time_ok += 1
                answer_ok += int(record["meta"]["expected"] in first["text"])
        count = len(records)
        summary["proactive"] = {
            "questions": count,
            "fired": fired,
            "time_accuracy": time_ok / count if count else None,
            "answer_accuracy": answer_ok / count if count else None,
            "early_fires": sum(offset < -2.0 for offset in offsets),
            "late_fires": sum(offset > 2.0 for offset in offsets),
            "median_offset_seconds": statistics.median(offsets) if offsets else None,
        }
    answers = [answer for record in records for answer in record["answers"]]
    if answers and "truth" in answers[0]:
        by_type: dict[str, list[bool]] = defaultdict(list)
        for answer in answers:
            by_type[answer["task_type"]].append(answer["answer"] == answer["truth"])
        flat = [value for values in by_type.values() for value in values]
        summary["accuracy"] = {
            "questions": len(flat),
            "overall": sum(flat) / len(flat),
            "by_type": {
                key: {"n": len(v), "accuracy": sum(v) / len(v)}
                for key, v in sorted(by_type.items())
            },
            "memory_used_mean": _mean(answer.get("memory_used") for answer in answers),
        }
    if answers:
        summary["latency"] = {
            "questions": len(answers),
            "mean_latency_seconds": _mean(answer["latency_seconds"] for answer in answers),
            "p95_latency_seconds": _percentile(
                [answer["latency_seconds"] for answer in answers], 0.95
            ),
            "mean_wait_seconds": _mean(answer["wait_seconds"] for answer in answers),
            "p95_wait_seconds": _percentile([answer["wait_seconds"] for answer in answers], 0.95),
        }
    summary["runtime"] = {
        "mean_encode_ms": _mean(record["mean_encode_ms"] for record in records),
        "mean_prefill_ms": _mean(record["mean_prefill_ms"] for record in records),
        "max_arrival_to_prefilled_s": max(
            (record["max_arrival_to_prefilled_s"] or 0.0 for record in records), default=None
        ),
        "evicted_streams": sum(record["trunk"].get("evicted_tokens", 0) > 0 for record in records),
        "gpu_busy_per_video_second": (
            sum(record["gpu_busy_seconds"] for record in records)
            / max(1e-9, sum(record["end_seconds"] - record["start_seconds"] for record in records))
        ),
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.only:
        wanted = [item.strip() for item in args.only.split(",") if item.strip()]
        limit, args.limit = args.limit, None
        plans = {plan.plan_id: plan for plan in PLAN_BUILDERS[args.suite](args)}
        missing = [item for item in wanted if item not in plans]
        if missing:
            raise SystemExit(f"unknown stream IDs: {missing}")
        plans = [plans[item] for item in wanted][:limit]
    else:
        plans = PLAN_BUILDERS[args.suite](args)
    output = args.output_dir / args.suite / f"{args.name}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done[record["plan_id"]] = record
    todo = [plan for plan in plans if plan.plan_id not in done]
    print(
        f"[live-eval] {args.suite}/{args.name}: {len(plans)} streams, {len(todo)} to run",
        flush=True,
    )
    if todo:
        from motiondrop.models import load_model_and_processor
        from motiondrop.streaming_runtime.llava import resolve_local_checkpoint

        model, processor, _spec, _checkpoint = load_model_and_processor(
            "llava-ov-7b",
            resolve_local_checkpoint(args.model_path),
            max_gpu_memory_gib=args.max_gpu_memory_gib,
            local_files_only=True,
        )
        for number, plan in enumerate(todo, 1):
            try:
                record = run_plan(model, processor, args, plan)
            except Exception:
                print(f"[live-eval] {plan.plan_id} failed:\n{traceback.format_exc()}", flush=True)
                continue
            done[plan.plan_id] = record
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[live-eval] {number}/{len(todo)} {plan.plan_id} windows={record['windows']} "
                f"wall={record['wall_seconds']:.1f}s",
                flush=True,
            )
    records = [done[plan.plan_id] for plan in plans if plan.plan_id in done]
    summary = {
        "name": args.name,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        **summarize(args.suite, records),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
