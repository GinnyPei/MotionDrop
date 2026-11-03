"""MotionDrop evaluation on OVO-Bench."""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from motiondrop.benchmarks.common import (
    add_runtime_arguments,
    ensure_run_config,
    file_sha256,
    finalize_runtime_arguments,
    load_adapter,
    runtime_config,
)
from motiondrop.benchmarks.ovobench_data import (
    ALL_TASKS,
    BACKWARD_TASKS,
    FORWARD_TASKS,
    REALTIME_TASKS,
    build_choice_prompt,
    build_forward_prompt,
    load_video_frames,
)
from motiondrop.models import get_model_spec
from motiondrop.paths import ovobench_annotations, ovobench_videos
from motiondrop.runtime import (
    DEFAULT_ANSWER_LABELS,
    append_jsonl,
    compression_tag,
    load_jsonl,
    make_video_conversation,
    set_reproducibility,
    validate_compression_stats,
    write_json,
)


@dataclass(frozen=True)
class InferenceUnit:
    unit_id: str
    sample_id: int
    category: str
    task: str
    video_path: str
    prompt: str
    annotation: dict
    test_info_index: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motiondrop ovobench",
        description="Run MotionDrop on OVO-Bench's causal video clips.",
    )
    add_runtime_arguments(parser, "ovobench")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ovobench_annotations(),
        help="Annotation JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--videos",
        type=Path,
        default=ovobench_videos(),
        help="Chunked MP4 directory (default: %(default)s).",
    )
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=list(ALL_TASKS))
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Initial clip sampling rate before the uniform --max-frames cap.",
    )
    parser.add_argument(
        "--limit-annotations",
        type=int,
        default=None,
        help="Use only the first N annotation rows before expanding forward tasks.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum expanded inference units.")
    parser.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help="Permit an intentional partial-data run.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.limit_annotations is not None and args.limit_annotations < 1:
        parser.error("--limit-annotations must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return finalize_runtime_arguments(parser, args, "ovobench")


def build_units(
    annotations: list[dict],
    videos: Path,
    tasks: set[str],
) -> tuple[list[InferenceUnit], list[str]]:
    units: list[InferenceUnit] = []
    missing_videos: list[str] = []
    for annotation in annotations:
        sample_id = int(annotation["id"])
        task = annotation["task"]
        if task not in tasks:
            continue
        if task in BACKWARD_TASKS + REALTIME_TASKS:
            video_path = videos / f"{sample_id}.mp4"
            if not video_path.is_file():
                missing_videos.append(str(video_path))
                continue
            units.append(
                InferenceUnit(
                    unit_id=f"{sample_id}:-1",
                    sample_id=sample_id,
                    category="backward" if task in BACKWARD_TASKS else "realtime",
                    task=task,
                    video_path=str(video_path),
                    prompt=build_choice_prompt(
                        task,
                        annotation["question"],
                        annotation["options"],
                    ),
                    annotation=annotation,
                    test_info_index=None,
                )
            )
        elif task in FORWARD_TASKS:
            for index in range(len(annotation.get("test_info", []))):
                video_path = videos / f"{sample_id}_{index}.mp4"
                if not video_path.is_file():
                    missing_videos.append(str(video_path))
                    continue
                units.append(
                    InferenceUnit(
                        unit_id=f"{sample_id}:{index}",
                        sample_id=sample_id,
                        category="forward",
                        task=task,
                        video_path=str(video_path),
                        prompt=build_forward_prompt(task, annotation, index),
                        annotation=annotation,
                        test_info_index=index,
                    )
                )
        else:
            raise ValueError(f"Unknown OVO-Bench task: {task}")
    return units, missing_videos


def allowed_labels(unit: InferenceUnit) -> list[str]:
    if unit.task in BACKWARD_TASKS + REALTIME_TASKS:
        option_count = len(unit.annotation["options"])
        if not 2 <= option_count <= 5:
            raise ValueError(
                f"OVO-Bench choice question {unit.unit_id} has {option_count} options; "
                "expected 2 through 5"
            )
        return list("ABCDE"[:option_count])
    if unit.task in {"SSR", "CRR"}:
        return ["Yes", "No"]
    if unit.task == "REC":
        return [str(index) for index in range(11)]
    raise ValueError(unit.task)


def ground_truth(unit: InferenceUnit) -> str:
    if unit.task in BACKWARD_TASKS + REALTIME_TASKS:
        return chr(65 + int(unit.annotation["gt"]))
    info = unit.annotation["test_info"][unit.test_info_index]
    if unit.task == "REC":
        return str(int(info["count"]))
    return "Yes" if int(info["type"]) == 1 else "No"


def score_units(records: list[dict]) -> dict:
    by_task: dict[str, list[bool]] = defaultdict(list)
    by_category: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        correct = record["response"] == record["ground_truth"]
        by_task[record["task"]].append(correct)
        by_category[record["category"]].append(correct)

    def summarize(values: list[bool]) -> dict:
        correct = sum(values)
        return {"completed": len(values), "correct": correct, "accuracy": correct / len(values)}

    task_scores = {task: summarize(values) for task, values in sorted(by_task.items())}
    category_scores = {}
    for category, values in sorted(by_category.items()):
        task_names = sorted(
            {record["task"] for record in records if record["category"] == category}
        )
        summary = summarize(values)
        summary["macro_task_accuracy"] = sum(
            task_scores[name]["accuracy"] for name in task_names
        ) / len(task_names)
        category_scores[category] = summary
    all_values = [value for values in by_task.values() for value in values]
    return {
        "overall_micro": summarize(all_values),
        "by_category": category_scores,
        "by_task": task_scores,
    }


def assemble_official(annotations: list[dict], records: list[dict]) -> dict:
    index = {record["unit_id"]: record for record in records}
    output = {"backward": [], "realtime": [], "forward": []}
    for annotation in annotations:
        sample_id = int(annotation["id"])
        task = annotation["task"]
        if task in BACKWARD_TASKS + REALTIME_TASKS:
            record = index.get(f"{sample_id}:-1")
            if record is None:
                continue
            output[record["category"]].append(
                {
                    "id": sample_id,
                    "video": annotation.get("video"),
                    "task": task,
                    "question": annotation.get("question"),
                    "response": record["response"],
                    "ground_truth": record["ground_truth"],
                    "sampled_frames": record["sampled_frames"],
                    "compression_stats": record["compression_stats"],
                    "candidate_score_mode": record["candidate_score_mode"],
                    "candidate_scores": record["candidate_scores"],
                    "candidate_probs": record["candidate_probs"],
                }
            )
        elif task in FORWARD_TASKS:
            item = copy.deepcopy(annotation)
            any_record = False
            for test_index, test_info in enumerate(item.get("test_info", [])):
                record = index.get(f"{sample_id}:{test_index}")
                if record is None:
                    continue
                any_record = True
                test_info.update(
                    {
                        "response": record["response"],
                        "sampled_frames": record["sampled_frames"],
                        "compression_stats": record["compression_stats"],
                        "candidate_score_mode": record["candidate_score_mode"],
                        "candidate_scores": record["candidate_scores"],
                        "candidate_probs": record["candidate_probs"],
                    }
                )
            if any_record:
                output["forward"].append(item)
    return output


def run(args: argparse.Namespace) -> None:
    set_reproducibility(args.seed)
    with args.annotations.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, list):
        raise ValueError("OVO-Bench annotation file must contain a JSON list")
    if args.limit_annotations is not None:
        annotations = annotations[: args.limit_annotations]

    units, missing_videos = build_units(annotations, args.videos, set(args.tasks))
    if missing_videos and not args.allow_missing_videos:
        raise FileNotFoundError(
            f"Missing {len(missing_videos)} OVO-Bench clips; first paths: {missing_videos[:5]}. "
            "Use --allow-missing-videos only for an intentional partial run."
        )
    if args.limit is not None:
        units = units[: args.limit]
    if not units:
        raise ValueError("OVO-Bench inputs produced no inference units")
    for unit in units:
        labels = allowed_labels(unit)
        truth = ground_truth(unit)
        if truth not in labels:
            raise ValueError(
                f"Ground truth {truth!r} is outside the supported candidates "
                f"for OVO-Bench unit {unit.unit_id}: {labels}"
            )

    manifest = {
        "benchmark": "OVO-Bench",
        "annotations": len(annotations),
        "inference_units": len(units),
        "tasks": sorted({unit.task for unit in units}),
        "missing_video_count": len(missing_videos),
        "missing_video_examples": missing_videos[:10],
        "max_frames": args.max_frames,
        "compression_rates": args.compression_rates,
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if args.manifest_only:
        return

    run_config = runtime_config(args, "ovobench") | {
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": file_sha256(args.annotations),
        "videos": str(args.videos.resolve()),
        "tasks": args.tasks,
        "fps": args.fps,
        "unit_ids": [unit.unit_id for unit in units],
    }
    ensure_run_config(args.output, run_config)
    paths = {
        rate: args.output / compression_tag(rate) / "units.jsonl" for rate in args.compression_rates
    }
    existing: dict[float, list[dict]] = {}
    completed: dict[float, set[tuple]] = {}
    for rate, path in paths.items():
        existing[rate], completed[rate] = load_jsonl(path, ["unit_id"])
    pending = [
        unit
        for unit in units
        if any((unit.unit_id,) not in completed[rate] for rate in args.compression_rates)
    ]
    print(f"Pending inference units: {len(pending)}/{len(units)}", flush=True)

    spec = get_model_spec(args.model)
    failures: list[dict] = []
    checkpoint = args.model_path or spec.default_checkpoint
    model_family = spec.family
    adapter = None
    if pending:
        adapter, spec, checkpoint = load_adapter(args)
        model_family = spec.family
        label_to_column = {label: index for index, label in enumerate(DEFAULT_ANSWER_LABELS)}

        def infer_group(group: list[tuple[InferenceUnit, list, list[int]]]) -> None:
            try:
                conversations = [
                    make_video_conversation(
                        frames,
                        unit.prompt,
                        min_pixels=args.min_pixels,
                        max_pixels=args.max_pixels,
                    )
                    for unit, frames, _ in group
                ]
                for rate in args.compression_rates:
                    started = time.perf_counter()
                    scores, batch_stats = adapter.infer(
                        conversations,
                        target_compression=rate,
                        sampled_frame_counts=[len(frames) for _, frames, _ in group],
                        candidate_labels=[allowed_labels(unit) for unit, _, _ in group],
                    )
                    elapsed = time.perf_counter() - started
                    new_records = []
                    for row, (unit, _frames, frame_indices), stats in zip(
                        scores, group, batch_stats, strict=True
                    ):
                        if (unit.unit_id,) in completed[rate]:
                            continue
                        validate_compression_stats(stats, rate)
                        labels = allowed_labels(unit)
                        columns = torch.tensor([label_to_column[label] for label in labels])
                        selected_scores = row.index_select(0, columns).float()
                        probabilities = torch.softmax(selected_scores, dim=0)
                        prediction = labels[int(probabilities.argmax().item())]
                        truth = ground_truth(unit)
                        new_records.append(
                            {
                                "schema": "motiondrop_ovobench_unit_v1",
                                "model_alias": args.model,
                                "model_family": model_family,
                                "unit_id": unit.unit_id,
                                "id": unit.sample_id,
                                "category": unit.category,
                                "task": unit.task,
                                "test_info_index": unit.test_info_index,
                                "video": unit.annotation.get("video"),
                                "question": unit.annotation.get("question"),
                                "response": prediction,
                                "ground_truth": truth,
                                "correct": prediction == truth,
                                "candidate_labels": labels,
                                "candidate_token_ids": [
                                    list(adapter.answer_token_sequences[label]) for label in labels
                                ],
                                "candidate_score_mode": "mean_token_log_probability",
                                "candidate_scores": [
                                    float(value) for value in selected_scores.tolist()
                                ],
                                "candidate_probs": [
                                    float(value) for value in probabilities.tolist()
                                ],
                                "sampled_frames": frame_indices,
                                "compression_stats": stats,
                                "batch_size_effective": len(group),
                                "batch_inference_seconds": elapsed,
                            }
                        )
                    append_jsonl(paths[rate], new_records)
                    existing[rate].extend(new_records)
                    completed[rate].update((record["unit_id"],) for record in new_records)
            except Exception as exc:
                if len(group) > 1:
                    torch.cuda.empty_cache()
                    midpoint = len(group) // 2
                    infer_group(group[:midpoint])
                    infer_group(group[midpoint:])
                    return
                unit = group[0][0]
                if args.debug:
                    raise
                failures.append({"unit_id": unit.unit_id, "error": repr(exc)})
                print(f"[error] {unit.unit_id}: {exc}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()

        for start in tqdm(range(0, len(pending), args.batch_size), desc="OVO-Bench"):
            prepared: list[tuple[InferenceUnit, list, list[int]]] = []
            for unit in pending[start : start + args.batch_size]:
                try:
                    frames, frame_indices = load_video_frames(
                        unit.video_path,
                        fps=args.fps,
                        max_frames=args.max_frames,
                    )
                    prepared.append((unit, frames, frame_indices))
                except Exception as exc:
                    if args.debug:
                        raise
                    failures.append({"unit_id": unit.unit_id, "error": repr(exc)})
            if prepared:
                infer_group(prepared)

    expected_ids = {unit.unit_id for unit in units}
    any_missing = False
    for rate in args.compression_rates:
        rate_records = [record for record in existing[rate] if record["unit_id"] in expected_ids]
        actual_ids = [record["unit_id"] for record in rate_records]
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError(f"Duplicate unit IDs for {compression_tag(rate)}")
        missing = sorted(expected_ids - set(actual_ids))
        any_missing = any_missing or bool(missing)
        output_dir = args.output / compression_tag(rate)
        write_json(output_dir / "results.json", assemble_official(annotations, rate_records))
        summary = {
            "schema": "motiondrop_ovobench_summary_v1",
            "benchmark": "OVO-Bench",
            "model_alias": args.model,
            "model": checkpoint,
            "model_family": model_family,
            "method": "motiondrop",
            "target_compression": rate,
            "expected_units": len(units),
            "completed_units": len(rate_records),
            "missing_unit_ids": missing,
            "scores": score_units(rate_records) if rate_records else {},
            "protocol": manifest,
            "failures": [item for item in failures if item["unit_id"] in missing],
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if any_missing:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
