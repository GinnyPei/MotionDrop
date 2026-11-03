#!/usr/bin/env python3
"""Render live-eval results as a per-sample Markdown report.

Every sample lists its video path, question, ground truth and the model's
answer under each condition found in ``results/live-eval/<suite>/*.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROACTIVE_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)


def load(suite_dir: Path) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(suite_dir.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        runs[path.stem] = {record["plan_id"]: record for record in records}
    return runs


def summary(suite_dir: Path, name: str) -> dict:
    path = suite_dir / f"{name}.summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def real(path: str) -> str:
    return str(Path(path).resolve())


def clock(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    whole = int(round(seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def cell(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).replace("|", "\\|").strip()


def question_text(prompt: str) -> str:
    """The question line of a multiple-choice prompt."""

    match = re.search(r"Question:\s*(.+?)(?:\n|Options:)", prompt, re.S)
    return cell(match.group(1) if match else prompt.splitlines()[0])


def standing_lines(record: dict, limit: int = 6) -> list[str]:
    lines = []
    for task in record["tasks"].values():
        if task["kind"] == "standing" and task["outputs"]:
            for output in task["outputs"][:limit]:
                lines.append(f"[{clock(output['seconds'])}] {cell(output['text'])}")
            break
    return lines


def simulate_watch(trace: list, threshold: float) -> float | None:
    for seconds, p_yes in trace:
        if p_yes >= threshold:
            return seconds
    return None


def proactive_section(suite_dir: Path) -> list[str]:
    runs = load(suite_dir)
    if not runs:
        return []
    out = ["## Proactive Output（条件触发任务，StreamingBench）", ""]
    out.append(
        "评分沿用官方规则：首次触发时间与真值相差 ±2 s 内算时间正确，且输出包含期望文本算答案正确。"
    )
    out.append("")
    trace = runs.get("trace", {})
    if trace:
        out += ["### p(yes) 轨迹与阈值模拟（watch-only，阈值设为 0.999 只记录不触发）", ""]
        header = "| 阈值 | " + " | ".join(str(t) for t in PROACTIVE_THRESHOLDS) + " |"
        out += [header, "|---" * (len(PROACTIVE_THRESHOLDS) + 1) + "|"]
        hits = []
        for threshold in PROACTIVE_THRESHOLDS:
            ok = 0
            for record in trace.values():
                fired = simulate_watch(record["tasks"][record["probe_task"]]["check_trace"], threshold)
                ok += fired is not None and abs(fired - record["meta"]["truth"]) <= 2.0
            hits.append(f"{ok}/{len(trace)}")
        out += ["| 时间正确 | " + " | ".join(hits) + " |", ""]
        variants = [name for name in runs if name.startswith("trace") and name != "trace"]
        if variants:
            out += [
                "### p(yes) 轨迹对比",
                "",
                "与 watch-only（90% 压缩，无其他任务）的轨迹逐点比较。"
                "“最大 |Δp|”为 0 说明其他任务对该 watch 任务完全没有影响。",
                "",
                "| 样本 | " + " | ".join(f"{name}：最大 |Δp| / 真值±2s 最大 p" for name in variants) + " |",
                "|---" * (len(variants) + 1) + "|",
            ]
            for plan_id, base in trace.items():
                base_trace = base["tasks"][base["probe_task"]]["check_trace"]
                truth = base["meta"]["truth"]
                cells = []
                for name in variants:
                    other = runs[name].get(plan_id)
                    if other is None:
                        cells.append("—")
                        continue
                    other_trace = other["tasks"][other["probe_task"]]["check_trace"]
                    delta = max(
                        (abs(a[1] - b[1]) for a, b in zip(base_trace, other_trace)), default=0.0
                    )
                    near = [p for t, p in other_trace if abs(t - truth) <= 2.0]
                    cells.append(f"{delta:.3f} / {max(near):.2f}" if near else f"{delta:.3f} / —")
                out.append(f"| {plan_id} | " + " | ".join(cells) + " |")
            out.append("")
    for plan_id, record in next(iter(runs.values())).items():
        meta = record["meta"]
        out += [
            f"### {plan_id}",
            "",
            f"- 视频：`{real(record['video'])}`",
            f"- 问题：{cell(meta['question'])}",
            f"- 提问时刻 {clock(meta['asked'])}，真值时刻 {clock(meta['truth'])}，期望输出 `{meta['expected']}`",
            f"- 推流区间 {clock(record['start_seconds'])}–{clock(record['end_seconds'])}",
        ]
        if plan_id in trace:
            points = trace[plan_id]["tasks"][trace[plan_id]["probe_task"]]["check_trace"]
            peak = max(points, key=lambda item: item[1]) if points else None
            near = [p for t, p in points if abs(t - meta["truth"]) <= 2.0]
            out.append(
                f"- p(yes)：首次检查 {points[0][1]:.2f}，峰值 {peak[1]:.2f} @ {clock(peak[0])}，"
                f"真值 ±2 s 内最大 {max(near):.2f}" if points and near else "- p(yes)：无检查"
            )
        conditions = [name for name in runs if not name.startswith("trace") and plan_id in runs[name]]
        if conditions:
            out += ["", "| 条件 | 触发时刻 | 偏差 | 输出 | 时间正确 | 答案正确 |", "|---|---|---|---|---|---|"]
        for name, records in runs.items():
            if name.startswith("trace") or plan_id not in records:
                continue
            run = records[plan_id]
            outputs = run["tasks"][run["probe_task"]]["outputs"]
            if outputs:
                first = outputs[0]
                offset = first["seconds"] - meta["truth"]
                time_ok = abs(offset) <= 2.0
                answer_ok = time_ok and meta["expected"] in first["text"]
                out.append(
                    f"| {name} | {clock(first['seconds'])} | {offset:+.1f}s | {cell(first['text'])} | "
                    f"{'✓' if time_ok else '✗'} | {'✓' if answer_ok else '✗'} |"
                )
            else:
                out.append(f"| {name} | 未触发 | — | — | ✗ | ✗ |")
        out.append("")
    return out


def choice_section(title: str, suite_dir: Path, note: str) -> list[str]:
    runs = load(suite_dir)
    if not runs:
        return []
    names = list(runs)
    out = [f"## {title}", "", note, ""]
    out += ["| 条件 | 正确 / 题数 | 平均延迟 | 平均引用记忆 |", "|---|---|---|---|"]
    for name in names:
        info = summary(suite_dir, name)
        accuracy = info.get("accuracy", {})
        latency = info.get("latency", {})
        if accuracy:
            correct = round(accuracy["overall"] * accuracy["questions"])
            out.append(
                f"| {name} | {correct}/{accuracy['questions']} | "
                f"{latency.get('mean_latency_seconds', 0):.2f}s | {accuracy.get('memory_used_mean') or 0:.1f} |"
            )
    out.append("")
    first = runs[names[0]]
    for plan_id, record in first.items():
        out += [f"### {plan_id}", "", f"- 视频：`{real(record['video'])}`"]
        commentary = None
        for name in names:
            if plan_id in runs[name]:
                commentary = commentary or standing_lines(runs[name][plan_id])
        for index, answer in enumerate(record["answers"]):
            out.append("")
            out.append(f"**Q{index + 1} @ {clock(answer['seconds'])}**（{answer['task_type']}）：{question_text(answer['question'])}")
            options = " / ".join(f"{chr(65 + i)}. {cell(o)}" for i, o in enumerate(answer["choices"]))
            out.append(f"- 选项：{options}")
            out.append(f"- 正确答案：**{answer['truth']}**")
            for name in names:
                run = runs[name].get(plan_id)
                if not run or index >= len(run["answers"]):
                    continue
                got = run["answers"][index]
                mark = "✓" if got["answer"] == got["truth"] else "✗"
                memory = f"，引用记忆 {got['memory_used']} 条" if got.get("memory_used") else ""
                out.append(f"- {name}：{got['answer']} {mark}（{got['latency_seconds']:.2f}s{memory}）")
        if commentary:
            out += ["", "<details><summary>同一视频上并行解说任务的前几条输出</summary>", ""]
            out += [f"- {line}" for line in commentary]
            out += ["", "</details>"]
        out.append("")
    return out


def realtime_section(suite_dir: Path) -> list[str]:
    runs = load(suite_dir)
    if not runs:
        return []
    out = ["## Realtime（按真实时间回放，测干扰与延迟）", ""]
    out += [
        "| 条件 | 提问平均延迟 | 提问 p95 延迟 | 平均排队 | 解说条数 | 解说平均滞后 | 被抢占 | 丢弃 tick | GPU 占用/视频秒 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in runs:
        info = summary(suite_dir, name)
        latency = info.get("latency", {})
        standing = info.get("standing", {})
        runtime = info.get("runtime", {})
        out.append(
            f"| {name} | {latency.get('mean_latency_seconds', 0):.2f}s | {latency.get('p95_latency_seconds', 0):.2f}s | "
            f"{latency.get('mean_wait_seconds', 0):.2f}s | {standing.get('outputs', '—')} | "
            f"{(standing.get('mean_lag_seconds') or 0):.2f}s | {standing.get('preempted', '—')} | "
            f"{standing.get('dropped', '—')} | {runtime.get('gpu_busy_per_video_second', 0):.2f} |"
        )
    out.append("")
    first = next(iter(runs.values()))
    for plan_id, record in first.items():
        out += [f"### {plan_id}", "", f"- 视频：`{real(record['video'])}`（0–{clock(record['end_seconds'])}）", ""]
        for name, records in runs.items():
            run = records.get(plan_id)
            if not run:
                continue
            out.append(f"**{name}**")
            for answer in run["answers"]:
                out.append(
                    f"- [{clock(answer['seconds'])}] 问：{answer['question']} → 答：{cell(answer['answer'])}"
                    f"（{answer['latency_seconds']:.2f}s，排队 {answer['wait_seconds']:.2f}s）"
                )
            for task in run["tasks"].values():
                if task["kind"] == "standing" and task["outputs"]:
                    lines = "；".join(
                        f"[{clock(o['seconds'])}] {cell(o['text'])}" for o in task["outputs"][:5]
                    )
                    out.append(f"- 持续任务“{task['instruction']}”前 5 条：{lines}")
            out.append("")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/live-eval"))
    parser.add_argument("--output", type=Path, default=Path("results/live-eval/REPORT.md"))
    parser.add_argument("--preface", type=Path, help="Markdown inserted after the title.")
    args = parser.parse_args()
    lines = ["# Live 多任务运行时小样本评测", ""]
    if args.preface is not None:
        lines += [args.preface.read_text(encoding="utf-8").rstrip(), ""]
    lines += proactive_section(args.results / "proactive")
    lines += choice_section(
        "StreamingBench Real-Time Visual Understanding（瞬时选择题）",
        args.results / "instant",
        "每个视频一条流，问题在各自时间戳提出。base = 只有提问；commentary = 同时运行解说任务（分支隔离）；"
        "commentary-shared = 解说和回答都写进主干 KV（旧设计）。",
    )
    lines += choice_section(
        "OVO-Bench backward（回溯题，检验 KV 淘汰与记忆）",
        args.results / "ovo-backward",
        "问题在片段末尾提出。full = 不淘汰；evict1024 = 主干超过 1024 token 时淘汰最旧窗口（连续位置重旋转）；"
        "-absolute = 淘汰后保留原位置（有位置间隔）；-memory = 淘汰 + 每 4 s 后台记笔记供提问检索。",
    )
    lines += realtime_section(args.results / "realtime")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
