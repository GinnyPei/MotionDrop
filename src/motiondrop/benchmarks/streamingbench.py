"""MotionDrop evaluation on StreamingBench causal prefixes."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
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
from motiondrop.benchmarks.streamingbench_data import (
    DATASET_SPECS,
    build_choice_prompt,
    load_choice_manifest,
    load_question_id_file,
    load_video_prefix_frames,
)
from motiondrop.models import get_model_spec
from motiondrop.paths import streamingbench_root
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motiondrop streamingbench",
        description="Run MotionDrop on causal StreamingBench video prefixes.",
    )
    add_runtime_arguments(parser, "streamingbench")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=streamingbench_root(),
        help="Dataset root (default: %(default)s).",
    )
    parser.add_argument(
        "--question-id-file",
        type=Path,
        default=None,
        help="Optional newline-delimited allowlist of official question IDs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=sorted(DATASET_SPECS),
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Use only the first N selected questions in each dataset group.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum questions after manifest filtering.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("limit", "limit_per_dataset"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return finalize_runtime_arguments(parser, args, "streamingbench")


def score_records(records: list[dict]) -> dict:
    by_dataset: dict[str, list[bool]] = defaultdict(list)
    by_task: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        correct = record["prediction"] == record["ground_truth"]
        by_dataset[record["dataset"]].append(correct)
        by_task[record["task_type"]].append(correct)

    def summarize(values: list[bool]) -> dict:
        correct = sum(values)
        return {"completed": len(values), "correct": correct, "accuracy": correct / len(values)}

    return {
        "overall": summarize(
            [record["prediction"] == record["ground_truth"] for record in records]
        ),
        "by_dataset": {name: summarize(values) for name, values in sorted(by_dataset.items())},
        "by_task": {name: summarize(values) for name, values in sorted(by_task.items())},
    }


def run(args: argparse.Namespace) -> None:
    set_reproducibility(args.seed)
    question_ids = (
        load_question_id_file(args.question_id_file) if args.question_id_file is not None else None
    )
    manifest, manifest_report = load_choice_manifest(
        args.data_root,
        args.datasets,
        limit_per_dataset=args.limit_per_dataset,
        question_ids=question_ids,
    )
    if args.limit is not None:
        manifest = manifest[: args.limit]
        manifest_report["total_rows_after_limit"] = len(manifest)
    if not manifest:
        raise ValueError("StreamingBench inputs produced an empty manifest")
    manifest_report.update(
        {
            "question_id_file": (
                str(args.question_id_file.resolve()) if args.question_id_file is not None else None
            ),
            "protocol": "causal prefix through question timestamp; uniform frames including endpoint",
            "max_frames": args.max_frames,
            "compression_rates": args.compression_rates,
        }
    )
    print(
        json.dumps(
            {"benchmark": "StreamingBench", "manifest": manifest_report},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.manifest_only:
        return

    run_config = runtime_config(args, "streamingbench") | {
        "data_root": str(args.data_root.resolve()),
        "datasets": args.datasets,
        "annotation_csv_sha256": {
            dataset: file_sha256(args.data_root / DATASET_SPECS[dataset].csv_relative_path)
            for dataset in args.datasets
        },
        "question_id_file": (
            str(args.question_id_file.resolve()) if args.question_id_file is not None else None
        ),
        "question_id_file_sha256": (
            file_sha256(args.question_id_file) if args.question_id_file is not None else None
        ),
        "question_ids": [item["question_id"] for item in manifest],
    }
    ensure_run_config(args.output, run_config)
    paths = {
        rate: args.output / compression_tag(rate) / "records.jsonl"
        for rate in args.compression_rates
    }
    existing: dict[float, list[dict]] = {}
    completed: dict[float, set[tuple]] = {}
    for rate, path in paths.items():
        existing[rate], completed[rate] = load_jsonl(path, ["question_id"])
    pending = [
        annotation
        for annotation in manifest
        if any(
            (annotation["question_id"],) not in completed[rate] for rate in args.compression_rates
        )
    ]
    print(f"Pending questions: {len(pending)}/{len(manifest)}", flush=True)

    spec = get_model_spec(args.model)
    checkpoint = args.model_path or spec.default_checkpoint
    failures: list[dict] = []
    adapter = None
    if pending:
        adapter, spec, checkpoint = load_adapter(args)
        option_columns = torch.tensor([DEFAULT_ANSWER_LABELS.index(label) for label in "ABCD"])

        def infer_group(group: list[tuple[dict, list, list[int], dict]]) -> None:
            try:
                conversations = [
                    make_video_conversation(
                        frames,
                        build_choice_prompt(annotation["question"], annotation["options"]),
                        min_pixels=args.min_pixels,
                        max_pixels=args.max_pixels,
                    )
                    for annotation, frames, _, _ in group
                ]
                for rate in args.compression_rates:
                    started = time.perf_counter()
                    scores, batch_stats = adapter.infer(
                        conversations,
                        target_compression=rate,
                        sampled_frame_counts=[len(frames) for _, frames, _, _ in group],
                        candidate_labels=[list("ABCD")] * len(group),
                    )
                    elapsed = time.perf_counter() - started
                    new_records = []
                    for row, group_item, stats in zip(scores, group, batch_stats, strict=True):
                        annotation, _frames, frame_indices, video_metadata = group_item
                        question_id = annotation["question_id"]
                        if (question_id,) in completed[rate]:
                            continue
                        validate_compression_stats(stats, rate)
                        selected_scores = row.index_select(0, option_columns).float()
                        probabilities = torch.softmax(selected_scores, dim=0)
                        prediction = chr(65 + int(probabilities.argmax().item()))
                        new_records.append(
                            {
                                "schema": "motiondrop_streamingbench_record_v1",
                                "model_alias": args.model,
                                "model_family": spec.family,
                                **{
                                    key: annotation[key]
                                    for key in (
                                        "question_id",
                                        "dataset",
                                        "task_type",
                                        "sample_number",
                                        "question_number",
                                        "question",
                                        "time_stamp",
                                        "cutoff_seconds",
                                        "ground_truth",
                                        "options",
                                        "frames_required",
                                        "temporal_clue_type",
                                    )
                                },
                                "prediction": prediction,
                                "correct": prediction == annotation["ground_truth"],
                                "option_token_ids": [
                                    list(adapter.answer_token_sequences[label]) for label in "ABCD"
                                ],
                                "option_score_mode": "mean_token_log_probability",
                                "option_scores": [
                                    float(value) for value in selected_scores.tolist()
                                ],
                                "option_probs": [float(value) for value in probabilities.tolist()],
                                "frame_indices": frame_indices,
                                "video_metadata": video_metadata,
                                "compression_stats": stats,
                                "batch_size_effective": len(group),
                                "batch_inference_seconds": elapsed,
                            }
                        )
                    append_jsonl(paths[rate], new_records)
                    existing[rate].extend(new_records)
                    completed[rate].update((record["question_id"],) for record in new_records)
            except Exception as exc:
                if len(group) > 1:
                    torch.cuda.empty_cache()
                    midpoint = len(group) // 2
                    infer_group(group[:midpoint])
                    infer_group(group[midpoint:])
                    return
                question_id = group[0][0]["question_id"]
                if args.debug:
                    raise
                failures.append({"question_id": question_id, "error": repr(exc)})
                print(f"[error] {question_id}: {exc}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()

        for start in tqdm(
            range(0, len(pending), args.batch_size),
            desc="StreamingBench",
        ):
            prepared: list[tuple[dict, list, list[int], dict]] = []
            for annotation in pending[start : start + args.batch_size]:
                try:
                    frames, frame_indices, video_metadata = load_video_prefix_frames(
                        annotation["video_path"],
                        annotation["cutoff_seconds"],
                        max_frames=args.max_frames,
                    )
                    prepared.append((annotation, frames, frame_indices, video_metadata))
                except Exception as exc:
                    if args.debug:
                        raise
                    failures.append({"question_id": annotation["question_id"], "error": repr(exc)})
            if prepared:
                infer_group(prepared)

    expected_ids = {annotation["question_id"] for annotation in manifest}
    any_missing = False
    for rate in args.compression_rates:
        rate_records = [
            record for record in existing[rate] if record["question_id"] in expected_ids
        ]
        ids = [record["question_id"] for record in rate_records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate question IDs for {compression_tag(rate)}")
        missing = sorted(expected_ids - set(ids))
        any_missing = any_missing or bool(missing)
        rate_records.sort(
            key=lambda item: (
                item["dataset"],
                item["sample_number"],
                item["question_number"],
            )
        )
        summary = {
            "schema": "motiondrop_streamingbench_summary_v1",
            "benchmark": "StreamingBench",
            "model_alias": args.model,
            "model": checkpoint,
            "model_family": spec.family,
            "method": "motiondrop",
            "target_compression": rate,
            "expected_records": len(manifest),
            "completed_records": len(rate_records),
            "missing_question_ids": missing,
            "scores": score_records(rate_records) if rate_records else {},
            "manifest": manifest_report,
            "failures": [item for item in failures if item["question_id"] in missing],
        }
        output_dir = args.output / compression_tag(rate)
        write_json(output_dir / "records.json", rate_records)
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if any_missing:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
