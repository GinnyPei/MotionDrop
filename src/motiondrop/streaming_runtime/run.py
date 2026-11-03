"""Command-line entry point for the pipelined streaming runtime."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from motiondrop.models import load_model_and_processor
from motiondrop.options import add_motiondrop_arguments, motiondrop_config
from motiondrop.paths import results_dir, sample_video
from motiondrop.streaming_runtime.config import RuntimeConfig
from motiondrop.streaming_runtime.latency import DEFAULT_SLO_MS, LatencyModel
from motiondrop.streaming_runtime.llava import LlavaStreamingBackend, resolve_local_checkpoint
from motiondrop.streaming_runtime.pipeline import StreamingPipeline
from motiondrop.video import decode_frame_indices, probe_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motiondrop stream",
        description="Run the bounded MotionDrop encoding/prefill pipeline on one video.",
    )
    parser.add_argument("--video", type=Path, default=sample_video())
    parser.add_argument("--query", required=True)
    parser.add_argument("--model-path")
    parser.add_argument(
        "--output", type=Path, default=results_dir() / "streaming-runtime" / "run.json"
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--query-at-seconds", type=float)
    parser.add_argument("--max-stream-frames", type=int)
    parser.add_argument("--window-frames", type=int, default=RuntimeConfig.window_frames)
    parser.add_argument("--queue-size", type=int, default=RuntimeConfig.queue_size)
    parser.add_argument("--fixed-compression", type=float, default=RuntimeConfig.fixed_compression)
    parser.add_argument("--latency-model", type=Path, help="Use SLO-driven budgets.")
    parser.add_argument("--slo-ms", type=float, help=f"SLO target (default: {DEFAULT_SLO_MS:g}).")
    parser.add_argument("--max-new-tokens", type=int, default=RuntimeConfig.max_new_tokens)
    add_motiondrop_arguments(parser)
    parser.add_argument("--max-gpu-memory-gib", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.query_at_seconds is not None and args.query_at_seconds < 0.0:
        parser.error("--query-at-seconds must be non-negative")
    if args.max_stream_frames is not None and args.max_stream_frames < 1:
        parser.error("--max-stream-frames must be positive")
    if args.max_gpu_memory_gib is not None and args.max_gpu_memory_gib < 1:
        parser.error("--max-gpu-memory-gib must be positive")
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if args.slo_ms is not None and args.latency_model is None:
        parser.error("--slo-ms requires --latency-model")
    if args.latency_model is not None and args.slo_ms is None:
        args.slo_ms = DEFAULT_SLO_MS
    if args.slo_ms is not None and args.slo_ms <= 0.0:
        parser.error("--slo-ms must be positive")
    args.motiondrop = motiondrop_config(parser, args)
    return args


def sampled_indices(
    total_frames: int,
    source_fps: float,
    *,
    target_fps: float,
    query_at_seconds: float | None,
    maximum: int | None,
) -> list[int]:
    last = total_frames - 1
    if query_at_seconds is not None:
        last = min(last, max(0, math.floor(query_at_seconds * source_fps)))
    step = max(1, round(source_fps / target_fps))
    indices = list(range(0, last + 1, step)) or [0]
    if maximum is not None and len(indices) > maximum:
        # Spread the cap over the whole span up to the query, as the benchmark
        # runners do, instead of keeping only its first seconds.
        positions = np.linspace(0, len(indices) - 1, maximum, dtype=int)
        indices = [indices[position] for position in positions]
    return indices


def run(args: argparse.Namespace) -> dict:
    config = RuntimeConfig(
        window_frames=args.window_frames,
        queue_size=args.queue_size,
        fixed_compression=args.fixed_compression,
        motiondrop=args.motiondrop,
        max_new_tokens=args.max_new_tokens,
    )
    checkpoint = resolve_local_checkpoint(args.model_path)
    model, processor, spec, resolved_checkpoint = load_model_and_processor(
        "llava-ov-7b",
        checkpoint,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        local_files_only=args.local_files_only or checkpoint is not None,
    )
    latency_model = None if args.latency_model is None else LatencyModel.load(args.latency_model)
    backend = LlavaStreamingBackend(
        model,
        processor,
        config=config,
        latency_model=latency_model,
        slo_ms=args.slo_ms,
    )
    pipeline = StreamingPipeline(backend, queue_size=config.queue_size)

    total_frames, source_fps = probe_video(args.video)
    indices = sampled_indices(
        total_frames,
        source_fps,
        target_fps=args.fps,
        query_at_seconds=args.query_at_seconds,
        maximum=args.max_stream_frames,
    )
    pipeline.start()
    decoders: list[str] = []
    started = time.perf_counter()
    for start in range(0, len(indices), config.window_frames):
        window_indices = indices[start : start + config.window_frames]
        frames, decoded, decoder = decode_frame_indices(
            args.video,
            window_indices,
            source_fps,
        )
        decoders.append(decoder)
        pipeline.submit(frames, decoded)
    answer, stats = pipeline.query(args.query, max_new_tokens=config.max_new_tokens)
    finished = time.perf_counter()
    result = {
        "schema": "motiondrop_streaming_run_v1",
        "model_alias": spec.alias,
        "model": str(resolved_checkpoint),
        "video": str(args.video.resolve()),
        "query": args.query,
        "answer": answer,
        "sampling": {
            "source_frames": total_frames,
            "source_fps": source_fps,
            "target_fps": args.fps,
            "query_at_seconds": args.query_at_seconds,
            "sampled_frames": len(indices),
            "requested_frame_indices": indices,
            "decoders": sorted(set(decoders)),
        },
        "runtime_config": config.to_dict(),
        "latency_model": None if latency_model is None else latency_model.to_dict(),
        "slo_ms": args.slo_ms,
        "end_to_end_seconds": finished - started,
        **stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    summary = {
        "output": str(args.output),
        "answer": result["answer"],
        "sampled_frames": result["sampling"]["sampled_frames"],
        "windows": result["runtime"]["submitted_windows"],
        "pipeline_wall_seconds": result["runtime"]["pipeline_wall_seconds"],
        "time_to_first_token_seconds": result["query"]["time_to_first_token_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
