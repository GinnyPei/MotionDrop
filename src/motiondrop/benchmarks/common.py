"""Shared command-line and model setup for benchmark runners."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from motiondrop import __version__
from motiondrop.adapters import create_adapter
from motiondrop.models import DEFAULT_MODEL, MODEL_SPECS, load_model_and_processor
from motiondrop.options import add_motiondrop_arguments, motiondrop_config, motiondrop_options
from motiondrop.paths import models_dir, results_dir
from motiondrop.runtime import (
    DEFAULT_ANSWER_LABELS,
    DEFAULT_COMPRESSION_RATES,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    parse_compression_rates,
)


def add_runtime_arguments(parser: argparse.ArgumentParser, benchmark: str) -> None:
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local checkpoint path or Hugging Face ID; defaults to the selected model registry entry.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Result directory (default: {results_dir() / benchmark}/MODEL).",
    )
    parser.add_argument(
        "--compression-rates",
        default=",".join(f"{rate * 100:g}" for rate in DEFAULT_COMPRESSION_RATES),
        help="Comma-separated ratios or percentages, e.g. 0,50,70,90,95.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Videos per model call; failed batches are split and retried.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum uniformly sampled frames per causal clip/prefix.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=DEFAULT_MIN_PIXELS,
        help="Minimum source-frame area before model preprocessing.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help="Maximum source-frame area before model preprocessing.",
    )
    parser.add_argument(
        "--device-map",
        default="balanced",
        help="Transformers/Accelerate device placement strategy (default: balanced).",
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=int,
        default=None,
        help="Per-GPU placement cap in GiB (default: let Accelerate decide).",
    )
    parser.add_argument(
        "--max-cpu-memory-gib",
        type=int,
        default=96,
        help="CPU placement cap in GiB when a GPU cap is set.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact the Hugging Face Hub.",
    )
    add_motiondrop_arguments(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Validate annotations and videos without loading a model.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Raise the first sample error instead of recording and continuing.",
    )


def finalize_runtime_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    benchmark: str,
) -> argparse.Namespace:
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        parser.error("--min-pixels/--max-pixels are invalid")
    if args.max_gpu_memory_gib is not None and args.max_gpu_memory_gib < 1:
        parser.error("--max-gpu-memory-gib must be positive")
    if args.max_cpu_memory_gib < 1:
        parser.error("--max-cpu-memory-gib must be positive")
    args.motiondrop = motiondrop_config(parser, args)
    try:
        args.compression_rates = parse_compression_rates(args.compression_rates)
    except ValueError as exc:
        parser.error(str(exc))
    if args.model_path is None:
        from motiondrop.downloads import local_model_path

        local_checkpoint = local_model_path(models_dir(), args.model)
        if local_checkpoint is not None:
            args.model_path = str(local_checkpoint)
            args.local_files_only = True
    if args.output is None:
        args.output = results_dir() / benchmark / args.model
    return args


def load_adapter(args: argparse.Namespace):
    model, processor, spec, checkpoint = load_model_and_processor(
        args.model,
        args.model_path,
        device_map=args.device_map,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        local_files_only=args.local_files_only,
    )
    adapter = create_adapter(
        spec,
        model,
        processor,
        DEFAULT_ANSWER_LABELS,
        motiondrop=args.motiondrop,
    )
    return adapter, spec, checkpoint


def ensure_run_config(output: Path, config: dict) -> None:
    """Prevent append-only result files from mixing incompatible runs."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_config.json"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != config:
            raise ValueError(
                f"Output directory already belongs to a different run: {path}. "
                "Choose another --output directory or restore the original arguments."
            )
        return
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_config(args: argparse.Namespace, benchmark: str) -> dict:
    return {
        "schema": "motiondrop_run_config_v1",
        "protocol_revision": 1,
        "runtime_versions": {
            "motiondrop": __version__,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "benchmark": benchmark,
        "method": "motiondrop",
        "model_alias": args.model,
        "model_path": args.model_path or MODEL_SPECS[args.model].default_checkpoint,
        "compression_rates": args.compression_rates,
        "max_frames": args.max_frames,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "batch_size": args.batch_size,
        "device_map": args.device_map,
        "max_gpu_memory_gib": args.max_gpu_memory_gib,
        "max_cpu_memory_gib": args.max_cpu_memory_gib,
        **motiondrop_options(args),
        "seed": args.seed,
    }


__all__ = [
    "add_runtime_arguments",
    "ensure_run_config",
    "file_sha256",
    "finalize_runtime_arguments",
    "load_adapter",
    "runtime_config",
]
