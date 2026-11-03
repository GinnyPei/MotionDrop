"""Profile LLaVA stages and fit the MotionDrop quadratic latency model."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import numpy as np
import torch

from motiondrop.adapters.llava_vision import encode_video_features_with_attention
from motiondrop.models import load_model_and_processor
from motiondrop.runtime import module_device
from motiondrop.streaming_runtime.latency import fit_latency_model
from motiondrop.streaming_runtime.llava import resolve_local_checkpoint
from motiondrop.video import decode_frame_indices, probe_video


def _positive_csv(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile encoder/prefill latency and fit the paper's quadratic model."
    )
    parser.add_argument("--output", type=Path, required=True, help="Fitted model JSON path.")
    parser.add_argument("--samples", type=Path, help="Fit an existing profile JSON.")
    parser.add_argument("--samples-output", type=Path, help="Optional raw profile JSON path.")
    parser.add_argument("--video", type=Path, help="Representative video used for profiling.")
    parser.add_argument("--model-path")
    parser.add_argument("--frame-counts", type=_positive_csv, default=_positive_csv("1,2,4,6"))
    parser.add_argument(
        "--budgets",
        type=_positive_csv,
        default=_positive_csv("32,64,128,256,512,768,1024"),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-gpu-memory-gib", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.samples is None and args.video is None:
        parser.error("--video is required unless --samples is provided")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    return args


def _synchronize() -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def _median_latency(function, *, warmup: int, repeats: int) -> tuple[float, object]:
    result = None
    for _ in range(warmup):
        result = function()
    measurements = []
    for _ in range(repeats):
        _synchronize()
        starts = []
        ends = []
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(index))
                starts.append(start)
                ends.append(end)
        result = function()
        for index, end in enumerate(ends):
            with torch.cuda.device(index):
                end.record(torch.cuda.current_stream(index))
        for end in ends:
            end.synchronize()
        measurements.append(
            max(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
        )
    return statistics.median(measurements), result


@torch.inference_mode()
def profile(args: argparse.Namespace) -> dict:
    checkpoint = resolve_local_checkpoint(args.model_path)
    model, processor, spec, resolved_checkpoint = load_model_and_processor(
        "llava-ov-7b",
        checkpoint,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        local_files_only=args.local_files_only or checkpoint is not None,
    )
    total_frames, source_fps = probe_video(args.video)
    maximum_frames = max(args.frame_counts)
    if total_frames < maximum_frames:
        raise ValueError(
            f"profiling needs {maximum_frames} source frames, video has {total_frames}"
        )
    all_indices = np.linspace(0, total_frames - 1, maximum_frames, dtype=int).tolist()
    all_frames, decoded_indices, decoder = decode_frame_indices(
        args.video,
        [int(value) for value in all_indices],
        source_fps,
    )

    encoder_samples = []
    feature_bank = None
    for frame_count in args.frame_counts:
        positions = np.linspace(0, maximum_frames - 1, frame_count, dtype=int).tolist()
        frames = [all_frames[int(position)] for position in positions]
        processed = processor.video_processor(
            videos=frames,
            do_sample_frames=False,
            return_tensors="pt",
        )
        pixels = processed["pixel_values_videos"]

        def encode(active_pixels=pixels):
            return encode_video_features_with_attention(
                model,
                active_pixels,
                capture_attention=True,
            )

        latency_ms, result = _median_latency(encode, warmup=args.warmup, repeats=args.repeats)
        outputs, _attention = result
        features = outputs.pooler_output.detach()
        visual_tokens = int(features.shape[1])
        encoder_samples.append(
            {
                "frame_count": frame_count,
                "visual_tokens": visual_tokens,
                "latency_ms": latency_ms,
            }
        )
        if feature_bank is None or int(features.shape[1]) > int(feature_bank.shape[1]):
            feature_bank = features

    assert feature_bank is not None
    language_model = model.model.language_model
    embedding_device = module_device(model.model.get_input_embeddings())
    hidden = feature_bank[0].to(embedding_device)
    prefill_samples = []
    for budget in args.budgets:
        repetitions = (budget + int(hidden.shape[0]) - 1) // int(hidden.shape[0])
        inputs_embeds = hidden.repeat(repetitions, 1)[:budget].unsqueeze(0).contiguous()
        positions = torch.arange(budget, device=embedding_device, dtype=torch.long)

        def prefill(active_embeds=inputs_embeds, active_positions=positions):
            return language_model(
                input_ids=None,
                inputs_embeds=active_embeds,
                use_cache=False,
                return_dict=True,
                position_ids=active_positions.unsqueeze(0),
                cache_position=active_positions,
                attention_mask=None,
            )

        latency_ms, _ = _median_latency(prefill, warmup=args.warmup, repeats=args.repeats)
        prefill_samples.append({"retained_tokens": budget, "latency_ms": latency_ms})

    return {
        "schema": "motiondrop_latency_profile_v1",
        "measurement": "cuda_event_max_across_visible_devices",
        "model_alias": spec.alias,
        "model": str(resolved_checkpoint),
        "video": str(args.video.resolve()),
        "decoded_frame_indices": decoded_indices,
        "decoder": decoder,
        "platform": {
            "host": platform.node(),
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
            "gpu_count": torch.cuda.device_count(),
            "torch": torch.__version__,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "encoder_samples": encoder_samples,
        "prefill_samples": prefill_samples,
    }


def fit(profile_data: dict, output: Path) -> dict:
    if profile_data.get("schema") != "motiondrop_latency_profile_v1":
        raise ValueError("unsupported latency-profile schema")
    metadata = {
        "model_alias": profile_data.get("model_alias"),
        "model": profile_data.get("model"),
        "platform": profile_data.get("platform", {}),
        "measurement": profile_data.get("measurement"),
        "profile": {
            "warmup": profile_data.get("warmup"),
            "repeats": profile_data.get("repeats"),
            "encoder_samples": profile_data.get("encoder_samples"),
            "prefill_samples": profile_data.get("prefill_samples"),
        },
    }
    model, metrics = fit_latency_model(
        profile_data["encoder_samples"],
        profile_data["prefill_samples"],
        metadata=metadata,
    )
    data = model.to_dict()
    data["fit_metrics"] = metrics
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.samples is not None:
        profile_data = json.loads(args.samples.read_text(encoding="utf-8"))
    else:
        profile_data = profile(args)
        samples_output = args.samples_output or args.output.with_suffix(".samples.json")
        samples_output.parent.mkdir(parents=True, exist_ok=True)
        samples_output.write_text(
            json.dumps(profile_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    fitted = fit(profile_data, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "coefficients": fitted["coefficients"],
                "fit_metrics": fitted["fit_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
