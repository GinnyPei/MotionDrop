"""Supported model registry and checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ModelFamily = Literal["llava_onevision", "qwen3_vl", "qwen3_5"]

# Backbone of the paper's main experiments.
DEFAULT_MODEL = "llava-ov-7b"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    family: ModelFamily
    default_checkpoint: str
    dtype: torch.dtype
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "llava-ov-7b": ModelSpec(
        alias="llava-ov-7b",
        family="llava_onevision",
        default_checkpoint="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        dtype=torch.float16,
        description="LLaVA-OneVision Qwen2 7B (Transformers-converted checkpoint)",
    ),
    "qwen3-vl-8b": ModelSpec(
        alias="qwen3-vl-8b",
        family="qwen3_vl",
        default_checkpoint="Qwen/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        description="Qwen3-VL 8B Instruct",
    ),
    "qwen3.5-9b": ModelSpec(
        alias="qwen3.5-9b",
        family="qwen3_5",
        default_checkpoint="Qwen/Qwen3.5-9B",
        dtype=torch.bfloat16,
        description="Qwen3.5 9B (non-thinking benchmark mode)",
    ),
}


def get_model_spec(alias: str) -> ModelSpec:
    try:
        return MODEL_SPECS[alias]
    except KeyError as exc:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unknown model alias {alias!r}; choose one of: {choices}") from exc


def _model_class(family: ModelFamily):
    import transformers

    class_name = {
        "llava_onevision": "LlavaOnevisionForConditionalGeneration",
        "qwen3_vl": "Qwen3VLForConditionalGeneration",
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
    }[family]
    model_class = getattr(transformers, class_name, None)
    if model_class is None:
        raise RuntimeError(
            f"Installed Transformers does not provide {class_name}. "
            "Run scripts/install.sh to install the tested dependency set."
        )
    return model_class


def load_model_and_processor(
    alias: str,
    model_path: str | None = None,
    *,
    device_map: str = "balanced",
    max_gpu_memory_gib: int | None = None,
    max_cpu_memory_gib: int = 96,
    local_files_only: bool = False,
):
    """Load one supported checkpoint and its processor.

    ``max_gpu_memory_gib`` applies to every visible GPU.  When omitted,
    Accelerate determines placement from currently available memory.
    """

    from transformers import AutoConfig, AutoProcessor
    from transformers.utils import logging as transformers_logging

    spec = get_model_spec(alias)
    checkpoint = model_path or spec.default_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("MotionDrop benchmark inference requires at least one CUDA GPU")

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=local_files_only)
    if spec.family == "llava_onevision" and config.model_type != "llava_onevision":
        raise ValueError(
            "llava-ov-7b requires the Transformers-converted checkpoint with "
            "model_type='llava_onevision'. Use "
            "llava-hf/llava-onevision-qwen2-7b-ov-hf, not the original "
            "LLaVA-NeXT training checkpoint."
        )

    transformers_logging.disable_progress_bar()
    processor_kwargs = {"local_files_only": local_files_only}
    if spec.family == "llava_onevision":
        processor_kwargs["use_fast"] = False
    processor = AutoProcessor.from_pretrained(checkpoint, **processor_kwargs)

    max_memory = None
    if max_gpu_memory_gib is not None:
        if max_gpu_memory_gib < 1:
            raise ValueError("max_gpu_memory_gib must be positive")
        max_memory = {
            index: f"{max_gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
        }
        max_memory["cpu"] = f"{max_cpu_memory_gib}GiB"

    model = _model_class(spec.family).from_pretrained(
        checkpoint,
        config=config,
        dtype=spec.dtype,
        attn_implementation="sdpa",
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.eval()
    model.config.use_cache = False
    return model, processor, spec, checkpoint


def format_model_table() -> str:
    header = ("ALIAS", "DEFAULT CHECKPOINT", "DESCRIPTION")
    rows = [
        header,
        *((spec.alias, spec.default_checkpoint, spec.description) for spec in MODEL_SPECS.values()),
    ]
    widths = [max(len(row[index]) for row in rows) for index in range(3)]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows
    )


__all__ = [
    "DEFAULT_MODEL",
    "MODEL_SPECS",
    "ModelSpec",
    "format_model_table",
    "get_model_spec",
    "load_model_and_processor",
]
