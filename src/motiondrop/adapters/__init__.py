"""Model adapter factory."""

from __future__ import annotations

from collections.abc import Sequence

from motiondrop.compression import MotionDropConfig
from motiondrop.models import ModelSpec


def create_adapter(
    spec: ModelSpec,
    model,
    processor,
    answer_labels: Sequence[str],
    *,
    motiondrop: MotionDropConfig | None = None,
):
    if spec.family == "llava_onevision":
        from motiondrop.adapters.llava_onevision import (
            LlavaOnevisionMotionDropBatchAdapter,
        )

        adapter_class = LlavaOnevisionMotionDropBatchAdapter
    elif spec.family == "qwen3_vl":
        from motiondrop.adapters.qwen import Qwen3VLMotionDropBatchAdapter

        adapter_class = Qwen3VLMotionDropBatchAdapter
    elif spec.family == "qwen3_5":
        from motiondrop.adapters.qwen import Qwen35MotionDropBatchAdapter

        adapter_class = Qwen35MotionDropBatchAdapter
    else:
        raise ValueError(f"Unsupported model family: {spec.family}")
    return adapter_class(model, processor, answer_labels, motiondrop=motiondrop)


__all__ = ["create_adapter"]
