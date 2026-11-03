"""MotionDrop: motion-aware visual token compression for video VLMs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from motiondrop.compression import MotionDropConfig, select_motiondrop_tokens
    from motiondrop.models import MODEL_SPECS, ModelSpec

__all__ = ["MODEL_SPECS", "ModelSpec", "MotionDropConfig", "select_motiondrop_tokens"]
__version__ = "0.1.0"


def __getattr__(name: str):
    """Keep package metadata lightweight for setup/download preflights."""

    if name in {"MotionDropConfig", "select_motiondrop_tokens"}:
        from motiondrop import compression

        return getattr(compression, name)
    if name in {"MODEL_SPECS", "ModelSpec"}:
        from motiondrop import models

        return getattr(models, name)
    raise AttributeError(name)
