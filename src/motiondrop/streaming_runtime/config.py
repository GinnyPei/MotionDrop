"""Validated configuration for the streaming runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from motiondrop.compression import MotionDropConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime controls shared by the pipeline and LLaVA backend."""

    window_frames: int = 4
    queue_size: int = 2
    fixed_compression: float = 0.9
    motiondrop: MotionDropConfig = field(default_factory=MotionDropConfig)
    max_new_tokens: int = 64
    add_window_newline: bool = True
    # Live-runtime controls. The defaults reproduce the original benchmark
    # behaviour: no time markers, unbounded context, contiguous positions.
    time_markers: bool = False
    max_context_tokens: int | None = None
    evict_to_ratio: float = 0.75
    position_mode: str = "contiguous"
    # Legacy shared history: write every answer into the trunk KV.
    commit_answers: bool = False

    def __post_init__(self) -> None:
        if self.window_frames < 1:
            raise ValueError("window_frames must be positive")
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive")
        if not 0.0 <= self.fixed_compression < 1.0:
            raise ValueError("fixed_compression must be in [0, 1)")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.max_context_tokens is not None and self.max_context_tokens < 256:
            raise ValueError("max_context_tokens must be at least 256")
        if not 0.0 < self.evict_to_ratio < 1.0:
            raise ValueError("evict_to_ratio must be in (0, 1)")
        if self.position_mode not in ("contiguous", "absolute"):
            raise ValueError("position_mode must be 'contiguous' or 'absolute'")

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["RuntimeConfig"]
