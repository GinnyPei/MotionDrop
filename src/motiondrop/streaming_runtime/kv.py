"""KV-cache history with branch rollback and RoPE-aware eviction.

The live runtime keeps one chronological *trunk* (system prefix followed by
video windows). Every task runs as a temporary *branch* appended to the trunk
and is rolled back afterwards, so tasks never see each other's prompts or
outputs. Old video windows can be evicted from the middle of the cache; in the
``contiguous`` position mode the surviving keys are re-rotated so positions
stay gap-free, in ``absolute`` mode the original positions are kept.
"""

from __future__ import annotations

import torch

POSITION_MODES = ("contiguous", "absolute")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _cache_layers(cache) -> list:
    if cache is None or not hasattr(cache, "layers"):
        return []
    return [layer for layer in cache.layers if getattr(layer, "keys", None) is not None]


def cache_length(cache) -> int:
    layers = _cache_layers(cache)
    lengths = [int(layer.keys.shape[-2]) for layer in layers if layer.keys.numel()]
    if not lengths:
        return 0
    if len(set(lengths)) != 1:
        raise ValueError(f"incremental cache layers have different lengths: {lengths}")
    return lengths[0]


class KVHistory:
    """Owns the decoder KV cache and the mapping from cache slots to RoPE positions."""

    def __init__(self, language_model, *, device, position_mode: str = "contiguous") -> None:
        if position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode must be one of {POSITION_MODES}")
        self.language_model = language_model
        self.device = device
        self.position_mode = position_mode
        self.cache = None
        # Absolute RoPE position of cache slot i is i + position_offset.
        self.position_offset = 0
        self.evicted_tokens = 0
        self.evictions = 0

    @property
    def length(self) -> int:
        return cache_length(self.cache)

    @property
    def next_position(self) -> int:
        return self.length + self.position_offset

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("exactly one of input_ids or inputs_embeds is required")
        length = int(input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1])
        slot = self.length
        cache_positions = torch.arange(slot, slot + length, dtype=torch.long, device=self.device)
        positions = cache_positions + self.position_offset
        output = self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            position_ids=positions.unsqueeze(0),
            cache_position=cache_positions,
            attention_mask=None,
        )
        self.cache = output.past_key_values
        return output

    def checkpoint(self) -> int:
        return self.length

    def rollback(self, length: int) -> None:
        """Drop every cache slot at or after ``length`` (a branch's private tokens)."""

        if length < 0 or length > self.length:
            raise ValueError(f"invalid rollback target {length} for cache length {self.length}")
        for layer in _cache_layers(self.cache):
            layer.keys = layer.keys[..., :length, :]
            layer.values = layer.values[..., :length, :]
        if self.length != length:
            raise AssertionError(f"rollback left {self.length} slots, expected {length}")

    def _inverse_rotation(self, shift: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        rotary = getattr(self.language_model, "rotary_emb", None)
        inv_freq = getattr(rotary, "inv_freq", None)
        if inv_freq is None:
            raise RuntimeError("contiguous eviction needs the language model's rotary inv_freq")
        angles = float(shift) * inv_freq.detach().to("cpu", torch.float64)
        angles = torch.cat((angles, angles))
        if int(angles.shape[0]) != head_dim:
            raise RuntimeError(
                f"rotary dimension {int(angles.shape[0])} does not match head_dim {head_dim}"
            )
        return angles.cos().to(torch.float32), angles.sin().to(torch.float32)

    def evict(self, start: int, end: int) -> int:
        """Remove cache slots ``[start, end)`` and return how many were removed."""

        total = self.length
        if not 0 <= start <= end <= total:
            raise ValueError(f"invalid eviction range [{start}, {end}) for length {total}")
        removed = end - start
        if removed == 0:
            return 0
        rerotate = self.position_mode == "contiguous" and end < total
        cos = sin = None
        for layer in _cache_layers(self.cache):
            keys, values = layer.keys, layer.values
            tail_keys = keys[..., end:, :]
            if rerotate:
                if cos is None:
                    cos, sin = self._inverse_rotation(removed, int(keys.shape[-1]))
                layer_cos = cos.to(keys.device)
                layer_sin = sin.to(keys.device)
                tail = tail_keys.to(torch.float32)
                # k(p - d) = R(-d) k(p): cos(-d)=cos(d), sin(-d)=-sin(d).
                tail_keys = (tail * layer_cos - rotate_half(tail) * layer_sin).to(keys.dtype)
            layer.keys = torch.cat((keys[..., :start, :], tail_keys), dim=-2)
            layer.values = torch.cat((values[..., :start, :], values[..., end:, :]), dim=-2)
        if self.position_mode == "absolute":
            self.position_offset += removed
        self.evicted_tokens += removed
        self.evictions += 1
        if self.length != total - removed:
            raise AssertionError("eviction produced an inconsistent cache length")
        return removed

    def stats(self) -> dict:
        return {
            "cache_length": self.length,
            "next_position": self.next_position,
            "position_mode": self.position_mode,
            "position_offset": self.position_offset,
            "evicted_tokens": self.evicted_tokens,
            "evictions": self.evictions,
        }


__all__ = ["KVHistory", "POSITION_MODES", "cache_length", "rotate_half"]
