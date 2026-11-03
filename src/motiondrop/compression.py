"""Paper-aligned MotionDrop visual-token compression."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


def allocate_exact(
    weights: torch.Tensor,
    total: int,
    *,
    minimum: int = 1,
    maximum: int,
) -> list[int]:
    """Allocate an exact bounded integer budget using water filling.

    The continuous allocation is rounded with the largest-remainder method.
    Invalid or zero weights receive a small positive floor so the result stays
    deterministic and feasible.
    """

    count = int(weights.numel())
    if count == 0:
        return []
    minimum = int(minimum)
    maximum = int(maximum)
    if minimum < 0:
        raise ValueError("minimum must be non-negative")
    if maximum < minimum:
        raise ValueError("maximum must be greater than or equal to minimum")

    total = min(max(int(total), count * minimum), count * maximum)
    if minimum == maximum:
        return [minimum] * count

    remaining = total - count * minimum
    if remaining == 0:
        return [minimum] * count

    raw_weights = weights.detach().reshape(-1).to(torch.float64)
    raw_weights = torch.where(
        torch.isfinite(raw_weights), raw_weights, torch.zeros_like(raw_weights)
    ).clamp_min(1e-6)
    capacity = maximum - minimum
    continuous = torch.zeros_like(raw_weights)
    active = torch.ones(count, dtype=torch.bool, device=raw_weights.device)
    amount_left = float(remaining)

    while amount_left > 0.0 and bool(active.any().item()):
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        shares = amount_left * raw_weights[active_indices] / raw_weights[active_indices].sum()
        saturated = shares >= float(capacity)
        if not bool(saturated.any().item()):
            continuous[active_indices] = shares
            break
        saturated_indices = active_indices[saturated]
        continuous[saturated_indices] = float(capacity)
        amount_left -= float(capacity * int(saturated_indices.numel()))
        active[saturated_indices] = False

    increments = torch.floor(continuous).long()
    result = increments + minimum
    leftover = total - int(result.sum().item())
    if leftover:
        fractions = continuous - increments.to(continuous.dtype)
        eligible = result < maximum
        order = torch.argsort(fractions, descending=True, stable=True)
        order = order[eligible[order]]
        if leftover > int(order.numel()):
            raise RuntimeError("MotionDrop could not allocate the exact token budget")
        result[order[:leftover]] += 1

    budgets = [int(value) for value in result.tolist()]
    if sum(budgets) != total:
        raise AssertionError((budgets, total))
    return budgets


FIRST_FRAME_POLICIES = ("window_mean", "zero")


@dataclass(frozen=True)
class MotionDropConfig:
    """Hyperparameters shared by every supported model adapter."""

    target_compression: float = 0.0
    noise_threshold: float = 0.1
    local_radius: int = 1
    preserve_token_order: bool = True
    # How a first frame without any reference frame is budgeted.  "window_mean"
    # gives it the mean motion magnitude of the other frames in the window (a
    # neutral share); "zero" is the legacy behavior, which leaves it with only
    # the one-token minimum.  Either way it is ranked by spatial saliency alone.
    first_frame_policy: str = "window_mean"
    # Frame-allocation shape.  Each frame is guaranteed ``budget_floor`` of the
    # uniform per-frame share; the rest is split in proportion to
    # ``E_t ** motion_power``.  The defaults (0, 1) are Eq. (3) exactly.
    budget_floor: float = 0.0
    motion_power: float = 1.0
    # Rank last any static token whose matched token in the previous frame was
    # already kept, so the frame's budget goes to content not yet in context.
    # False reproduces the paper method exactly.
    dedup: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_compression < 1.0:
            raise ValueError("target_compression must be in [0, 1)")
        if self.noise_threshold <= 0.0:
            raise ValueError("noise_threshold must be positive")
        if self.local_radius < 0:
            raise ValueError("local_radius must be non-negative")
        if self.first_frame_policy not in FIRST_FRAME_POLICIES:
            raise ValueError(f"first_frame_policy must be one of {FIRST_FRAME_POLICIES}")
        if not 0.0 <= self.budget_floor <= 1.0:
            raise ValueError("budget_floor must be in [0, 1]")
        if self.motion_power <= 0.0:
            raise ValueError("motion_power must be positive")


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    """Map one-dimensional scores to deterministic ranks in ``[0, 1]``."""

    if values.ndim != 1:
        raise ValueError(f"Rank normalization expects one dimension, got {tuple(values.shape)}")
    if values.numel() <= 1:
        return torch.zeros_like(values, dtype=torch.float32)
    order = torch.argsort(values.float(), stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks / float(values.numel() - 1)


def _spatial_shape(tokens: int, requested: tuple[int, int] | None) -> tuple[int, int]:
    if requested is not None:
        height, width = (int(requested[0]), int(requested[1]))
        if height < 1 or width < 1 or height * width != tokens:
            raise ValueError(f"Invalid spatial shape {requested} for {tokens} tokens")
        return height, width
    side = int(tokens**0.5)
    if side * side != tokens:
        # Real adapters provide their two-dimensional patch grid.  A 1-D grid
        # keeps custom adapters and lightweight tests usable when metadata is
        # unavailable, while retaining local matching semantics.
        return 1, tokens
    return side, side


def _local_match(
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    height: int,
    width: int,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation (1): residual to the best local match in a reference frame,
    and that match's index in the reference frame."""

    current = F.normalize(current.float(), dim=-1)
    reference = F.normalize(reference.float(), dim=-1)
    tokens = height * width
    if radius == 0:
        residual = 1.0 - (current * reference).sum(dim=-1)
        return residual, torch.arange(tokens, device=current.device)

    hidden = int(reference.shape[-1])
    kernel = 2 * radius + 1
    reference_grid = reference.T.reshape(1, hidden, height, width)
    neighborhoods = F.unfold(reference_grid, kernel_size=kernel, padding=radius)
    neighborhoods = neighborhoods.reshape(hidden, kernel * kernel, tokens)
    neighborhoods = neighborhoods.permute(2, 1, 0)

    def unfold_map(grid: torch.Tensor) -> torch.Tensor:
        return (
            F.unfold(grid.reshape(1, 1, height, width), kernel_size=kernel, padding=radius)
            .reshape(kernel * kernel, tokens)
            .T
        )

    ones = torch.ones(tokens, device=reference.device)
    valid = unfold_map(ones).bool()
    # Exact for any realistic grid: float32 holds integers up to 2**24.
    neighbor_index = unfold_map(torch.arange(tokens, device=reference.device, dtype=torch.float32))
    similarities = (current[:, None, :] * neighborhoods).sum(dim=-1)
    similarities = similarities.masked_fill(~valid, float("-inf"))
    best, slot = similarities.max(dim=-1)
    match = neighbor_index.gather(1, slot[:, None]).squeeze(1).long()
    return 1.0 - best, match


def _fallback_spatial_saliency(features: torch.Tensor) -> torch.Tensor:
    centered = features.float() - features.float().mean(dim=1, keepdim=True)
    return centered.norm(dim=-1)


def select_motiondrop_tokens(
    features: torch.Tensor,
    config: MotionDropConfig,
    *,
    spatial_scores: torch.Tensor | None = None,
    spatial_shape: tuple[int, int] | None = None,
    reference_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Select visual tokens from a ``[time, space, hidden]`` feature tensor.

    Returned indices index ``features.flatten(0, 1)``.  The target is computed per temporal
    unit, matching the published benchmark protocol and guaranteeing at least
    one retained token in every unit.
    """

    if features.ndim != 3:
        raise ValueError(f"Expected [time, space, hidden], got {tuple(features.shape)}")
    temporal_units, tokens_per_unit, _ = features.shape
    if temporal_units < 1 or tokens_per_unit < 1:
        raise ValueError("features must contain at least one temporal unit and one token")

    original = temporal_units * tokens_per_unit
    target = (
        original
        if config.target_compression == 0.0
        else max(temporal_units, int(round(original * (1.0 - config.target_compression))))
    )
    height, width = _spatial_shape(tokens_per_unit, spatial_shape)
    if spatial_scores is not None and tuple(spatial_scores.shape) != (
        temporal_units,
        tokens_per_unit,
    ):
        raise ValueError(
            "spatial_scores shape mismatch: "
            f"{tuple(spatial_scores.shape)} != {(temporal_units, tokens_per_unit)}"
        )
    if reference_features is not None and tuple(reference_features.shape) != tuple(
        features.shape[1:]
    ):
        raise ValueError(
            "reference_features shape mismatch: "
            f"{tuple(reference_features.shape)} != {tuple(features.shape[1:])}"
        )
    saliency_source = "encoder_mean_received_attention"
    if spatial_scores is None:
        spatial_scores = _fallback_spatial_saliency(features)
        saliency_source = "projected_feature_norm_fallback"

    residual_maps = torch.zeros(
        temporal_units,
        tokens_per_unit,
        device=features.device,
        dtype=torch.float32,
    )
    match_maps = torch.zeros(
        temporal_units, tokens_per_unit, device=features.device, dtype=torch.long
    )
    first_with_reference = 0 if reference_features is not None else 1
    for time_index in range(first_with_reference, temporal_units):
        reference = reference_features if time_index == 0 else features[time_index - 1]
        residual_maps[time_index], match_maps[time_index] = _local_match(
            features[time_index],
            reference,
            height=height,
            width=width,
            radius=config.local_radius,
        )
    motion_magnitudes = (residual_maps - config.noise_threshold).clamp_min(0.0).mean(dim=1)
    frame_lambdas = motion_magnitudes / (motion_magnitudes + config.noise_threshold)
    if reference_features is None:
        # Frame 0 has no residual, so its zero magnitude means "unknown", not
        # "static".  It keeps lambda = 0 (spatial-only ranking) regardless.
        if config.first_frame_policy == "window_mean" and temporal_units > 1:
            motion_magnitudes[0] = motion_magnitudes[1:].mean()
        frame_lambdas[0] = 0.0
    floor = max(1, int(config.budget_floor * target / temporal_units))
    budgets = allocate_exact(
        motion_magnitudes.pow(config.motion_power),
        target,
        minimum=min(floor, tokens_per_unit),
        maximum=tokens_per_unit,
    )

    selected: list[torch.Tensor] = []
    duplicates_skipped: list[int] = []
    previous_kept: torch.Tensor | None = None
    for time_index in range(temporal_units):
        budget = budgets[time_index]
        if config.target_compression == 0.0 or budget >= tokens_per_unit:
            local = torch.arange(tokens_per_unit, device=features.device)
        else:
            temporal_rank = _rank_normalize(residual_maps[time_index])
            spatial_rank = _rank_normalize(spatial_scores[time_index])
            frame_lambda = frame_lambdas[time_index]
            importance = frame_lambda * temporal_rank + (1.0 - frame_lambda) * spatial_rank
            if config.dedup and previous_kept is not None:
                duplicate = (residual_maps[time_index] <= config.noise_threshold) & (
                    previous_kept[match_maps[time_index]]
                )
                # Ranks lie in [0, 1], so this orders every duplicate last.
                importance = importance - 2.0 * duplicate.float()
            local = torch.topk(importance, budget, sorted=False).indices
            if config.dedup and previous_kept is not None:
                duplicates_skipped.append(int(duplicate.sum()) - int(duplicate[local].sum()))
        if config.preserve_token_order:
            local = local.sort().values
        previous_kept = torch.zeros(tokens_per_unit, dtype=torch.bool, device=features.device)
        previous_kept[local] = True
        selected.append(local + time_index * tokens_per_unit)

    keep_indices = torch.cat(selected)
    if int(keep_indices.numel()) != target:
        raise AssertionError((int(keep_indices.numel()), target))

    stats = {
        "schema": "motiondrop_compression_v2",
        "method": "motiondrop",
        "adapter_stage": "post_vision_encoder_pre_language_model",
        "target_compression": float(config.target_compression),
        "temporal_units": int(temporal_units),
        "tokens_per_temporal_unit_before": int(tokens_per_unit),
        "original_visual_tokens": int(original),
        "final_visual_tokens": int(target),
        "actual_visual_compression": 1.0 - target / original,
        "per_temporal_unit_keep": budgets,
        "motion_magnitudes": [float(value) for value in motion_magnitudes.tolist()],
        "frame_lambdas": [float(value) for value in frame_lambdas.tolist()],
        "duplicates_skipped": int(sum(duplicates_skipped)),
        "reference_frame": "immediately_preceding_temporal_unit",
        "frame0_policy": (
            "previous_window_reference"
            if reference_features is not None
            else f"{config.first_frame_policy}_motion_spatial_only"
        ),
        "frame_budget_mode": "motion_magnitude_proportional_exact",
        "temporal_score_mode": "best_local_reference_cosine_residual",
        "spatial_score_mode": saliency_source,
        "token_score_mode": "lambda * ranknorm(temporal) + (1-lambda) * ranknorm(spatial)",
        "spatial_shape": [height, width],
        "preserve_visual_token_order": bool(config.preserve_token_order),
        "config": asdict(config),
    }
    return keep_indices, stats


__all__ = [
    "FIRST_FRAME_POLICIES",
    "MotionDropConfig",
    "allocate_exact",
    "select_motiondrop_tokens",
]
