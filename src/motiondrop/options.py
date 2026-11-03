"""Command-line options generated from :class:`MotionDropConfig`.

The dataclass owns every default; this module only names the flags.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from motiondrop.compression import FIRST_FRAME_POLICIES, MotionDropConfig

# Config field -> (command-line flag, help text).
_OPTIONS = {
    "noise_threshold": ("--motion-noise-threshold", "Residual noise threshold tau."),
    "local_radius": (
        "--motion-local-radius",
        "Residual search radius; 1 means a 3x3 neighborhood.",
    ),
    "first_frame_policy": (
        "--first-frame-policy",
        "Budget for a first frame that has no reference frame.",
    ),
    "budget_floor": (
        "--budget-floor",
        "Guaranteed fraction of the uniform per-frame share.",
    ),
    "motion_power": ("--motion-power", "Allocate frame budgets by E_t ** power."),
    "dedup": (
        "--dedup",
        "Rank last static tokens whose match in the previous frame was kept; "
        "--no-dedup is the paper method.",
    ),
}
_CHOICES = {"first_frame_policy": FIRST_FRAME_POLICIES}


def _dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def add_motiondrop_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = {field.name: field.default for field in fields(MotionDropConfig)}
    group = parser.add_argument_group("MotionDrop hyperparameters")
    for name, (flag, help_text) in _OPTIONS.items():
        default = defaults[name]
        kind = (
            {"action": argparse.BooleanOptionalAction}
            if isinstance(default, bool)
            else {"type": type(default), "choices": _CHOICES.get(name)}
        )
        group.add_argument(
            flag, default=default, help=f"{help_text} (default: %(default)s)", **kind
        )


def motiondrop_options(args: argparse.Namespace) -> dict:
    """Flat ``{cli_dest: value}`` record of the hyperparameters, for run configs."""

    return {_dest(flag): getattr(args, _dest(flag)) for flag, _ in _OPTIONS.values()}


def motiondrop_config(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> MotionDropConfig:
    """Build a validated config, reporting invalid values as CLI errors."""

    values = {name: getattr(args, _dest(flag)) for name, (flag, _) in _OPTIONS.items()}
    try:
        return MotionDropConfig(**values)
    except ValueError as exc:
        parser.error(str(exc))


__all__ = ["add_motiondrop_arguments", "motiondrop_config", "motiondrop_options"]
