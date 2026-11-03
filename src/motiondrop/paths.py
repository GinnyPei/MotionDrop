"""Project directories, each overridable with an environment variable.

Variables can also be set in a ``.env`` file (``NAME=value`` lines) in the
working directory; the process environment takes precedence.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path


@cache
def _dotenv() -> dict[str, str]:
    path = Path(".env")
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and not name.startswith("#"):
            values[name.strip()] = value.strip().strip("\"'")
    return values


def _setting(variable: str) -> str | None:
    return os.environ.get(variable) or _dotenv().get(variable) or None


def _directory(variable: str, default: Path | str) -> Path:
    return Path(_setting(variable) or default).expanduser()


def models_dir() -> Path:
    """Checkpoint root (``MOTIONDROP_MODELS_DIR``, default ``checkpoints``)."""

    return _directory("MOTIONDROP_MODELS_DIR", "checkpoints")


def data_dir() -> Path:
    """Dataset root (``MOTIONDROP_DATA_DIR``, default ``data``)."""

    return _directory("MOTIONDROP_DATA_DIR", "data")


def results_dir() -> Path:
    """Output root (``MOTIONDROP_RESULTS_DIR``, default ``results``)."""

    return _directory("MOTIONDROP_RESULTS_DIR", "results")


def ovobench_dir() -> Path:
    """OVO-Bench root (``MOTIONDROP_OVOBENCH_DIR``, default ``data/ovobench``)."""

    return _directory("MOTIONDROP_OVOBENCH_DIR", data_dir() / "ovobench")


def ovobench_annotations() -> Path:
    return ovobench_dir() / "ovo_bench_new.json"


def ovobench_videos() -> Path:
    return ovobench_dir() / "chunked_videos"


def streamingbench_root() -> Path:
    """StreamingBench root (``MOTIONDROP_STREAMINGBENCH_DIR``, default ``data/streamingbench``)."""

    return _directory("MOTIONDROP_STREAMINGBENCH_DIR", data_dir() / "streamingbench")


def sample_video() -> Path:
    """OVO-Bench clip used by the streaming runtime and demo when none is given."""

    return ovobench_videos() / "3.mp4"


__all__ = [
    "data_dir",
    "models_dir",
    "ovobench_annotations",
    "ovobench_dir",
    "ovobench_videos",
    "results_dir",
    "sample_video",
    "streamingbench_root",
]
