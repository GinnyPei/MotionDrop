"""Shared benchmark runtime utilities."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_ANSWER_LABELS = tuple("ABCDE") + ("Yes", "No") + tuple(str(index) for index in range(11))
DEFAULT_COMPRESSION_RATES = (0.0, 0.5, 0.7, 0.9, 0.95)
# Benchmark protocol: frames per causal clip and source-frame area bounds.
DEFAULT_MAX_FRAMES = 6
DEFAULT_MIN_PIXELS = 224 * 224
DEFAULT_MAX_PIXELS = 384 * 384


def parse_compression_rates(value: str) -> list[float]:
    """Parse comma-separated ratios or percentages into unique ratios."""

    rates: list[float] = []
    for item in value.split(","):
        item = item.strip().rstrip("%")
        if not item:
            continue
        rate = float(item)
        if rate > 1.0:
            rate /= 100.0
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Compression rate must be in [0, 1) or [0, 100): {item!r}")
        rates.append(rate)
    if not rates:
        raise ValueError("At least one compression rate is required")
    if len(rates) != len(set(rates)):
        raise ValueError("Compression rates must be unique")
    return rates


def compression_tag(rate: float) -> str:
    return f"c{int(round(rate * 100)):02d}"


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def make_video_conversation(
    frames: Sequence[Image.Image],
    prompt: str,
    *,
    min_pixels: int,
    max_pixels: int,
) -> list[dict]:
    """Build one native-video chat while bounding source-frame resolution."""

    if not frames:
        raise ValueError("A native-video conversation requires at least one frame")
    width, height = frames[0].size
    area = width * height
    bounded_area = min(max(area, int(min_pixels)), int(max_pixels))
    scale = (bounded_area / area) ** 0.5
    resized_width = max(32, int(round(width * scale)))
    resized_height = max(32, int(round(height * scale)))
    if (resized_width, resized_height) == (width, height):
        resized_frames = list(frames)
    else:
        resized_frames = [
            frame.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
            for frame in frames
        ]
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": resized_frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def single_token_ids(tokenizer, labels: Iterable[str]) -> dict[str, int]:
    sequences = label_token_sequences(tokenizer, labels)
    result = {}
    for label, token_ids in sequences.items():
        if len(token_ids) != 1:
            raise ValueError(f"Answer label {label!r} is not one token: {list(token_ids)}")
        result[label] = token_ids[0]
    return result


def label_token_sequences(tokenizer, labels: Iterable[str]) -> dict[str, tuple[int, ...]]:
    """Tokenize constrained answers without silently dropping multi-token labels."""

    result: dict[str, tuple[int, ...]] = {}
    for label in labels:
        token_ids = tuple(
            int(token_id) for token_id in tokenizer(label, add_special_tokens=False).input_ids
        )
        if not token_ids:
            raise ValueError(f"Answer label {label!r} maps to an empty token sequence")
        if len(token_ids) > 2:
            raise ValueError(
                f"Answer label {label!r} uses {len(token_ids)} tokens; "
                "MotionDrop currently supports answer labels of at most two tokens"
            )
        result[label] = token_ids
    if len(result) != len(set(result.values())):
        raise ValueError(f"Answer labels do not map to unique token sequences: {result}")
    return result


def module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError(f"Could not resolve an execution device for {type(module).__name__}")


def score_candidate_sequences(
    first_hidden: torch.Tensor,
    head: torch.nn.Module,
    answer_labels: Sequence[str],
    answer_token_sequences: dict[str, tuple[int, ...]],
    *,
    candidate_labels: Sequence[Sequence[str]] | None = None,
    continuation_hidden: Callable[[int], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Score one- or two-token answers with mean token log probability.

    The official OVO-Bench annotations include the answer ``10``, which is two
    tokens in all supported tokenizers.  Mean log probability keeps candidate
    scores comparable across one- and two-token labels.  The continuation
    callback is evaluated only when a requested candidate actually needs it.
    """

    if first_hidden.ndim != 2:
        raise ValueError(f"Expected [batch, hidden], got {tuple(first_hidden.shape)}")
    batch_size = int(first_hidden.shape[0])
    if candidate_labels is None:
        candidate_labels = [answer_labels] * batch_size
    if len(candidate_labels) != batch_size:
        raise ValueError("candidate_labels must contain one label sequence per sample")

    label_to_column = {label: index for index, label in enumerate(answer_labels)}
    requested_prefixes: set[int] = set()
    for sample_labels in candidate_labels:
        if not sample_labels:
            raise ValueError("Each sample requires at least one candidate label")
        for label in sample_labels:
            if label not in label_to_column:
                raise ValueError(f"Unknown candidate label: {label!r}")
            token_ids = answer_token_sequences[label]
            if len(token_ids) == 2:
                requested_prefixes.add(token_ids[0])

    head_device = module_device(head)
    first_logits = head(first_hidden.to(head_device)).float()
    first_logprobs = F.log_softmax(first_logits, dim=-1)
    first_token_ids = torch.tensor(
        [answer_token_sequences[label][0] for label in answer_labels],
        dtype=torch.long,
        device=head_device,
    )
    scores = first_logprobs.index_select(1, first_token_ids)

    continuation_logprobs: dict[int, torch.Tensor] = {}
    for prefix_token_id in sorted(requested_prefixes):
        if continuation_hidden is None:
            raise ValueError("A multi-token candidate requires continuation_hidden")
        hidden = continuation_hidden(prefix_token_id)
        if hidden.shape != first_hidden.shape:
            raise ValueError(
                "Continuation hidden-state shape mismatch: "
                f"expected={tuple(first_hidden.shape)}, actual={tuple(hidden.shape)}"
            )
        continuation_logprobs[prefix_token_id] = F.log_softmax(
            head(hidden.to(head_device)).float(),
            dim=-1,
        )

    for sample_index, sample_labels in enumerate(candidate_labels):
        for label in sample_labels:
            token_ids = answer_token_sequences[label]
            if len(token_ids) == 1:
                continue
            column = label_to_column[label]
            scores[sample_index, column] = (
                first_logprobs[sample_index, token_ids[0]]
                + continuation_logprobs[token_ids[0]][sample_index, token_ids[1]]
            ) / 2.0
    return scores.cpu()


def validate_compression_stats(stats: dict, rate: float) -> None:
    temporal_units = int(stats["temporal_units"])
    tokens_per_unit = int(stats["tokens_per_temporal_unit_before"])
    expected = max(temporal_units, int(round(temporal_units * tokens_per_unit * (1.0 - rate))))
    if rate == 0.0:
        expected = int(stats["original_visual_tokens"])
    if int(stats["final_visual_tokens"]) != expected:
        raise AssertionError(f"Visual-token target mismatch: expected={expected}, stats={stats}")


def load_jsonl(path: str | Path, key_fields: Sequence[str]) -> tuple[list[dict], set[tuple]]:
    path = Path(path)
    if not path.is_file():
        return [], set()
    records: list[dict] = []
    keys: list[tuple] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            records.append(record)
            keys.append(tuple(record[field] for field in key_fields))
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate result keys in {path}")
    return records, set(keys)


def append_jsonl(path: str | Path, records: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


__all__ = [
    "DEFAULT_ANSWER_LABELS",
    "DEFAULT_COMPRESSION_RATES",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_MIN_PIXELS",
    "append_jsonl",
    "compression_tag",
    "label_token_sequences",
    "load_jsonl",
    "make_video_conversation",
    "module_device",
    "parse_compression_rates",
    "score_candidate_sequences",
    "set_reproducibility",
    "single_token_ids",
    "validate_compression_stats",
    "write_json",
]
