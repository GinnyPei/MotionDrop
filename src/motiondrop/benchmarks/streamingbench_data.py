"""StreamingBench manifest parsing and causal-prefix video sampling."""

from __future__ import annotations

import ast
import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from motiondrop.runtime import DEFAULT_MAX_FRAMES
from motiondrop.video import decode_frame_indices, probe_video

CHOICE_PROMPT_TEMPLATE = """Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (A, B, C, or D).
Do not include any additional text or explanation in your response."""

QUESTION_ID_RE = re.compile(r"_sample_(\d+)_(\d+)$")
OPTION_LABEL_RE = re.compile(r"^\s*[A-D][.)]\s*", flags=re.IGNORECASE)
MALFORMED_OPTION_LABEL_RE = re.compile(
    r"(?:^|\n|(?<=[.!?'\"]))\s*([A-D])[.)]\s+",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class StreamingBenchDatasetSpec:
    name: str
    csv_relative_path: str
    video_relative_path: str
    task_type: str | None = None


DATASET_SPECS = {
    "anomaly_context": StreamingBenchDatasetSpec(
        "anomaly_context",
        "StreamingBench/Contextual_Understanding.csv",
        "videos/anomaly_context",
        "Anomaly Context Understanding",
    ),
    "emotion_recognition": StreamingBenchDatasetSpec(
        "emotion_recognition",
        "StreamingBench/Omni_Source_Understanding.csv",
        "videos/emotion_recognition",
        "Emotion Recognition",
    ),
    "misleading_context": StreamingBenchDatasetSpec(
        "misleading_context",
        "StreamingBench/Contextual_Understanding.csv",
        "videos/misleading_context",
        "Misleading Context Recognition",
    ),
    "multimodal_alignment": StreamingBenchDatasetSpec(
        "multimodal_alignment",
        "StreamingBench/Omni_Source_Understanding.csv",
        "videos/multimodal_alignment",
        "Multimodal Alignment",
    ),
    "realtime_visual_understanding": StreamingBenchDatasetSpec(
        "realtime_visual_understanding",
        "StreamingBench/Real_Time_Visual_Understanding.csv",
        "videos/realtime_visual_understanding",
    ),
    "scene_understanding": StreamingBenchDatasetSpec(
        "scene_understanding",
        "StreamingBench/Omni_Source_Understanding.csv",
        "videos/scene_understanding",
        "Scene Understanding",
    ),
    "sequential_question_answering": StreamingBenchDatasetSpec(
        "sequential_question_answering",
        "StreamingBench/Sequential_Question_Answering.csv",
        "videos/sequential_question_answering",
        "Sequential Question Answering",
    ),
    "source_discrimination": StreamingBenchDatasetSpec(
        "source_discrimination",
        "StreamingBench/Omni_Source_Understanding.csv",
        "videos/source_discrimination",
        "Source Discrimination",
    ),
}


def parse_timestamp(value: str) -> float:
    parts = str(value).strip().split(":")
    if len(parts) == 2:
        hours = 0.0
        minutes, seconds = (float(part) for part in parts)
    elif len(parts) == 3:
        hours, minutes, seconds = (float(part) for part in parts)
    else:
        raise ValueError(f"Invalid StreamingBench timestamp: {value!r}")
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Invalid StreamingBench timestamp: {value!r}")
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_options(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected four options, got: {value!r}")
    if len(parsed) != 4:
        # A small number of official rows contain punctuation that splits one
        # option. Reconstruct the final complete A/B/C/D cycle.
        joined = "\n".join(str(option).strip() for option in parsed)
        matches = list(MALFORMED_OPTION_LABEL_RE.finditer(joined))
        repaired_cycles: list[list[str]] = []
        for start in range(len(matches)):
            cycle = matches[start : start + 4]
            if [match.group(1).upper() for match in cycle] != list("ABCD"):
                continue
            repaired = []
            for offset, match in enumerate(cycle):
                next_index = start + offset + 1
                end = matches[next_index].start() if next_index < len(matches) else len(joined)
                repaired.append(f"{match.group(1).upper()}. {joined[match.end() : end].strip()}")
            repaired_cycles.append(repaired)
        if not repaired_cycles:
            raise ValueError(f"Could not repair malformed four-option row: {value!r}")
        parsed = repaired_cycles[-1]
    return [OPTION_LABEL_RE.sub("", str(option).strip()) for option in parsed]


def build_choice_prompt(question: str, options: list[str]) -> str:
    if len(options) != 4:
        raise ValueError(f"Expected four options, got {len(options)}")
    formatted = "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(options))
    return CHOICE_PROMPT_TEMPLATE.format(question.strip(), formatted)


def _parse_question_id(question_id: str) -> tuple[int, int]:
    match = QUESTION_ID_RE.search(question_id)
    if not match:
        raise ValueError(f"Cannot parse sample/question index from {question_id!r}")
    return int(match.group(1)), int(match.group(2))


def load_question_id_file(path: str | Path) -> set[str]:
    with Path(path).open(encoding="utf-8") as handle:
        question_ids = [line.strip() for line in handle if line.strip()]
    if not question_ids:
        raise ValueError(f"Question-ID file is empty: {path}")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"Question-ID file contains duplicates: {path}")
    return set(question_ids)


def load_choice_manifest(
    data_root: str | Path,
    datasets: list[str] | tuple[str, ...],
    *,
    limit_per_dataset: int | None = None,
    question_ids: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Intersect official four-choice annotations with locally extracted videos."""

    data_root = Path(data_root)
    annotations: list[dict] = []
    report = {"datasets": {}, "skipped": {"proactive_output": "not a four-choice task"}}
    selected_question_ids: set[str] = set()

    for dataset_name in datasets:
        if dataset_name not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset {dataset_name!r}; choices={sorted(DATASET_SPECS)}")
        spec = DATASET_SPECS[dataset_name]
        csv_path = data_root / spec.csv_relative_path
        video_root = data_root / spec.video_relative_path
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        if not video_root.is_dir():
            raise FileNotFoundError(f"Missing extracted videos: {video_root}")

        csv_rows = matching_rows = missing_video_rows = 0
        selected: list[dict] = []
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                csv_rows += 1
                if spec.task_type is not None and row.get("task_type") != spec.task_type:
                    continue
                matching_rows += 1
                question_id = row["question_id"].strip()
                if question_ids is not None and question_id not in question_ids:
                    continue
                sample_number, question_number = _parse_question_id(question_id)
                video_path = video_root / f"sample_{sample_number}" / "video.mp4"
                if not video_path.is_file():
                    missing_video_rows += 1
                    continue
                answer = row.get("answer", "").strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    raise ValueError(f"Invalid answer {answer!r} for {question_id}")
                if question_id in selected_question_ids:
                    raise ValueError(f"Duplicate question_id across manifest: {question_id}")
                selected_question_ids.add(question_id)
                selected.append(
                    {
                        "question_id": question_id,
                        "dataset": dataset_name,
                        "task_type": row["task_type"].strip(),
                        "sample_number": sample_number,
                        "question_number": question_number,
                        "question": row["question"].strip(),
                        "time_stamp": row["time_stamp"].strip(),
                        "cutoff_seconds": parse_timestamp(row["time_stamp"]),
                        "ground_truth": answer,
                        "options": parse_options(row["options"]),
                        "frames_required": row.get("frames_required", "").strip(),
                        "temporal_clue_type": row.get("temporal_clue_type", "").strip(),
                        "video_path": str(video_path),
                    }
                )

        selected.sort(key=lambda item: (item["sample_number"], item["question_number"]))
        if limit_per_dataset is not None:
            selected = selected[:limit_per_dataset]
        annotations.extend(selected)
        report["datasets"][dataset_name] = {
            "spec": asdict(spec),
            "csv_rows": csv_rows,
            "matching_rows": matching_rows,
            "missing_video_rows": missing_video_rows,
            "selected_rows": len(selected),
            "unique_videos": len({item["video_path"] for item in selected}),
        }

    report["total_rows"] = len(annotations)
    report["unique_videos"] = len({item["video_path"] for item in annotations})
    if question_ids is not None:
        selected_after_limits = {item["question_id"] for item in annotations}
        unavailable = sorted(question_ids - selected_question_ids)
        if unavailable:
            raise ValueError(
                f"Question-ID filter contains {len(unavailable)} unavailable IDs: {unavailable[:10]}"
            )
        report["question_id_filter_rows"] = len(question_ids)
        report["selected_question_id_rows"] = len(selected_after_limits)
    return annotations, report


def load_video_prefix_frames(
    video_path: str,
    cutoff_seconds: float,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> tuple[list[Image.Image], list[int], dict]:
    """Uniformly sample only the causal prefix through the question timestamp."""

    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    total_frames, source_fps = probe_video(video_path)
    cutoff_frame = int(math.floor(max(0.0, cutoff_seconds) * source_fps))
    prefix_last = min(total_frames - 1, cutoff_frame)
    prefix_frames = prefix_last + 1
    if prefix_frames <= max_frames:
        indices = list(range(prefix_frames))
    else:
        indices = np.linspace(0, prefix_last, max_frames, dtype=int).tolist()

    metadata = {
        "source_fps": source_fps,
        "source_total_frames": total_frames,
        "cutoff_seconds": float(cutoff_seconds),
        "prefix_last_frame": prefix_last,
        "prefix_frame_count": prefix_frames,
        "sampling_protocol": "uniform_max_frames_with_causal_endpoint",
        "requested_sample_count": len(indices),
    }
    images, decoded_indices, decode_backend = decode_frame_indices(
        video_path,
        indices,
        source_fps,
        prefer_decord=False,
    )
    metadata.update(
        {
            "decode_backend": decode_backend,
            "requested_frame_indices": [int(index) for index in indices],
            "decoded_frame_count": len(images),
        }
    )
    return images, decoded_indices, metadata


__all__ = [
    "DATASET_SPECS",
    "build_choice_prompt",
    "load_choice_manifest",
    "load_question_id_file",
    "load_video_prefix_frames",
    "parse_options",
    "parse_timestamp",
]
