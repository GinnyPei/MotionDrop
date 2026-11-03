"""Method-independent OVO-Bench prompts and video helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from motiondrop.runtime import DEFAULT_MAX_FRAMES
from motiondrop.video import decode_frame_indices, probe_video

BR_PROMPT_TEMPLATE = """Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response."""

REC_PROMPT_TEMPLATE = """You're watching a video in which people may perform a certain type of action repetitively.
The people performing this kind of action are referred to as 'they' in the following statement.
Your task is to count how many times different people in the video perform this action in total.
One complete motion counts as one.
Now answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3...).
Do not include any additional text or explanation in your response."""

SSR_PROMPT_TEMPLATE = """You're watching a tutorial video containing a sequence of steps.
The following is one step from the whole procedure:
{}
Your task is to determine whether the person in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response."""

CRR_PROMPT_TEMPLATE = """You're responsible for answering questions based on the video content.
The following question is relevant to the latest frames, i.e. the end of the video.
{}
Decide whether the available visual content, especially the latest frames, provides enough information to answer the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response."""

BACKWARD_TASKS = ("EPM", "ASI", "HLD")
REALTIME_TASKS = ("STU", "OJR", "ATR", "ACR", "OCR", "FPD")
FORWARD_TASKS = ("REC", "SSR", "CRR")
ALL_TASKS = BACKWARD_TASKS + REALTIME_TASKS + FORWARD_TASKS


def load_video_frames(
    video_path: str,
    *,
    fps: float = 1.0,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> tuple[list[Image.Image], list[int]]:
    """Uniformly sample at most ``max_frames`` from one causal video clip."""

    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    total, source_fps = probe_video(video_path)
    step = max(int(round(source_fps / fps)), 1)
    indices = list(range(0, total, step))
    if len(indices) > max_frames:
        indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()
    indices = [int(index) for index in (indices or [0])]
    images, decoded_indices, _ = decode_frame_indices(video_path, indices, source_fps)
    return images, decoded_indices


def build_choice_prompt(task: str, question: str, options: list[str]) -> str:
    if task not in BACKWARD_TASKS + REALTIME_TASKS:
        raise ValueError(f"Expected a backward/realtime task, got {task!r}")
    formatted = (
        "; ".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(options)) + ";"
    )
    return BR_PROMPT_TEMPLATE.format(question, formatted)


def build_forward_prompt(task: str, annotation: dict, index: int) -> str:
    if task == "REC":
        return REC_PROMPT_TEMPLATE.format(f"How many times did they {annotation['activity']}?")
    if task == "SSR":
        return SSR_PROMPT_TEMPLATE.format(annotation["test_info"][index]["step"])
    if task == "CRR":
        return CRR_PROMPT_TEMPLATE.format(annotation["question"])
    raise ValueError(f"Expected a forward task, got {task!r}")


__all__ = [
    "ALL_TASKS",
    "BACKWARD_TASKS",
    "FORWARD_TASKS",
    "REALTIME_TASKS",
    "build_choice_prompt",
    "build_forward_prompt",
    "load_video_frames",
]
