"""Bounded two-stage streaming pipeline from the MotionDrop runtime design."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image


def time_marker(seconds: float) -> str:
    """Text marker placed before each window when time markers are enabled."""

    return f"[{seconds:.1f}s]"


@dataclass(frozen=True)
class FrameWindow:
    index: int
    frames: tuple[Image.Image, ...]
    frame_indices: tuple[int, ...]
    submitted_at: float
    timestamps: tuple[float, ...] | None = None

    @classmethod
    def create(
        cls,
        index: int,
        frames: Sequence[Image.Image],
        frame_indices: Sequence[int],
        timestamps: Sequence[float] | None = None,
    ) -> FrameWindow:
        if index < 0:
            raise ValueError("window index must be non-negative")
        if not frames or len(frames) != len(frame_indices):
            raise ValueError("frames and frame_indices must be non-empty and equal length")
        if timestamps is not None and len(timestamps) != len(frames):
            raise ValueError("timestamps must match frames")
        return cls(
            index,
            tuple(frames),
            tuple(int(value) for value in frame_indices),
            time.perf_counter(),
            None if timestamps is None else tuple(float(value) for value in timestamps),
        )


@dataclass(frozen=True)
class EncodedWindow:
    index: int
    payload: Any
    stats: dict
    submitted_at: float
    encode_started_at: float
    encode_finished_at: float


class StreamingBackend(Protocol):
    def initialize_history(self) -> None: ...

    def encode_window(self, window: FrameWindow) -> EncodedWindow: ...

    def prefill_window(self, window: EncodedWindow) -> dict: ...

    def answer(self, query: str, *, max_new_tokens: int) -> tuple[str, dict]: ...


_STOP = object()
# How often blocked worker threads re-check for shutdown.
_POLL_SECONDS = 0.1


class StreamingPipeline:
    """Overlap visual encoding with incremental LLM prefill using bounded queues."""

    def __init__(self, backend: StreamingBackend, *, queue_size: int = 2) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.backend = backend
        self.input_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.prefill_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.errors: list[dict] = []
        self.events: list[dict] = []
        self.submitted_windows = 0
        self.prefilled_windows = 0
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.input_closed = False
        self.started = False
        self.encoder_thread: threading.Thread | None = None
        self.prefill_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.started:
            raise RuntimeError("streaming pipeline is already started")
        self.backend.initialize_history()
        self.started = True
        self.started_at = time.perf_counter()
        self.encoder_thread = threading.Thread(
            target=self._encoder_worker,
            name="motiondrop-encoder",
            daemon=True,
        )
        self.prefill_thread = threading.Thread(
            target=self._prefill_worker,
            name="motiondrop-prefill",
            daemon=True,
        )
        self.encoder_thread.start()
        self.prefill_thread.start()

    def _record_error(self, stage: str, exc: BaseException) -> None:
        with self.condition:
            self.errors.append(
                {
                    "stage": stage,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            self.stop_event.set()
            self.condition.notify_all()

    def _put(self, target: queue.Queue, value: Any) -> bool:
        while not self.stop_event.is_set():
            try:
                target.put(value, timeout=_POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def _encoder_worker(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.input_queue.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    continue
                try:
                    if item is _STOP:
                        self._put(self.prefill_queue, _STOP)
                        return
                    encoded = self.backend.encode_window(item)
                    if not self._put(self.prefill_queue, encoded):
                        return
                finally:
                    self.input_queue.task_done()
        except BaseException as exc:
            self._record_error("encoder", exc)
            self._put(self.prefill_queue, _STOP)

    def _prefill_worker(self) -> None:
        expected_index = 0
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.prefill_queue.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    continue
                try:
                    if item is _STOP:
                        return
                    if item.index != expected_index:
                        raise RuntimeError(
                            f"out-of-order encoded window: {item.index} != {expected_index}"
                        )
                    prefill_started = time.perf_counter()
                    prefill_stats = self.backend.prefill_window(item)
                    prefill_finished = time.perf_counter()
                    event = {
                        "window_index": item.index,
                        "frame_indices": item.stats.get("frame_indices", []),
                        "submitted_at": item.submitted_at,
                        "encode_started_at": item.encode_started_at,
                        "encode_finished_at": item.encode_finished_at,
                        "prefill_started_at": prefill_started,
                        "prefill_finished_at": prefill_finished,
                        "encode_seconds": item.encode_finished_at - item.encode_started_at,
                        "queue_wait_seconds": prefill_started - item.encode_finished_at,
                        "prefill_seconds": prefill_finished - prefill_started,
                        "compression": item.stats,
                        "prefill": prefill_stats,
                    }
                    with self.condition:
                        self.events.append(event)
                        self.prefilled_windows += 1
                        expected_index += 1
                        self.condition.notify_all()
                finally:
                    self.prefill_queue.task_done()
        except BaseException as exc:
            self._record_error("prefill", exc)

    def _raise_if_failed(self) -> None:
        if self.errors:
            first = self.errors[0]
            raise RuntimeError(f"Streaming {first['stage']} worker failed: {first['error']}")

    def submit(self, frames: Sequence[Image.Image], frame_indices: Sequence[int]) -> int:
        if not self.started or self.input_closed:
            raise RuntimeError("pipeline must be running and open before submit")
        self._raise_if_failed()
        index = self.submitted_windows
        window = FrameWindow.create(index, frames, frame_indices)
        if not self._put(self.input_queue, window):
            self._raise_if_failed()
            raise RuntimeError("streaming pipeline stopped before window submission")
        self.submitted_windows += 1
        return index

    def close_input(self) -> None:
        if not self.started:
            raise RuntimeError("pipeline is not started")
        if self.input_closed:
            return
        self.input_closed = True
        if not self._put(self.input_queue, _STOP):
            self._raise_if_failed()

    def wait_until_prefilled(self, count: int, timeout: float | None = None) -> None:
        if count < 0 or count > self.submitted_windows:
            raise ValueError("invalid prefill barrier")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while self.prefilled_windows < count and not self.errors:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(f"timed out waiting for {count} prefilled windows")
                self.condition.wait(timeout=remaining)
        self._raise_if_failed()

    def finish(self) -> dict:
        self.close_input()
        self.wait_until_prefilled(self.submitted_windows)
        assert self.encoder_thread is not None and self.prefill_thread is not None
        self.encoder_thread.join(timeout=5.0)
        self.prefill_thread.join(timeout=5.0)
        if self.encoder_thread.is_alive() or self.prefill_thread.is_alive():
            raise RuntimeError("streaming workers did not terminate")
        self._raise_if_failed()
        self.finished_at = time.perf_counter()
        return self.stats()

    def query(self, query: str, *, max_new_tokens: int) -> tuple[str, dict]:
        if not query.strip():
            raise ValueError("query must not be empty")
        runtime_stats = self.finish()
        answer, query_stats = self.backend.answer(query, max_new_tokens=max_new_tokens)
        return answer, {"runtime": runtime_stats, "query": query_stats}

    def answer_current(self, query: str, *, max_new_tokens: int) -> tuple[str, dict]:
        """Answer against all submitted windows while keeping the stream open."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if not self.started or self.input_closed:
            raise RuntimeError("pipeline must be running and open before an interactive query")
        self.wait_until_prefilled(self.submitted_windows)
        answer, query_stats = self.backend.answer(query, max_new_tokens=max_new_tokens)
        return answer, {"runtime": self.stats(), "query": query_stats}

    def stats(self) -> dict:
        finished = self.finished_at or time.perf_counter()
        wall = None if self.started_at is None else finished - self.started_at
        sequential = sum(
            event["encode_seconds"] + event["prefill_seconds"] for event in self.events
        )
        active_wall = None
        if self.events:
            active_wall = (
                self.events[-1]["prefill_finished_at"] - self.events[0]["encode_started_at"]
            )
        return {
            "schema": "motiondrop_streaming_pipeline_v1",
            "submitted_windows": self.submitted_windows,
            "prefilled_windows": self.prefilled_windows,
            "pipeline_wall_seconds": wall,
            "active_pipeline_seconds": active_wall,
            "sequential_stage_seconds": sequential,
            "overlap_speedup": (
                None if not active_wall or active_wall <= 0.0 else sequential / active_wall
            ),
            "events": list(self.events),
            "errors": list(self.errors),
        }


__all__ = [
    "EncodedWindow",
    "FrameWindow",
    "StreamingBackend",
    "StreamingPipeline",
    "time_marker",
]
