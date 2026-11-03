"""Fast random video access with a timestamp-safe PyAV path."""

from __future__ import annotations

from pathlib import Path

import av
from PIL import Image


def probe_video(video_path: str | Path) -> tuple[int, float]:
    """Return an MP4's frame count and average frame rate."""

    try:
        from decord import VideoReader, cpu

        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        total_frames = len(reader)
        source_fps = float(reader.get_avg_fps() or 25.0)
        if total_frames > 0 and source_fps > 0.0:
            return total_frames, source_fps
    except Exception:
        pass

    av.logging.set_level(av.logging.PANIC)
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"Video contains no video stream: {video_path}")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or stream.base_rate or 25.0)
        total_frames = int(stream.frames or 0)
        if total_frames < 1 and stream.duration is not None:
            total_frames = int(round(float(stream.duration * stream.time_base) * source_fps))
        if total_frames < 1 and container.duration is not None:
            total_frames = int(round(container.duration / av.time_base * source_fps))
    if total_frames < 1:
        raise ValueError(f"Could not determine a positive frame count: {video_path}")
    if source_fps <= 0.0:
        raise ValueError(f"Could not determine a positive frame rate: {video_path}")
    return total_frames, source_fps


def decode_frame_indices(
    video_path: str | Path,
    frame_indices: list[int],
    source_fps: float,
    *,
    prefer_decord: bool = True,
) -> tuple[list[Image.Image], list[int], str]:
    """Seek to requested frames and return causal (never later) decoded frames."""

    if not frame_indices:
        raise ValueError("At least one frame index is required")
    if frame_indices != sorted(frame_indices) or len(frame_indices) != len(set(frame_indices)):
        raise ValueError("Frame indices must be sorted and unique")
    if frame_indices[0] < 0:
        raise ValueError("Frame indices must be non-negative")

    if prefer_decord:
        try:
            from decord import VideoReader, cpu

            reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
            if frame_indices[-1] >= len(reader):
                raise IndexError(
                    f"Frame {frame_indices[-1]} is outside a {len(reader)}-frame video"
                )
            arrays = reader.get_batch(frame_indices).asnumpy()
            images = [Image.fromarray(array).convert("RGB") for array in arrays]
            return images, frame_indices, "decord2"
        except Exception:
            pass
    return _decode_frame_indices_pyav(video_path, frame_indices, source_fps)


def _decode_frame_indices_pyav(
    video_path: str | Path,
    frame_indices: list[int],
    source_fps: float,
) -> tuple[list[Image.Image], list[int], str]:
    av.logging.set_level(av.logging.PANIC)
    images: list[Image.Image] = []
    decoded_indices: list[int] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        start_pts = int(stream.start_time or 0)
        for target_index in frame_indices:
            target_seconds = target_index / source_fps
            target_pts = start_pts + int(target_seconds / float(stream.time_base))
            container.seek(
                max(start_pts, target_pts),
                stream=stream,
                backward=True,
                any_frame=False,
            )
            previous: tuple[Image.Image, int] | None = None
            selected: tuple[Image.Image, int] | None = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                relative_seconds = float((frame.pts - start_pts) * stream.time_base)
                decoded_index = max(0, int(round(relative_seconds * source_fps)))
                current = (frame.to_image().convert("RGB"), decoded_index)
                if decoded_index >= target_index:
                    selected = (
                        current if decoded_index == target_index or previous is None else previous
                    )
                    break
                previous = current
            if selected is None:
                selected = previous
            if selected is None:
                raise ValueError(f"Could not decode frame {target_index} from video: {video_path}")
            image, decoded_index = selected
            if decoded_indices and decoded_index <= decoded_indices[-1]:
                raise ValueError(
                    "Video seek returned duplicate or out-of-order frames: "
                    f"requested={frame_indices}, decoded={decoded_indices + [decoded_index]}"
                )
            images.append(image)
            decoded_indices.append(decoded_index)
    return images, decoded_indices, "pyav"


__all__ = ["decode_frame_indices", "probe_video"]


class SequentialFrameReader:
    """Decode a video once, front to back, keeping only requested frame indices.

    Live streams only move forward, so one open decoder is ~10x faster than
    re-opening and seeking for every window (which cannot keep up with 1080p50
    video in real time).
    """

    def __init__(self, video_path: str | Path, *, start_frame: int = 0) -> None:
        av.logging.set_level(av.logging.PANIC)
        self.container = av.open(str(video_path))
        stream = self.container.streams.video[0]
        stream.thread_type = "AUTO"
        self.position = 0
        self.exhausted = False
        # Seeking lands on the preceding keyframe; frame indices are then
        # recovered from timestamps (exact for constant-frame-rate video).
        self._fps = float(stream.average_rate or stream.guessed_rate or 0) or None
        self._origin = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
        self._timestamp_indices = start_frame > 0 and self._fps is not None
        if self._timestamp_indices:
            target = self._origin + start_frame / self._fps
            self.container.seek(int(target / stream.time_base), stream=stream, backward=True)
        self._frames = self.container.decode(stream)

    def read(self, frame_indices: list[int]) -> tuple[list[Image.Image], list[int]]:
        if not frame_indices:
            raise ValueError("At least one frame index is required")
        if frame_indices != sorted(frame_indices) or len(frame_indices) != len(set(frame_indices)):
            raise ValueError("Frame indices must be sorted and unique")
        if frame_indices[0] < self.position:
            raise ValueError(
                f"frame {frame_indices[0]} was already passed (decoder at {self.position})"
            )
        images: list[Image.Image] = []
        decoded: list[int] = []
        wanted = iter(frame_indices)
        target = next(wanted)
        while target is not None and not self.exhausted:
            try:
                frame = next(self._frames)
            except StopIteration:
                self.exhausted = True
                break
            if self._timestamp_indices and frame.time is not None:
                index = int(round((frame.time - self._origin) * self._fps))
                self.position = index + 1
            else:
                index = self.position
                self.position += 1
            while target is not None and index > target:
                # A requested frame was skipped (e.g. landing past it after a
                # seek); substitute the next decoded frame causally.
                images.append(frame.to_image().convert("RGB"))
                decoded.append(index)
                target = next(wanted, None)
                index = -1
            if index == target:
                images.append(frame.to_image().convert("RGB"))
                decoded.append(index)
                target = next(wanted, None)
        if not images:
            raise IndexError(f"video ended before frame {frame_indices[0]}")
        return images, decoded

    def close(self) -> None:
        self.container.close()
