"""Headless interactive web demo for the MotionDrop live multi-task runtime."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from motiondrop.models import load_model_and_processor
from motiondrop.paths import sample_video
from motiondrop.streaming_runtime.config import RuntimeConfig
from motiondrop.streaming_runtime.latency import DEFAULT_SLO_MS, LatencyModel
from motiondrop.streaming_runtime.live import INTENTS, EventBus, LiveConfig, LiveScheduler
from motiondrop.streaming_runtime.llava import LlavaStreamingBackend, resolve_local_checkpoint
from motiondrop.streaming_runtime.run import sampled_indices
from motiondrop.video import SequentialFrameReader, probe_video

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motiondrop demo",
        description="Serve the interactive MotionDrop live demo on a headless host.",
    )
    parser.add_argument("--video", type=Path, default=sample_video())
    parser.add_argument("--model-path")
    parser.add_argument(
        "--latency-model",
        type=Path,
        help="Fitted latency profile; omit to use --fixed-compression instead.",
    )
    parser.add_argument("--slo-ms", type=float, default=DEFAULT_SLO_MS)
    parser.add_argument("--fixed-compression", type=float, default=0.9)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument(
        "--max-stream-frames", type=int, default=0, help="0 streams the whole video."
    )
    parser.add_argument("--window-frames", type=int, default=2)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-context-tokens", type=int, default=16000)
    parser.add_argument("--position-mode", choices=("contiguous", "absolute"), default="contiguous")
    parser.add_argument("--no-time-markers", action="store_true")
    parser.add_argument("--standing-interval", type=float, default=4.0)
    parser.add_argument("--watch-interval", type=float, default=1.0)
    parser.add_argument(
        "--memory-interval",
        type=float,
        default=0.0,
        help="Background note-taking period in seconds (0 disables it).",
    )
    parser.add_argument(
        "--video-root",
        dest="video_roots",
        type=Path,
        action="append",
        default=[],
        help="Directory the browser may load videos from (repeatable). "
        "Without it any local video file can be selected.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-gpu-memory-gib", type=int)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if args.model_path is not None and not Path(args.model_path).is_dir():
        parser.error(f"model path does not exist: {args.model_path}")
    if args.latency_model is not None and not args.latency_model.is_file():
        parser.error(f"latency model does not exist: {args.latency_model}")
    if args.slo_ms <= 0 or args.fps <= 0:
        parser.error("--slo-ms and --fps must be positive")
    if args.max_stream_frames < 0 or not 1 <= args.port <= 65535:
        parser.error("--max-stream-frames and --port are invalid")
    return args


class InteractiveSession:
    """One browser-controlled chronological video stream with its tasks."""

    def __init__(self, state: DemoState) -> None:
        self.state = state
        self.lock = threading.RLock()
        self.cursor = 0
        self.video = state.video
        self.source_fps = state.source_fps
        self.frame_indices = list(state.frame_indices)
        self.reader = SequentialFrameReader(self.video)
        backend = LlavaStreamingBackend(
            state.model,
            state.processor,
            config=state.config,
            latency_model=state.latency_model,
            slo_ms=None if state.latency_model is None else state.args.slo_ms,
        )
        self.scheduler = LiveScheduler(backend, config=state.live_config, bus=state.bus)
        self.scheduler.start()
        self.closed = False

    def _advance_locked(self, seconds: float, *, flush: bool = False) -> None:
        if self.closed:
            raise RuntimeError("session is closed")
        frame_limit = max(0, int(seconds * self.source_fps))
        end = self.cursor
        while end < len(self.frame_indices):
            if self.frame_indices[end] > frame_limit:
                break
            end += 1
        available = end - self.cursor
        if not flush:
            available -= available % self.state.config.window_frames
        submit_end = self.cursor + available
        new_indices = self.frame_indices[self.cursor : submit_end]
        for start in range(0, len(new_indices), self.state.config.window_frames):
            chunk = new_indices[start : start + self.state.config.window_frames]
            frames, decoded = self.reader.read(chunk)
            timestamps = [index / self.source_fps for index in decoded]
            self.scheduler.submit(frames, decoded, timestamps)
        self.cursor = submit_end

    def advance(self, seconds: float) -> dict:
        with self.lock:
            self._advance_locked(seconds)
        return self.status()

    def message(self, text: str, seconds: float, kind: str) -> dict:
        if not text.strip():
            raise ValueError("message must not be empty")
        with self.lock:
            self._advance_locked(seconds, flush=True)
        # The GPU worker serialises everything; do not hold the session lock
        # while waiting so the browser can keep advancing the stream.
        response = self.scheduler.handle_message(text, seconds, kind=kind, timeout=300.0)
        return {**response, "status": self.status()}

    def cancel(self, task_id: str) -> dict:
        return {"task": self.scheduler.cancel_task(task_id), "status": self.status()}

    def status(self) -> dict:
        live = self.scheduler.status()
        return {
            "active": not self.closed,
            "processed_frames": self.cursor,
            "available_frames": len(self.frame_indices),
            **live,
        }

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.scheduler.abort()
            self.reader.close()
            self.closed = True


class DemoState:
    """Model-owning process state; one interactive browser session at a time."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        checkpoint = resolve_local_checkpoint(args.model_path)
        self.model, self.processor, self.spec, self.checkpoint = load_model_and_processor(
            "llava-ov-7b",
            checkpoint,
            max_gpu_memory_gib=args.max_gpu_memory_gib,
            local_files_only=True,
        )
        self.latency_model = (
            None if args.latency_model is None else LatencyModel.load(args.latency_model)
        )
        self.config = RuntimeConfig(
            window_frames=args.window_frames,
            queue_size=args.queue_size,
            fixed_compression=args.fixed_compression,
            max_new_tokens=args.max_new_tokens,
            time_markers=not args.no_time_markers,
            max_context_tokens=args.max_context_tokens or None,
            position_mode=args.position_mode,
        )
        self.live_config = LiveConfig(
            tick_policy="realtime",
            standing_interval_s=args.standing_interval,
            watch_interval_s=args.watch_interval,
            memory_interval_s=args.memory_interval,
            instant_max_new_tokens=args.max_new_tokens,
        )
        self.video_roots = [root.resolve() for root in args.video_roots]
        self.bus = EventBus()
        self.lock = threading.RLock()
        self.session: InteractiveSession | None = None
        self.video_version = 0
        self._load_video(args.video)

    def _load_video(self, path: Path) -> None:
        total_frames, source_fps = probe_video(path)
        if total_frames < 1 or source_fps <= 0:
            raise ValueError(f"not a readable video: {path}")
        self.video = path
        self.total_frames, self.source_fps = total_frames, source_fps
        self.frame_indices = sampled_indices(
            total_frames,
            source_fps,
            target_fps=self.args.fps,
            query_at_seconds=None,
            maximum=self.args.max_stream_frames or None,
        )
        self.duration_seconds = total_frames / source_fps
        self.video_version += 1

    def set_video(self, raw_path: str) -> dict:
        """Switch the demo to another local video; closes the current session."""

        path = Path(raw_path.strip()).expanduser()
        if not raw_path.strip() or not path.is_absolute():
            raise ValueError("please give an absolute path to a video file")
        path = path.resolve()
        if path.is_dir() and (path / "video.mp4").is_file():
            path = path / "video.mp4"
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(
                f"unsupported video type {path.suffix!r}; use {sorted(VIDEO_SUFFIXES)}"
            )
        if not path.is_file():
            raise ValueError(f"video does not exist: {path}")
        if self.video_roots and not any(path.is_relative_to(root) for root in self.video_roots):
            raise ValueError("video is outside the allowed --video-root directories")
        with self.lock:
            if self.session is not None:
                self.session.close()
                self.session = None
            try:
                self._load_video(path)
            except Exception as exc:
                raise ValueError(f"cannot read video {path}: {exc}") from exc
            self.bus.publish("video_changed", video=str(path))
            return self.public_config()

    def start_session(self) -> dict:
        with self.lock:
            if self.session is not None:
                self.session.close()
            self.session = InteractiveSession(self)
            self.bus.publish("session_started")
            return self.session.status()

    def require_session(self) -> InteractiveSession:
        with self.lock:
            if self.session is None or self.session.closed:
                raise RuntimeError("start a session before playing or asking")
            return self.session

    def public_config(self) -> dict:
        return {
            "model": self.spec.alias,
            "video": self.video.name,
            "video_path": str(self.video),
            "video_version": self.video_version,
            "duration_seconds": self.duration_seconds,
            "source_fps": self.source_fps,
            "stream_fps": self.args.fps,
            "max_stream_frames": len(self.frame_indices),
            "max_stream_seconds": self.frame_indices[-1] / self.source_fps,
            "window_frames": self.config.window_frames,
            "budget": (
                f"SLO {self.args.slo_ms:g} ms"
                if self.latency_model is not None
                else f"{self.config.fixed_compression:.0%} compression"
            ),
            "max_context_tokens": self.config.max_context_tokens,
            "intents": ["auto", *INTENTS],
        }


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length < 1 or length > 1_000_000:
        raise ValueError("invalid JSON request size")
    value = json.loads(handler.rfile.read(length))
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def _serve_video(handler: BaseHTTPRequestHandler, path: Path, *, send_body: bool) -> None:
    size = path.stat().st_size
    start, end = 0, size - 1
    status = HTTPStatus.OK
    byte_range = handler.headers.get("Range")
    if byte_range:
        if not byte_range.startswith("bytes=") or "," in byte_range:
            handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        first, _, last = byte_range.removeprefix("bytes=").partition("-")
        if first:
            start = int(first)
            end = min(size - 1, int(last)) if last else size - 1
        elif last:
            start = max(0, size - int(last))
        if start < 0 or start > end or start >= size:
            handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        status = HTTPStatus.PARTIAL_CONTENT
    handler.send_response(status)
    handler.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(end - start + 1))
    if status == HTTPStatus.PARTIAL_CONTENT:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    if not send_body:
        return
    remaining = end - start + 1
    with path.open("rb") as source:
        source.seek(start)
        while remaining:
            block = source.read(min(1024 * 1024, remaining))
            if not block:
                break
            handler.wfile.write(block)
            remaining -= len(block)


def _stream_events(handler: BaseHTTPRequestHandler, bus: EventBus, since: int) -> None:
    """Server-sent events; the browser's EventSource reconnects with Last-Event-ID."""

    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()
    try:
        while True:
            events = bus.wait(since, timeout=15.0)
            if not events:
                handler.wfile.write(b": keep-alive\n\n")
            for event in events:
                payload = json.dumps(event, ensure_ascii=False)
                handler.wfile.write(f"id: {event['seq']}\ndata: {payload}\n\n".encode())
                since = event["seq"]
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        return


def make_handler(state: DemoState, html_path: Path):
    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "MotionDropDemo/0.2"

        def _json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, exc: BaseException) -> None:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_HEAD(self) -> None:  # noqa: N802
            if urlparse(self.path).path == "/video":
                _serve_video(self, state.video, send_body=False)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path
            if route == "/":
                payload = html_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif route == "/video":
                _serve_video(self, state.video, send_body=True)
            elif route == "/api/config":
                self._json(state.public_config())
            elif route == "/api/status":
                session = state.session
                self._json({"active": False} if session is None else session.status())
            elif route == "/api/events":
                query = parse_qs(parsed.query)
                since = self.headers.get("Last-Event-ID") or query.get("since", ["0"])[0]
                try:
                    since_value = int(since)
                except ValueError:
                    since_value = 0
                _stream_events(self, state.bus, since_value)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            try:
                data = _read_json(self)
                if route == "/api/start":
                    self._json(state.start_session())
                elif route == "/api/advance":
                    self._json(state.require_session().advance(float(data["seconds"])))
                elif route == "/api/message":
                    self._json(
                        state.require_session().message(
                            str(data["text"]),
                            float(data["seconds"]),
                            str(data.get("kind", "auto")),
                        )
                    )
                elif route == "/api/query":
                    response = state.require_session().message(
                        str(data["question"]), float(data["seconds"]), "instant"
                    )
                    self._json({"exchange": response["answer"], "status": response["status"]})
                elif route == "/api/video":
                    self._json(state.set_video(str(data["path"])))
                elif route == "/api/tasks/cancel":
                    self._json(state.require_session().cancel(str(data["task_id"])))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                self._error(exc)

        def log_message(self, format: str, *args) -> None:
            if "/api/advance" in self.requestline or "/api/status" in self.requestline:
                return
            print(f"[web] {self.address_string()} {format % args}", flush=True)

    return DemoHandler


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    html_path = Path(__file__).resolve().parents[3] / "streaming_demo" / "index.html"
    if not html_path.is_file():
        raise FileNotFoundError(f"demo UI is missing: {html_path}")
    print("Loading LLaVA-OV-7B for the live demo...", flush=True)
    state = DemoState(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, html_path))
    server.daemon_threads = True
    print(f"MotionDrop demo: http://{args.host}:{args.port}", flush=True)
    print(
        f"Video={state.video} | {state.public_config()['budget']} | streaming "
        f"{len(state.frame_indices)} frames at {args.fps:g} fps",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if state.session is not None:
            state.session.close()


if __name__ == "__main__":
    main()
