"""Live multi-task runtime: long-running tasks and instant questions on one stream.

One GPU worker thread owns the KV trunk. It serves three kinds of work:

* instant jobs (questions, intent routing): highest priority, may preempt;
* video windows: prefilled into the trunk in order, oldest evicted when full;
* periodic tasks ("standing" updates such as live commentary, and "watch"
  conditions that fire an alert): run when the trunk is caught up.

Every task runs as a branch on top of the trunk and is rolled back, so tasks are
isolated from each other. Task outputs are kept as text in a timestamped memory
log that instant questions can retrieve after the video itself was evicted.
"""

from __future__ import annotations

import contextlib
import math
import queue
import re
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from PIL import Image

from motiondrop.streaming_runtime.pipeline import (
    _POLL_SECONDS,
    EncodedWindow,
    FrameWindow,
    time_marker,
)

TICK_POLICIES = ("realtime", "lockstep")
STANDING_STYLES = ("incremental", "describe")
TASK_KINDS = ("standing", "watch")
INTENTS = ("instant", "standing", "watch", "update", "cancel")
SILENT = "<SILENT>"
_BRACKETED = re.compile(r"\s*[<\[（(【][^<>\[\]]{0,12}[>\]）)】]\s*[.。]?\s*")
CHOICE_LETTERS = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True)
class LiveConfig:
    """Scheduling and prompting controls for the live runtime."""

    tick_policy: str = "realtime"
    standing_interval_s: float = 4.0
    watch_interval_s: float = 1.0
    watch_threshold: float = 0.5
    watch_cooldown_s: float = 8.0
    standing_max_new_tokens: int = 40
    instant_max_new_tokens: int = 96
    alert_max_new_tokens: int = 32
    repetition_penalty: float = 1.2
    # A standing update this similar to a recent one is treated as silence.
    duplicate_similarity: float = 0.6
    # Recent standing outputs checked for near-duplicates.
    history_lines: int = 8
    # Previous lines shown in the standing prompt. Showing them makes the model
    # copy (and propagate hallucinated names); 0 relies on dedup instead.
    prompt_history_lines: int = 0
    standing_style: str = "describe"
    memory_entries: int = 6
    memory_interval_s: float = 0.0
    use_memory: bool = True
    # "evicted": questions only see notes about video no longer in the KV
    # (while video is resident the model should look at it, not at possibly
    # wrong notes); "all": every note is eligible.
    memory_scope: str = "evicted"
    dialog_turns: int = 3
    max_backlog_windows: int = 64
    # Legacy shared history: write task outputs into the trunk instead of a branch.
    commit_outputs: bool = False

    def __post_init__(self) -> None:
        if self.tick_policy not in TICK_POLICIES:
            raise ValueError(f"tick_policy must be one of {TICK_POLICIES}")
        for name in ("standing_interval_s", "watch_interval_s", "repetition_penalty"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.duplicate_similarity <= 1.0:
            raise ValueError("duplicate_similarity must be in (0, 1]")
        if not 0.0 < self.watch_threshold < 1.0:
            raise ValueError("watch_threshold must be in (0, 1)")
        if self.memory_interval_s < 0.0 or self.watch_cooldown_s < 0.0:
            raise ValueError("memory_interval_s and watch_cooldown_s must be non-negative")
        if self.standing_style not in STANDING_STYLES:
            raise ValueError(f"standing_style must be one of {STANDING_STYLES}")
        if self.memory_scope not in ("evicted", "all"):
            raise ValueError("memory_scope must be 'evicted' or 'all'")
        if self.max_backlog_windows < 1:
            raise ValueError("max_backlog_windows must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


def format_clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    whole = max(0, int(round(seconds)))
    return f"{whole // 60:02d}:{whole % 60:02d}"


# ---------------------------------------------------------------------------
# Memory log
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[㐀-鿿]")


def lexical_keys(text: str) -> set[str]:
    """Latin words plus CJK character bigrams (and single characters)."""

    lowered = text.lower()
    keys = {word for word in _WORD.findall(lowered) if len(word) > 1}
    chars = _CJK.findall(lowered)
    keys.update(chars)
    keys.update(a + b for a, b in zip(chars, chars[1:], strict=False))
    return keys


@dataclass(frozen=True)
class MemoryEntry:
    seconds: float
    source: str
    text: str


class MemoryLog:
    """Timestamped text notes written by tasks; survives KV eviction."""

    def __init__(self, limit: int = 5000) -> None:
        self.entries: deque[MemoryEntry] = deque(maxlen=limit)
        self.lock = threading.Lock()

    def add(self, seconds: float, source: str, text: str) -> None:
        if text.strip():
            with self.lock:
                self.entries.append(MemoryEntry(float(seconds), source, text.strip()))

    def __len__(self) -> int:
        return len(self.entries)

    def retrieve(
        self,
        query: str,
        *,
        limit: int,
        recent: int = 2,
        before: float | None = None,
    ) -> list[MemoryEntry]:
        """Most recent ``recent`` notes plus the best lexical matches, in time order.

        ``before`` restricts the search to notes written before that video time.
        """

        with self.lock:
            entries = list(self.entries)
        if before is not None:
            entries = [entry for entry in entries if entry.seconds < before]
        if not entries or limit < 1:
            return []
        chosen = {id(entry): entry for entry in entries[-min(recent, limit) :]}
        keys = lexical_keys(query)
        if keys:
            newest = entries[-1].seconds
            scored = []
            for entry in entries:
                overlap = len(keys & lexical_keys(entry.text))
                if overlap:
                    recency = 1.0 / (1.0 + max(0.0, newest - entry.seconds) / 600.0)
                    scored.append((overlap + 0.5 * recency, entry))
            scored.sort(key=lambda item: item[0], reverse=True)
            for _, entry in scored:
                if len(chosen) >= limit:
                    break
                chosen.setdefault(id(entry), entry)
        return sorted(chosen.values(), key=lambda entry: entry.seconds)


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


class EventBus:
    """Append-only event log that HTTP clients can long-poll or stream."""

    def __init__(self, limit: int = 5000) -> None:
        self.events: deque[dict] = deque(maxlen=limit)
        self.sequence = 0
        self.condition = threading.Condition()

    def publish(self, event_type: str, **data: Any) -> dict:
        with self.condition:
            self.sequence += 1
            event = {"seq": self.sequence, "type": event_type, "wall_time": time.time(), **data}
            self.events.append(event)
            self.condition.notify_all()
            return event

    def since(self, sequence: int) -> list[dict]:
        with self.condition:
            return [event for event in self.events if event["seq"] > sequence]

    def wait(self, sequence: int, timeout: float) -> list[dict]:
        with self.condition:
            if self.sequence <= sequence:
                self.condition.wait(timeout=timeout)
            return [event for event in self.events if event["seq"] > sequence]


# ---------------------------------------------------------------------------
# Tasks and jobs
# ---------------------------------------------------------------------------


@dataclass
class LiveTask:
    id: str
    kind: str
    instruction: str
    created_seconds: float
    interval_s: float
    next_due: float
    last_covered: float
    hidden: bool = False
    once: bool = False
    check_prompt: str | None = None
    output_prompt: str | None = None
    status: str = "active"
    amendments: list[str] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    runs: int = 0
    silent: int = 0
    duplicates: int = 0
    preempted: int = 0
    dropped: int = 0
    checks: int = 0
    check_trace: list[tuple[float, float]] = field(default_factory=list)
    last_fired: float | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "instruction": self.instruction,
            "amendments": list(self.amendments),
            "status": self.status,
            "created_seconds": self.created_seconds,
            "interval_s": self.interval_s,
            "runs": self.runs,
            "outputs": len(self.outputs),
            "silent": self.silent,
            "duplicates": self.duplicates,
            "preempted": self.preempted,
            "dropped": self.dropped,
            "checks": self.checks,
        }


@dataclass
class _Job:
    kind: str
    run: Callable[[], Any]
    required_windows: int
    submitted_at: float = field(default_factory=time.perf_counter)
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


_STOP = object()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _memory_block(entries: Sequence[MemoryEntry]) -> str:
    if not entries:
        return ""
    lines = "\n".join(f"[{format_clock(entry.seconds)}] {entry.text}" for entry in entries)
    return f"Notes written earlier during this stream (may be incomplete):\n{lines}\n\n"


def language_instruction(text: str, *, brief: bool = False) -> str:
    """Explicit output-language instruction; the model ignores "same language" hints."""

    if _CJK.search(text):
        if brief:
            return "请用中文回答，不超过30个字，直接描述动作，不要以“在这张图片中”“在最新的画面中”开头。"
        return "请用中文回答。"
    if brief:
        return "Answer in English, at most 20 words, starting directly with the action."
    return "Answer in English."


def similarity(first: str, second: str) -> float:
    """Jaccard overlap of lexical keys (CJK bigrams and Latin words)."""

    a, b = lexical_keys(first), lexical_keys(second)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def standing_prompt(
    task: LiveTask,
    now: float,
    history_lines: int,
    *,
    time_markers: bool = False,
    style: str = "incremental",
) -> str:
    newest = "the newest frames" + (
        f" (those after {time_marker(task.last_covered)})" if time_markers else ""
    )
    if style == "describe":
        lines = [f'Task from the user: "{task.instruction}"']
        lines.extend(f'Additional requirement: "{text}"' for text in task.amendments)
        if history_lines > 0 and task.outputs:
            lines.append(
                "Already said (do not repeat): "
                + " / ".join(item["text"] for item in task.outputs[-history_lines:])
            )
        lines.append(
            f"As a live commentator, describe in one short sentence what is happening in "
            f"{newest}: the specific action, who is doing it, or a change of score or "
            "on-screen text. Do not mention the time or the video."
        )
        lines.append(language_instruction(task.instruction, brief=True))
        return "\n".join(lines)
    lines = [
        "You are performing a continuous task on a live video stream.",
        f'Task from the user: "{task.instruction}"',
    ]
    for amendment in task.amendments:
        lines.append(f'Additional requirement from the user: "{amendment}"')
    recent = [output for output in task.outputs[-history_lines:]]
    if recent:
        lines.append("Your previous lines, oldest first:")
        lines.extend(f"- {item['text']}" for item in recent)
    lines.append(
        f"Write the next line for this task based on {newest}: one short, specific sentence about a concrete event, such as an action, who "
        "did what, or a change of score or on-screen text. Do not mention the time, the "
        "video or your previous lines, and do not repeat them. If nothing new is "
        f"happening, output exactly {SILENT}"
    )
    lines.append(language_instruction(task.instruction))
    return "\n".join(lines)


def watch_check_prompt(task: LiveTask, now: float) -> str:
    if task.check_prompt:
        return task.check_prompt
    return (
        f'The user asked: "{task.instruction}"\n'
        f"The video is now at {format_clock(now)}. Looking at the most recent frames, is the "
        "condition in the user's request happening right now? Answer Yes or No."
    )


def watch_output_prompt(task: LiveTask, now: float) -> str:
    if task.output_prompt:
        return task.output_prompt
    example = "某某刚刚……！" if _CJK.search(task.instruction) else "<who> just <did what>!"
    return (
        f'The user asked to be alerted: "{task.instruction}"\n'
        "This has just happened in the newest frames. Write the alert: one short sentence "
        f'saying what just happened, in the form "{example}". Do not repeat the request. '
        + language_instruction(task.instruction)
    )


def instant_prompt(
    question: str,
    now: float,
    *,
    memory: Sequence[MemoryEntry],
    dialog: Sequence[dict],
    plain: bool,
) -> str:
    if plain:
        return question
    parts = [_memory_block(memory)]
    if dialog:
        turns = "\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in dialog)
        parts.append(f"Earlier questions and your answers:\n{turns}\n\n")
    parts.append(f"The video is now at {format_clock(now)}.\n{question}")
    if not any(marker in question for marker in ("option", "letter")):
        parts.append("\n" + language_instruction(question))
    return "".join(parts)


_ROUTE_RULES: tuple[tuple[str, re.Pattern], ...] = (
    (
        "cancel",
        re.compile(
            r"(停止|停下|别再|不要再|取消|结束|关掉|关闭)(解说|描述|播报|提醒|任务|了)?|^停|"
            r"(别|不要|不用)(再)?(解说|描述|播报|提醒)|"
            r"\b(stop|cancel|quit)\b",
            re.I,
        ),
    ),
    (
        "watch",
        re.compile(
            r"(提醒我|通知我|叫我|告诉我一声)|"
            r"\b(alert|notify|remind)\b|let me know|tell me when",
            re.I,
        ),
    ),
    (
        "update",
        re.compile(
            r"(多|少|更|重点|侧重|着重|别|不要)(一些|一点|点)?(关注|讲|说|聊|描述|介绍|简短|详细|啰嗦)|"
            r"(简短|详细|短)一(点|些)|\b(focus (more )?on|shorter|more about|less about)\b",
            re.I,
        ),
    ),
    (
        "standing",
        re.compile(
            r"(解说|实况|播报|持续|一直|不断|实时)(描述|讲|解说|报道|告诉|说)?|"
            r"commentat|narrat|play-by-play|keep (describ|telling|track)|live (commentary|description)",
            re.I,
        ),
    ),
    (
        "instant",
        re.compile(
            r"[?？]\s*$|多少|什么|啥|谁|哪|吗|呢|几|怎么|为什么|是否|有没有|"
            r"^(what|who|where|why|how|is|are|was|were|did|does|do|which|can|could)\b",
            re.I,
        ),
    ),
    (
        "watch",
        re.compile(
            r"(当|一旦|每当|如果).{1,30}(时|的时候|之后|就)|^(when|once|if|whenever)\b", re.I
        ),
    ),
)


_WATCH_WORDS = re.compile(r"提醒|通知|alert|notif|remind", re.I)
_STANDING_WORDS = re.compile(r"解说|描述|播报|commentat|narrat|describ", re.I)


def rule_route(text: str, *, has_task: bool) -> str | None:
    """Keyword routing; ``None`` when no rule applies (the model decides)."""

    for intent, pattern in _ROUTE_RULES:
        if intent in ("update", "cancel") and not has_task:
            continue
        if pattern.search(text.strip()):
            return intent
    return None


ROUTER_PROMPT = (
    "Classify the user's latest request about the live video.\n"
    "A: a one-time question that should be answered right now\n"
    "B: an ongoing task that needs repeated updates while the video plays "
    "(e.g. commentate, keep describing, narrate)\n"
    "C: a request to be notified when a specific thing happens later\n"
    "D: a change to an ongoing task (e.g. focus more on something, be shorter)\n"
    "E: a request to stop an ongoing task\n"
    'Examples: "what is the score now?" -> A; "commentate this match" -> B; '
    '"tell me when someone scores" -> C; "talk more about defence" -> D; '
    '"stop commentating" -> E\n'
    'Request: "{text}"\n'
    "Answer with one letter."
)
ROUTER_LABELS = {"A": "instant", "B": "standing", "C": "watch", "D": "update", "E": "cancel"}


def mc_letter(scores: Sequence[float]) -> str:
    return CHOICE_LETTERS[max(range(len(scores)), key=lambda index: scores[index])]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class LiveScheduler:
    """Owns the stream: encoder thread + one GPU worker serving all tasks."""

    def __init__(self, backend, *, config: LiveConfig, bus: EventBus | None = None) -> None:
        self.backend = backend
        self.config = config
        self.bus = bus or EventBus()
        self.memory = MemoryLog()
        self.input_queue: queue.Queue = queue.Queue(maxsize=config.max_backlog_windows)
        self.condition = threading.Condition()
        self.encoded: deque[EncodedWindow] = deque()
        self.jobs: deque[_Job] = deque()
        self.tasks: dict[str, LiveTask] = {}
        self.dialog: list[dict] = []
        self.errors: list[dict] = []
        self.window_log: list[dict] = []
        self.window_submitted_at: dict[int, float] = {}
        self.submitted_windows = 0
        self.prefilled_windows = 0
        self.processed_windows = 0
        self.video_seconds = 0.0
        self.last_window_submitted_at: float | None = None
        self.gpu_busy_seconds = 0.0
        self.trunk_snapshot: dict = {}
        self.stopping = False
        self.input_closed = False
        self.encoder_done = False
        self.started = False
        self.encoder_thread: threading.Thread | None = None
        self.gpu_thread: threading.Thread | None = None
        if config.memory_interval_s > 0.0:
            self._add_task_locked(
                "standing",
                "Briefly note what is happening in the video: who, what, and any score "
                "or text shown on screen.",
                0.0,
                interval_s=config.memory_interval_s,
                hidden=True,
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self.started:
            raise RuntimeError("live scheduler already started")
        self.backend.initialize_history()
        self.started = True
        self.encoder_thread = threading.Thread(
            target=self._encoder_worker, name="motiondrop-live-encoder", daemon=True
        )
        self.gpu_thread = threading.Thread(
            target=self._gpu_worker, name="motiondrop-live-gpu", daemon=True
        )
        self.encoder_thread.start()
        self.gpu_thread.start()

    def _record_error(self, stage: str, exc: BaseException) -> None:
        with self.condition:
            self.errors.append(
                {"stage": stage, "error": repr(exc), "traceback": traceback.format_exc()}
            )
            self.stopping = True
            self.condition.notify_all()
        self.bus.publish("error", stage=stage, error=repr(exc))

    def _raise_if_failed(self) -> None:
        if self.errors:
            first = self.errors[0]
            raise RuntimeError(f"live {first['stage']} worker failed: {first['error']}")

    def close(self, timeout: float = 30.0) -> None:
        """Stop accepting video, drain queued windows and stop the workers."""

        if not self.started:
            return
        if not self.input_closed:
            self.input_closed = True
            while not self.stopping:
                try:
                    self.input_queue.put(_STOP, timeout=_POLL_SECONDS)
                    break
                except queue.Full:
                    continue
        with self.condition:
            deadline = time.monotonic() + timeout
            while not self.stopping and (
                not self.encoder_done
                or self.encoded
                or self.jobs
                or self.prefilled_windows < self.submitted_windows
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=min(remaining, _POLL_SECONDS))
            self.stopping = True
            self.condition.notify_all()
        for thread in (self.encoder_thread, self.gpu_thread):
            if thread is not None:
                thread.join(timeout=5.0)

    def abort(self) -> None:
        """Stop immediately without draining (e.g. the viewer seeked backwards)."""

        if not self.started:
            return
        self.input_closed = True
        with self.condition:
            self.stopping = True
            for job in self.jobs:
                job.error = RuntimeError("live session was aborted")
                job.done.set()
            self.jobs.clear()
            self.condition.notify_all()
        with contextlib.suppress(queue.Full):
            self.input_queue.put_nowait(_STOP)
        for thread in (self.encoder_thread, self.gpu_thread):
            if thread is not None:
                thread.join(timeout=30.0)

    # -- video input -------------------------------------------------------

    def submit(
        self,
        frames: Sequence[Image.Image],
        frame_indices: Sequence[int],
        timestamps: Sequence[float],
    ) -> int:
        if not self.started or self.input_closed:
            raise RuntimeError("live scheduler must be running and open before submit")
        self._raise_if_failed()
        index = self.submitted_windows
        window = FrameWindow.create(index, frames, frame_indices, timestamps)
        while True:
            try:
                self.input_queue.put(window, timeout=_POLL_SECONDS)
                break
            except queue.Full:
                self._raise_if_failed()
        with self.condition:
            self.window_submitted_at[index] = window.submitted_at
            self.submitted_windows += 1
            self.condition.notify_all()
        return index

    def _encoder_worker(self) -> None:
        try:
            while not self.stopping:
                try:
                    item = self.input_queue.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    continue
                if item is _STOP:
                    break
                encoded = self.backend.encode_window(item)
                with self.condition:
                    while (
                        len(self.encoded) >= self.config.max_backlog_windows and not self.stopping
                    ):
                        self.condition.wait(timeout=_POLL_SECONDS)
                    if self.stopping:
                        return
                    self.encoded.append(encoded)
                    self.condition.notify_all()
        except BaseException as exc:
            self._record_error("encoder", exc)
        finally:
            with self.condition:
                self.encoder_done = True
                self.condition.notify_all()

    # -- task management ---------------------------------------------------

    def _add_task_locked(
        self,
        kind: str,
        instruction: str,
        video_seconds: float,
        *,
        interval_s: float | None = None,
        hidden: bool = False,
        once: bool = False,
        check_prompt: str | None = None,
        output_prompt: str | None = None,
    ) -> LiveTask:
        if kind not in TASK_KINDS:
            raise ValueError(f"task kind must be one of {TASK_KINDS}")
        if not instruction.strip():
            raise ValueError("task instruction must not be empty")
        interval = interval_s or (
            self.config.standing_interval_s if kind == "standing" else self.config.watch_interval_s
        )
        task = LiveTask(
            id=uuid.uuid4().hex[:8],
            kind=kind,
            instruction=instruction.strip(),
            created_seconds=float(video_seconds),
            interval_s=float(interval),
            next_due=float(video_seconds) + float(interval),
            last_covered=float(video_seconds),
            hidden=hidden,
            once=once,
            check_prompt=check_prompt,
            output_prompt=output_prompt,
        )
        self.tasks[task.id] = task
        return task

    def add_task(self, kind: str, instruction: str, video_seconds: float, **options) -> dict:
        with self.condition:
            task = self._add_task_locked(kind, instruction, video_seconds, **options)
            self.condition.notify_all()
            public = task.public()
        self.bus.publish("task_created", task=public, video_seconds=float(video_seconds))
        return public

    def _latest_user_task(self, kind: str | None = None) -> LiveTask | None:
        active = [
            task
            for task in self.tasks.values()
            if task.status == "active" and not task.hidden and kind in (None, task.kind)
        ]
        return active[-1] if active else None

    def _task_for_message(self, text: str, *, prefer: str | None) -> LiveTask | None:
        """The task a follow-up message refers to: by keyword, else by preference."""

        if _WATCH_WORDS.search(text):
            kind = "watch"
        elif _STANDING_WORDS.search(text):
            kind = "standing"
        else:
            kind = prefer
        return self._latest_user_task(kind) or self._latest_user_task()

    def update_task(self, text: str, task_id: str | None = None) -> dict | None:
        with self.condition:
            task = (
                self.tasks.get(task_id)
                if task_id
                else self._task_for_message(text, prefer="standing")
            )
            if task is None or task.status != "active":
                return None
            task.amendments.append(text.strip())
            public = task.public()
        self.bus.publish("task_updated", task=public)
        return public

    def cancel_task(self, task_id: str | None = None, text: str = "") -> dict | None:
        with self.condition:
            task = self.tasks.get(task_id) if task_id else self._task_for_message(text, prefer=None)
            if task is None or task.status != "active":
                return None
            task.status = "cancelled"
            public = task.public()
        self.bus.publish("task_cancelled", task=public)
        return public

    def task(self, task_id: str) -> LiveTask:
        return self.tasks[task_id]

    # -- instant jobs ------------------------------------------------------

    def _submit_job(self, kind: str, run: Callable[[], Any], *, timeout: float | None) -> Any:
        self._raise_if_failed()
        with self.condition:
            job = _Job(kind=kind, run=run, required_windows=self.submitted_windows)
            self.jobs.append(job)
            self.condition.notify_all()
        if not job.done.wait(timeout=timeout):
            raise TimeoutError(f"{kind} job did not finish within {timeout} s")
        if job.error is not None:
            raise RuntimeError(f"{kind} job failed: {job.error!r}") from job.error
        return job.result

    def ask(
        self,
        question: str,
        *,
        choices: Sequence[str] | None = None,
        max_new_tokens: int | None = None,
        timeout: float | None = None,
        record: bool = True,
    ) -> dict:
        """Answer a one-off question against all video submitted so far."""

        if not question.strip():
            raise ValueError("question must not be empty")
        asked_at = time.perf_counter()

        def run() -> dict:
            now = self.video_seconds
            use_context = not self.config.commit_outputs
            memory = []
            if self.config.use_memory and use_context:
                before = None
                if self.config.memory_scope == "evicted":
                    trunk = self._trunk_stats()
                    evicted = trunk.get("evicted_tokens", 0) > 0
                    before = trunk.get("oldest_resident_seconds") if evicted else float("-inf")
                memory = self.memory.retrieve(
                    question, limit=self.config.memory_entries, before=before
                )
            dialog = self.dialog[-self.config.dialog_turns :] if use_context else []
            prompt = instant_prompt(
                question, now, memory=memory, dialog=dialog, plain=not use_context
            )
            started = time.perf_counter()
            if choices:
                letters = list(CHOICE_LETTERS[: len(choices)])
                scores, stats = self.backend.score_choices(prompt, letters)
                answer = mc_letter(scores)
                stats = {**stats, "scores": scores}
            else:
                result = self.backend.generate_branch(
                    prompt,
                    max_new_tokens=max_new_tokens or self.config.instant_max_new_tokens,
                    commit=self.config.commit_outputs,
                )
                answer, stats = result.text, result.stats
            finished = time.perf_counter()
            return {
                "question": question,
                "answer": answer,
                "video_seconds": now,
                "memory_used": len(memory),
                "wait_seconds": started - asked_at,
                "latency_seconds": finished - asked_at,
                "stats": stats,
            }

        result = self._submit_job("instant", run, timeout=timeout)
        if record and not choices:
            self.dialog.append({"question": question, "answer": result["answer"]})
        self.bus.publish("answer", **{k: v for k, v in result.items() if k != "stats"})
        return result

    def route(self, text: str, *, timeout: float | None = None) -> dict:
        """Classify a user message into one of ``INTENTS``: rules first, then the model."""

        with self.condition:
            has_task = self._latest_user_task() is not None
        ruled = rule_route(text, has_task=has_task)
        if ruled is not None:
            return {"intent": ruled, "source": "rule"}

        def run() -> dict:
            letters = list(ROUTER_LABELS)
            scores, stats = self.backend.score_choices(
                ROUTER_PROMPT.format(text=text.replace('"', "'")), letters
            )
            return {
                "intent": ROUTER_LABELS[mc_letter(scores)],
                "source": "model",
                "scores": scores,
                **stats,
            }

        routed = self._submit_job("route", run, timeout=timeout)
        with self.condition:
            has_task = self._latest_user_task() is not None
        if routed["intent"] in ("update", "cancel") and not has_task:
            routed["intent"] = "instant"
        return routed

    def handle_message(
        self,
        text: str,
        video_seconds: float,
        *,
        kind: str = "auto",
        timeout: float | None = None,
    ) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("message must not be empty")
        routed = None
        if kind == "auto":
            routed = self.route(text, timeout=timeout)
            kind = routed["intent"]
        if kind not in INTENTS:
            raise ValueError(f"message kind must be 'auto' or one of {INTENTS}")
        self.bus.publish("user_message", text=text, kind=kind, video_seconds=video_seconds)
        response: dict = {"kind": kind, "routing": routed}
        if kind == "instant":
            response["answer"] = self.ask(text, timeout=timeout)
        elif kind in TASK_KINDS:
            response["task"] = self.add_task(kind, text, video_seconds)
        elif kind == "update":
            response["task"] = self.update_task(text)
        else:
            response["task"] = self.cancel_task(text=text)
        return response

    # -- GPU worker --------------------------------------------------------

    def _ready_job(self) -> _Job | None:
        barrier = (
            self.processed_windows
            if self.config.tick_policy == "lockstep"
            else self.prefilled_windows
        )
        for job in self.jobs:
            if job.required_windows <= barrier:
                return job
        return None

    def _instant_waiting(self) -> bool:
        # Any queued job preempts, even one still waiting for windows: those
        # windows cannot be prefilled until the running branch stops.
        with self.condition:
            return bool(self.jobs)

    def _caught_up(self) -> bool:
        return (
            not self.encoded
            and self.prefilled_windows == self.submitted_windows
            and self.input_queue.empty()
        )

    def _due_tasks(self) -> list[LiveTask]:
        due = [
            task
            for task in self.tasks.values()
            if task.status == "active" and task.next_due <= self.video_seconds + 1e-6
        ]
        # User-visible tasks first, then the most overdue.
        return sorted(due, key=lambda task: (task.hidden, task.next_due))

    def _gpu_worker(self) -> None:
        try:
            while True:
                with self.condition:
                    while True:
                        if self.stopping:
                            return
                        job = self._ready_job()
                        if job is not None:
                            self.jobs.remove(job)
                            action = ("job", job)
                            break
                        if self.encoded:
                            action = ("window", self.encoded.popleft())
                            self.condition.notify_all()
                            break
                        if self.config.tick_policy == "realtime" and self._caught_up():
                            due = self._due_tasks()
                            if due:
                                action = ("task", due[0])
                                break
                        self.condition.wait(timeout=0.05)
                started = time.perf_counter()
                kind, item = action
                if kind == "job":
                    self._run_job(item)
                elif kind == "window":
                    self._prefill(item)
                    if self.config.tick_policy == "lockstep":
                        with self.condition:
                            due = self._due_tasks()
                        for task in due:
                            self._run_task(task, preemptible=False)
                    with self.condition:
                        self.processed_windows += 1
                        self.condition.notify_all()
                else:
                    self._run_task(item, preemptible=True)
                self.gpu_busy_seconds += time.perf_counter() - started
                # Status readers must not inspect the cache while a forward pass
                # is half-way through its layers, so publish a snapshot instead.
                self.trunk_snapshot = self._trunk_stats()
        except BaseException as exc:
            self._record_error("gpu", exc)
            with self.condition:
                for job in self.jobs:
                    job.error = exc
                    job.done.set()
                self.jobs.clear()

    def _run_job(self, job: _Job) -> None:
        try:
            job.result = job.run()
        except Exception as exc:  # reported to the caller, worker keeps running
            job.error = exc
        finally:
            job.done.set()

    def _prefill(self, window: EncodedWindow) -> None:
        prefill_started = time.perf_counter()
        stats = self.backend.prefill_window(window)
        finished = time.perf_counter()
        timestamps = window.stats.get("timestamps") or []
        with self.condition:
            if timestamps:
                self.video_seconds = float(timestamps[-1])
            self.last_window_submitted_at = window.submitted_at
            self.prefilled_windows += 1
            self.window_log.append(
                {
                    "window_index": window.index,
                    "video_seconds": self.video_seconds,
                    "encode_seconds": window.encode_finished_at - window.encode_started_at,
                    "prefill_seconds": finished - prefill_started,
                    "arrival_to_prefilled_seconds": finished - window.submitted_at,
                    "retained_tokens": stats.get("appended_tokens"),
                    "evicted_tokens": stats.get("evicted_tokens", 0),
                    "cache_length": stats.get("cache_length_final"),
                }
            )
            self.condition.notify_all()

    def _trunk_stats(self) -> dict:
        stats = getattr(self.backend, "trunk_stats", None)
        return {} if stats is None else stats()

    def _content_lag(self) -> float | None:
        if self.last_window_submitted_at is None:
            return None
        return time.perf_counter() - self.last_window_submitted_at

    def _run_task(self, task: LiveTask, *, preemptible: bool) -> None:
        now = self.video_seconds
        if task.status != "active":
            return
        overdue = now - task.next_due
        if overdue >= task.interval_s:
            task.dropped += int(math.floor(overdue / task.interval_s))
        should_abort = self._instant_waiting if preemptible else None
        if task.kind == "standing":
            self._run_standing(task, now, should_abort)
        else:
            self._run_watch(task, now, should_abort)

    def _emit_output(self, task: LiveTask, now: float, text: str, **extra: Any) -> None:
        output = {
            "task_id": task.id,
            "seconds": now,
            "text": text,
            "lag_seconds": self._content_lag(),
            "wall_time": time.time(),
            **extra,
        }
        with self.condition:
            task.outputs.append(output)
        self.memory.add(now, task.id, text)
        self.bus.publish(
            "memory" if task.hidden else ("alert" if task.kind == "watch" else "task_output"),
            task_kind=task.kind,
            **output,
        )

    def _run_standing(self, task: LiveTask, now: float, should_abort) -> None:
        prompt = standing_prompt(
            task,
            now,
            self.config.prompt_history_lines,
            time_markers=bool(
                getattr(getattr(self.backend, "config", None), "time_markers", False)
            ),
            style=self.config.standing_style,
        )
        recent_text = " ".join(item["text"] for item in task.outputs[-self.config.history_lines :])
        result = self.backend.generate_branch(
            prompt,
            max_new_tokens=self.config.standing_max_new_tokens,
            should_abort=should_abort,
            repetition_penalty=self.config.repetition_penalty,
            penalty_text=recent_text,
            silent_marker=SILENT,
            commit=self.config.commit_outputs,
        )
        if result.aborted:
            with self.condition:
                task.preempted += 1
            self.bus.publish("task_preempted", task_id=task.id, video_seconds=now)
            return
        with self.condition:
            task.runs += 1
            task.last_covered = now
            task.next_due = now + task.interval_s
        text = result.text.replace(SILENT, "").strip()
        # The model sometimes translates the marker (e.g. "<沉默>"); any bare
        # bracketed token is silence too.
        if _BRACKETED.fullmatch(text):
            text = ""
        recent = task.outputs[-self.config.history_lines :]
        duplicate = any(
            similarity(text, item["text"]) >= self.config.duplicate_similarity for item in recent
        )
        if result.stats["silent"] or not text or duplicate:
            with self.condition:
                task.silent += 1
                task.duplicates += int(duplicate)
            self.bus.publish(
                "task_silent", task_id=task.id, video_seconds=now, duplicate=duplicate, text=text
            )
            return
        self._emit_output(task, now, text, branch_seconds=result.stats["branch_seconds"])

    def _run_watch(self, task: LiveTask, now: float, should_abort) -> None:
        scores, stats = self.backend.score_choices(watch_check_prompt(task, now), ["Yes", "No"])
        yes = 1.0 / (1.0 + math.exp(scores[1] - scores[0]))
        with self.condition:
            task.checks += 1
            task.check_trace.append((now, yes))
            task.runs += 1
            task.last_covered = now
            task.next_due = now + task.interval_s
        cooling = (
            task.last_fired is not None and now - task.last_fired < self.config.watch_cooldown_s
        )
        if yes < self.config.watch_threshold or cooling:
            return
        result = self.backend.generate_branch(
            watch_output_prompt(task, now),
            max_new_tokens=self.config.alert_max_new_tokens,
            should_abort=should_abort,
            commit=self.config.commit_outputs,
        )
        if result.aborted:
            # The condition held; re-check right after the instant job.
            with self.condition:
                task.preempted += 1
                task.next_due = now
            return
        with self.condition:
            task.last_fired = now
            if task.once:
                task.status = "completed"
        self._emit_output(task, now, result.text, p_yes=yes, check_seconds=stats["branch_seconds"])

    # -- synchronisation and reporting ---------------------------------------

    def wait_processed(self, count: int, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while self.processed_windows < count and not self.errors:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"timed out waiting for {count} processed windows")
                self.condition.wait(
                    timeout=_POLL_SECONDS if remaining is None else min(remaining, _POLL_SECONDS)
                )
        self._raise_if_failed()

    def status(self) -> dict:
        with self.condition:
            tasks = [task.public() for task in self.tasks.values() if not task.hidden]
            return {
                "submitted_windows": self.submitted_windows,
                "prefilled_windows": self.prefilled_windows,
                "processed_windows": self.processed_windows,
                "pending_windows": self.submitted_windows - self.prefilled_windows,
                "video_seconds": self.video_seconds,
                "content_lag_seconds": self._content_lag(),
                "queued_jobs": len(self.jobs),
                "tasks": tasks,
                "memory_entries": len(self.memory),
                "trunk": dict(self.trunk_snapshot),
                "gpu_busy_seconds": self.gpu_busy_seconds,
                "errors": [{"stage": e["stage"], "error": e["error"]} for e in self.errors],
            }


__all__ = [
    "EventBus",
    "INTENTS",
    "LiveConfig",
    "LiveScheduler",
    "LiveTask",
    "MemoryEntry",
    "MemoryLog",
    "format_clock",
    "instant_prompt",
    "lexical_keys",
    "rule_route",
    "similarity",
    "standing_prompt",
]
