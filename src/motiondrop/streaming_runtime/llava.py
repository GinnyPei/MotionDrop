"""LLaVA-OneVision backend for pipelined MotionDrop streaming."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch

from motiondrop.adapters.llava_vision import encode_video_features_with_attention
from motiondrop.compression import select_motiondrop_tokens
from motiondrop.paths import models_dir
from motiondrop.runtime import module_device
from motiondrop.streaming_runtime.config import RuntimeConfig
from motiondrop.streaming_runtime.kv import KVHistory, cache_length
from motiondrop.streaming_runtime.latency import LatencyModel, fixed_budget
from motiondrop.streaming_runtime.pipeline import EncodedWindow, FrameWindow, time_marker

SYSTEM_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
BRANCH_OPEN = "\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
# Legacy shared-history mode closes the assistant turn and re-opens a user turn
# so later video windows continue in the same chronological KV history.
TURN_BOUNDARY = "<|im_end|>\n<|im_start|>user\n"


@dataclass(frozen=True)
class TrunkSegment:
    """One video window resident in the trunk KV cache."""

    window_index: int
    start: int
    length: int
    start_seconds: float | None
    end_seconds: float | None


@dataclass(frozen=True)
class BranchResult:
    text: str
    aborted: bool
    stats: dict


class LlavaStreamingBackend:
    """Stateful visual encoder and incremental-prefill backend.

    The trunk holds the system prefix followed by video windows only. Queries
    and task updates run as branches that are rolled back afterwards unless
    ``commit=True`` (the legacy shared-history behaviour).
    """

    def __init__(
        self,
        model,
        processor,
        *,
        config: RuntimeConfig,
        latency_model: LatencyModel | None = None,
        slo_ms: float | None = None,
    ) -> None:
        if (latency_model is None) != (slo_ms is None):
            raise ValueError("latency_model and slo_ms must be provided together")
        self.model = model
        self.processor = processor
        self.config = config
        self.latency_model = latency_model
        self.slo_ms = slo_ms
        self.language_model = model.model.language_model
        self.embedding = model.model.get_input_embeddings()
        self.embedding_device = module_device(self.embedding)
        self.head_device = module_device(model.lm_head)
        self.history = KVHistory(
            self.language_model,
            device=self.embedding_device,
            position_mode=config.position_mode,
        )
        self.prefix_tokens = 0
        self.total_visual_tokens = 0
        self.window_count = 0
        self.segments: list[TrunkSegment] = []
        self.committed_tokens = 0
        self.reference_features: torch.Tensor | None = None
        eos = processor.tokenizer.eos_token_id
        if eos is None:
            self.eos_ids: set[int] = set()
        elif isinstance(eos, list):
            self.eos_ids = {int(value) for value in eos}
        else:
            self.eos_ids = {int(eos)}
        end_turn = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(end_turn, int) and end_turn >= 0:
            self.eos_ids.add(end_turn)

    @property
    def cache(self):
        return self.history.cache

    def _tokenize(self, text: str) -> torch.Tensor:
        values = self.processor.tokenizer(text, add_special_tokens=False).input_ids
        if not values:
            raise ValueError("tokenizer returned an empty token sequence")
        return torch.tensor([values], dtype=torch.long, device=self.embedding_device)

    def _logits(self, output) -> torch.Tensor:
        hidden = output.last_hidden_state[:, -1]
        return self.model.lm_head(hidden.to(self.head_device))[0].float()

    @torch.inference_mode()
    def initialize_history(self) -> None:
        if self.history.cache is not None:
            raise RuntimeError("streaming history is already initialized")
        self.history.forward(input_ids=self._tokenize(SYSTEM_PREFIX))
        self.prefix_tokens = self.history.length

    def _budget(self, original_tokens: int, frame_count: int):
        if self.latency_model is None:
            return fixed_budget(
                original_tokens=original_tokens,
                temporal_units=frame_count,
                compression=self.config.fixed_compression,
            )
        return self.latency_model.largest_feasible_budget(
            visual_tokens=original_tokens,
            slo_ms=float(self.slo_ms),
            minimum=frame_count,
            maximum=original_tokens,
        )

    @torch.inference_mode()
    def encode_window(self, window: FrameWindow) -> EncodedWindow:
        started = time.perf_counter()
        processed = self.processor.video_processor(
            videos=list(window.frames),
            do_sample_frames=False,
            return_tensors="pt",
        )
        pixels = processed["pixel_values_videos"]
        outputs, spatial_attention = encode_video_features_with_attention(
            self.model,
            pixels,
            capture_attention=True,
        )
        features = outputs.pooler_output
        if features.ndim != 3 or int(features.shape[0]) != 1:
            raise ValueError(f"unexpected LLaVA video feature shape: {tuple(features.shape)}")
        frame_count = len(window.frames)
        original_tokens = int(features.shape[1])
        if original_tokens % frame_count:
            raise ValueError(
                f"cannot divide {original_tokens} visual tokens across {frame_count} frames"
            )
        tokens_per_frame = original_tokens // frame_count
        temporal = features[0].reshape(frame_count, tokens_per_frame, -1)
        scores = None if spatial_attention is None else spatial_attention[0].to(temporal.device)
        decision = self._budget(original_tokens, frame_count)
        compression = (
            0.0
            if decision.retained_tokens == original_tokens
            else 1.0 - decision.retained_tokens / original_tokens
        )
        keep, stats = select_motiondrop_tokens(
            temporal,
            replace(
                self.config.motiondrop,
                target_compression=compression,
                preserve_token_order=True,
            ),
            spatial_scores=scores,
            reference_features=self.reference_features,
        )
        self.reference_features = temporal[-1].detach()
        selected = features[0].index_select(0, keep.to(features.device))
        structural_tokens = 0
        if self.config.time_markers and window.timestamps is not None:
            marker = self._tokenize(time_marker(window.timestamps[0]))
            marker_embeds = self.embedding(marker)[0].to(selected.device, selected.dtype)
            selected = torch.cat((marker_embeds, selected), dim=0)
            structural_tokens += int(marker.shape[1])
        if self.config.add_window_newline:
            newline = self.model.model.image_newline[None].to(selected.device, selected.dtype)
            selected = torch.cat((selected, newline), dim=0)
            structural_tokens += 1
        selected = selected.unsqueeze(0).to(self.embedding_device)
        finished = time.perf_counter()
        stats.update(
            {
                "schema": "motiondrop_streaming_window_v1",
                "window_index": window.index,
                "frame_indices": list(window.frame_indices),
                "timestamps": None if window.timestamps is None else list(window.timestamps),
                "budget_mode": decision.mode,
                "window_budget": decision.retained_tokens,
                "predicted_latency_ms": decision.predicted_latency_ms,
                "slo_ms": decision.slo_ms,
                "structural_tokens": structural_tokens,
                "prefill_tokens": int(selected.shape[1]),
            }
        )
        return EncodedWindow(
            index=window.index,
            payload=selected,
            stats=stats,
            submitted_at=window.submitted_at,
            encode_started_at=started,
            encode_finished_at=finished,
        )

    @torch.inference_mode()
    def prefill_window(self, window: EncodedWindow) -> dict:
        before = self.history.length
        self.history.forward(inputs_embeds=window.payload)
        after = self.history.length
        appended = int(window.payload.shape[1])
        if after - before != appended:
            raise AssertionError(f"cache append mismatch: {after - before} != {appended}")
        timestamps = window.stats.get("timestamps") or None
        self.segments.append(
            TrunkSegment(
                window_index=window.index,
                start=before,
                length=appended,
                start_seconds=None if not timestamps else float(timestamps[0]),
                end_seconds=None if not timestamps else float(timestamps[-1]),
            )
        )
        self.total_visual_tokens += appended
        self.window_count += 1
        evicted = self.enforce_context_limit()
        return {
            "cache_length_before": before,
            "cache_length_after": after,
            "appended_tokens": appended,
            "evicted_tokens": evicted,
            "cache_length_final": self.history.length,
        }

    def enforce_context_limit(self) -> int:
        """Evict the oldest windows once the trunk exceeds ``max_context_tokens``.

        Eviction is batched down to ``evict_to_ratio`` of the limit so the
        (cheap but non-free) key re-rotation runs rarely. The newest window is
        never evicted and committed text is never split.
        """

        limit = self.config.max_context_tokens
        if limit is None or self.history.length <= limit:
            return 0
        target = int(limit * self.config.evict_to_ratio)
        drop = 0
        while drop < len(self.segments) - 1 and self.history.length - self._span(drop) > target:
            drop += 1
        if drop == 0:
            return 0
        start = self.segments[0].start
        end = self.segments[drop - 1].start + self.segments[drop - 1].length
        removed = self.history.evict(start, end)
        if removed != end - start:
            raise AssertionError("eviction removed an unexpected number of slots")
        survivors = []
        for segment in self.segments[drop:]:
            shift = removed if segment.start >= end else 0
            survivors.append(
                TrunkSegment(
                    segment.window_index,
                    segment.start - shift,
                    segment.length,
                    segment.start_seconds,
                    segment.end_seconds,
                )
            )
        self.segments = survivors
        return removed

    def _span(self, count: int) -> int:
        if count == 0:
            return 0
        last = self.segments[count - 1]
        return last.start + last.length - self.segments[0].start

    def trunk_stats(self) -> dict:
        oldest = self.segments[0] if self.segments else None
        return {
            **self.history.stats(),
            "prefix_tokens": self.prefix_tokens,
            "resident_windows": len(self.segments),
            "resident_visual_tokens": sum(segment.length for segment in self.segments),
            "oldest_resident_seconds": None if oldest is None else oldest.start_seconds,
            "committed_tokens": self.committed_tokens,
            "max_context_tokens": self.config.max_context_tokens,
        }

    def _penalize(self, logits: torch.Tensor, token_ids: set[int], penalty: float) -> None:
        if penalty == 1.0 or not token_ids:
            return
        index = torch.tensor(sorted(token_ids), dtype=torch.long, device=logits.device)
        values = logits.index_select(0, index)
        logits.index_copy_(0, index, torch.where(values > 0, values / penalty, values * penalty))

    @torch.inference_mode()
    def generate_branch(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        should_abort: Callable[[], bool] | None = None,
        repetition_penalty: float = 1.0,
        penalty_text: str = "",
        silent_marker: str | None = None,
        commit: bool = False,
    ) -> BranchResult:
        """Generate an assistant turn after the current trunk, then roll it back.

        ``should_abort`` is polled between tokens so a higher-priority job can
        preempt the branch; aborted text is returned but should be discarded.
        """

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        checkpoint = self.history.checkpoint()
        started = time.perf_counter()
        prompt_ids = self._tokenize(BRANCH_OPEN.format(prompt=prompt))
        generated: list[int] = []
        aborted = False
        silent = False
        first_token_at = None
        try:
            logits = self._logits(self.history.forward(input_ids=prompt_ids))
            first_token_at = time.perf_counter()
            penalty_ids = set()
            if penalty_text:
                penalty_ids.update(
                    self.processor.tokenizer(penalty_text, add_special_tokens=False).input_ids
                )
            for _ in range(max_new_tokens):
                if should_abort is not None and should_abort():
                    aborted = True
                    break
                self._penalize(logits, penalty_ids | set(generated), repetition_penalty)
                token = int(logits.argmax().item())
                if token in self.eos_ids:
                    break
                generated.append(token)
                if silent_marker and len(generated) <= 8:
                    head = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
                    if head.strip().startswith(silent_marker):
                        silent = True
                        break
                token_tensor = torch.tensor(
                    [[token]], dtype=torch.long, device=self.embedding_device
                )
                logits = self._logits(self.history.forward(input_ids=token_tensor))
            if commit and not aborted:
                self.history.forward(input_ids=self._tokenize(TURN_BOUNDARY))
                self.committed_tokens += self.history.length - checkpoint
        finally:
            if not commit or aborted:
                self.history.rollback(checkpoint)
        finished = time.perf_counter()
        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return BranchResult(
            text=text,
            aborted=aborted,
            stats={
                "prompt_tokens": int(prompt_ids.shape[1]),
                "generated_tokens": len(generated),
                "silent": silent,
                "committed": commit and not aborted,
                "time_to_first_token_seconds": (
                    None if first_token_at is None else first_token_at - started
                ),
                "branch_seconds": finished - started,
                "cache_length_after": self.history.length,
            },
        )

    @torch.inference_mode()
    def score_choices(
        self,
        prompt: str,
        choices: Sequence[str],
        *,
        assistant_prefix: str = "",
    ) -> tuple[list[float], dict]:
        """Log-probabilities of each choice's first token as the assistant reply."""

        if not choices:
            raise ValueError("choices must not be empty")
        checkpoint = self.history.checkpoint()
        started = time.perf_counter()
        prompt_ids = self._tokenize(BRANCH_OPEN.format(prompt=prompt) + assistant_prefix)
        try:
            logits = self._logits(self.history.forward(input_ids=prompt_ids))
        finally:
            self.history.rollback(checkpoint)
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = []
        for choice in choices:
            ids = self.processor.tokenizer(choice, add_special_tokens=False).input_ids
            if not ids:
                raise ValueError(f"choice {choice!r} tokenizes to nothing")
            scores.append(float(log_probs[ids[0]].item()))
        return scores, {
            "prompt_tokens": int(prompt_ids.shape[1]),
            "branch_seconds": time.perf_counter() - started,
        }

    def answer(self, query: str, *, max_new_tokens: int) -> tuple[str, dict]:
        started = time.perf_counter()
        result = self.generate_branch(
            query,
            max_new_tokens=max_new_tokens,
            commit=self.config.commit_answers,
        )
        return result.text, {
            "query_tokens": result.stats["prompt_tokens"],
            "generated_tokens": result.stats["generated_tokens"],
            "time_to_first_token_seconds": result.stats["time_to_first_token_seconds"],
            "query_total_seconds": time.perf_counter() - started,
            "cache_length_after_query": self.history.length,
            "history_prefix_tokens": self.prefix_tokens,
            "history_visual_tokens": self.total_visual_tokens,
            "history_windows": self.window_count,
        }


def resolve_local_checkpoint(model_path: str | None) -> str | None:
    if model_path:
        return model_path
    from motiondrop.downloads import local_model_path

    local = local_model_path(models_dir(), "llava-ov-7b")
    return None if local is None else str(local)


__all__ = [
    "BranchResult",
    "LlavaStreamingBackend",
    "TrunkSegment",
    "cache_length",
    "resolve_local_checkpoint",
]
