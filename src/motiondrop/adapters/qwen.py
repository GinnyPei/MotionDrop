"""Post-vision MotionDrop adapters for Qwen3-VL and Qwen3.5."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, replace

import torch

from motiondrop.compression import MotionDropConfig, select_motiondrop_tokens
from motiondrop.runtime import (
    label_token_sequences,
    module_device,
    score_candidate_sequences,
)


class _EnableThinkingProcessorWarningFilter(logging.Filter):
    """Hide only ProcessorMixin's duplicate routing warning for a Jinja flag."""

    def filter(self, record: logging.LogRecord) -> bool:
        return (
            "Keyword argument `enable_thinking` is not a valid argument for this processor"
            not in record.getMessage()
        )


class _QwenMotionDropBatchAdapter:
    """Shared native-video, true-batch posterior inference implementation."""

    family: str
    padding_side: str
    has_deepstack: bool
    thinking_mode: bool | None

    def __init__(
        self,
        model,
        processor,
        answer_labels: Sequence[str],
        *,
        motiondrop: MotionDropConfig | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.answer_labels = tuple(answer_labels)
        self.answer_token_sequences = label_token_sequences(processor.tokenizer, answer_labels)
        self.motiondrop = motiondrop or MotionDropConfig()
        self.processor.tokenizer.padding_side = self.padding_side

    def _processor_batch(self, conversations: Sequence[list[dict]]) -> dict[str, torch.Tensor]:
        from transformers.video_utils import VideoMetadata

        metadata = []
        for conversation in conversations:
            videos = [
                item["video"]
                for message in conversation
                for item in message["content"]
                if item.get("type") == "video"
            ]
            if len(videos) != 1:
                raise ValueError("Each benchmark conversation must contain exactly one video")
            frames = videos[0]
            metadata.append(
                VideoMetadata(
                    total_num_frames=len(frames),
                    fps=1.0,
                    width=frames[0].width,
                    height=frames[0].height,
                    duration=float(len(frames)),
                    frames_indices=list(range(len(frames))),
                )
            )

        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "do_sample_frames": False,
            "video_metadata": metadata,
            "padding": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.thinking_mode is False:
            kwargs["enable_thinking"] = False
            processor_logger = logging.getLogger("transformers.processing_utils")
            warning_filter = _EnableThinkingProcessorWarningFilter()
            processor_logger.addFilter(warning_filter)
            try:
                result = self.processor.apply_chat_template(list(conversations), **kwargs)
            finally:
                processor_logger.removeFilter(warning_filter)
            self._validate_nonthinking_suffix(result)
        else:
            result = self.processor.apply_chat_template(list(conversations), **kwargs)

        required = {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw"}
        missing = sorted(required - set(result))
        if missing:
            raise ValueError(f"{self.family} processor omitted required fields: {missing}")
        return result

    def _validate_nonthinking_suffix(self, processor_result: dict[str, torch.Tensor]) -> None:
        nonthinking_suffix = self.processor.tokenizer.encode(
            "<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        if not nonthinking_suffix:
            raise ValueError("Qwen3.5 tokenizer produced an empty non-thinking suffix")
        for sample_index, (token_ids, attention) in enumerate(
            zip(
                processor_result["input_ids"],
                processor_result["attention_mask"],
                strict=True,
            )
        ):
            active_ids = token_ids[attention.bool()].tolist()
            if active_ids[-len(nonthinking_suffix) :] != nonthinking_suffix:
                raise ValueError(
                    "Qwen3.5 chat template did not render the required non-thinking "
                    f"assistant prefix for sample {sample_index}"
                )

    @torch.inference_mode()
    def infer(
        self,
        conversations: Sequence[list[dict]],
        *,
        target_compression: float,
        sampled_frame_counts: Sequence[int],
        candidate_labels: Sequence[Sequence[str]] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Return ``[batch, labels]`` posterior logits and compression metadata."""

        if len(conversations) != len(sampled_frame_counts):
            raise ValueError("conversations and sampled_frame_counts must have equal length")
        if not conversations:
            raise ValueError("Cannot infer an empty batch")

        config = replace(
            self.motiondrop,
            target_compression=target_compression,
            preserve_token_order=True,
        )
        processor_inputs = self._processor_batch(conversations)
        batch_size = len(conversations)
        input_ids_cpu = processor_inputs["input_ids"]
        attention_cpu = processor_inputs["attention_mask"]
        grids_cpu = processor_inputs["video_grid_thw"]
        if input_ids_cpu.shape[0] != batch_size or grids_cpu.shape[0] != batch_size:
            raise ValueError(
                f"Expected one native video per sample, batch={batch_size}, "
                f"input_ids={tuple(input_ids_cpu.shape)}, grids={tuple(grids_cpu.shape)}"
            )

        embedding = self.model.model.get_input_embeddings()
        embedding_device = module_device(embedding)
        visual_device = module_device(self.model.model.visual)
        input_ids = input_ids_cpu.to(embedding_device)
        attention_mask = attention_cpu.to(embedding_device)
        grids = grids_cpu.to(embedding_device)
        inputs_embeds = embedding(input_ids)

        video_outputs = self.model.model.get_video_features(
            pixel_values_videos=processor_inputs["pixel_values_videos"].to(visual_device),
            video_grid_thw=grids_cpu.to(visual_device),
            return_dict=True,
        )
        video_features = tuple(video_outputs.pooler_output)
        merge_size = int(self.model.config.vision_config.spatial_merge_size)
        split_sizes = [
            int(torch.prod(grid).item()) // (merge_size * merge_size) for grid in grids_cpu
        ]
        if (
            len(video_features) != batch_size
            or [len(value) for value in video_features] != split_sizes
        ):
            raise ValueError(
                f"Vision split mismatch: feature_sizes={[len(v) for v in video_features]}, "
                f"expected={split_sizes}"
            )

        position_ids, _ = self.model.model.get_rope_index(
            input_ids=input_ids,
            video_grid_thw=grids,
            attention_mask=attention_mask,
        )
        video_token_id = int(self.model.config.video_token_id)

        deepstack_features = getattr(video_outputs, "deepstack_features", None)
        ds_splits: list[tuple[torch.Tensor, ...]] = []
        if self.has_deepstack:
            if deepstack_features is None:
                raise ValueError("Qwen3-VL video encoder did not return DeepStack features")
            for layer_features in deepstack_features:
                ds_splits.append(tuple(torch.split(layer_features, split_sizes, dim=0)))

        compressed_embeds: list[torch.Tensor] = []
        compressed_positions: list[torch.Tensor] = []
        compressed_visual_masks: list[torch.Tensor] = []
        selected_ds: list[list[torch.Tensor]] = [[] for _ in ds_splits]
        stats: list[dict] = []

        for sample_index, (features_flat, grid) in enumerate(
            zip(video_features, grids_cpu, strict=True)
        ):
            temporal_units = int(grid[0].item())
            spatial_tokens = split_sizes[sample_index] // temporal_units
            features = features_flat.reshape(temporal_units, spatial_tokens, -1)
            spatial_height = int(grid[1].item()) // merge_size
            spatial_width = int(grid[2].item()) // merge_size
            keep_visual, sample_stats = select_motiondrop_tokens(
                features,
                config,
                spatial_shape=(spatial_height, spatial_width),
            )
            sample_stats.update(
                {
                    "schema": f"{self.family}_motiondrop_compression_v1",
                    "model_family": self.family,
                    "sampled_input_frames": int(sampled_frame_counts[sample_index]),
                    "video_grid_thw": [int(value) for value in grid.tolist()],
                    "spatial_merge_size": merge_size,
                    "thinking_mode": self.thinking_mode,
                    "batch_padding_side": self.padding_side,
                    "synchronized_tensors": [
                        "projected_visual_embeddings",
                        *(["deepstack_visual_embeddings"] if self.has_deepstack else []),
                        "attention_mask",
                        "mrope_position_ids",
                    ],
                    "config": asdict(config),
                }
            )

            active = attention_mask[sample_index].bool()
            visual_positions = torch.nonzero(
                (input_ids[sample_index] == video_token_id) & active,
                as_tuple=False,
            ).flatten()
            if int(visual_positions.numel()) != split_sizes[sample_index]:
                raise ValueError(
                    f"Video placeholder mismatch in sample {sample_index}: "
                    f"tokens={int(visual_positions.numel())}, features={split_sizes[sample_index]}"
                )

            features_on_embed = features_flat.to(embedding_device, inputs_embeds.dtype)
            inputs_embeds[sample_index, visual_positions] = features_on_embed
            selected_mask = torch.zeros(
                split_sizes[sample_index], dtype=torch.bool, device=embedding_device
            )
            selected_mask[keep_visual.to(embedding_device)] = True
            sequence_keep = active.clone()
            sequence_keep[visual_positions[~selected_mask]] = False
            sequence_indices = torch.nonzero(sequence_keep, as_tuple=False).flatten()

            compressed_embeds.append(inputs_embeds[sample_index, sequence_indices])
            compressed_positions.append(position_ids[:, sample_index, sequence_indices])
            compressed_visual_masks.append(
                input_ids[sample_index, sequence_indices] == video_token_id
            )
            for ds_index, layer_splits in enumerate(ds_splits):
                selected_ds[ds_index].append(
                    layer_splits[sample_index].index_select(
                        0, keep_visual.to(layer_splits[sample_index].device)
                    )
                )
            stats.append(sample_stats)

        lengths = [int(value.shape[0]) for value in compressed_embeds]
        max_length = max(lengths)
        hidden_size = int(inputs_embeds.shape[-1])
        packed_embeds = inputs_embeds.new_zeros(batch_size, max_length, hidden_size)
        packed_attention = attention_mask.new_zeros(batch_size, max_length)
        packed_positions = position_ids.new_zeros(3, batch_size, max_length)
        packed_visual_masks = torch.zeros(
            batch_size, max_length, dtype=torch.bool, device=embedding_device
        )
        for sample_index, length in enumerate(lengths):
            start = max_length - length if self.padding_side == "left" else 0
            stop = start + length
            packed_embeds[sample_index, start:stop] = compressed_embeds[sample_index]
            packed_attention[sample_index, start:stop] = 1
            packed_positions[:, sample_index, start:stop] = compressed_positions[sample_index]
            packed_visual_masks[sample_index, start:stop] = compressed_visual_masks[sample_index]

        packed_deepstack = [
            torch.cat(layer_values, dim=0).to(embedding_device, inputs_embeds.dtype)
            for layer_values in selected_ds
        ]
        language_kwargs = {
            "input_ids": None,
            "inputs_embeds": packed_embeds,
            "attention_mask": packed_attention,
            "position_ids": packed_positions,
            "use_cache": False,
            "return_dict": True,
        }
        if self.has_deepstack:
            language_kwargs.update(
                {
                    "visual_pos_masks": packed_visual_masks,
                    "deepstack_visual_embeds": packed_deepstack,
                }
            )
        outputs = self.model.model.language_model(**language_kwargs)
        if self.padding_side == "left":
            last_hidden = outputs.last_hidden_state[:, -1]
        else:
            output_device = outputs.last_hidden_state.device
            rows = torch.arange(batch_size, device=output_device)
            columns = torch.tensor(lengths, device=output_device) - 1
            last_hidden = outputs.last_hidden_state[rows, columns]

        def continuation_hidden(prefix_token_id: int) -> torch.Tensor:
            extended_length = max_length + 1
            extended_embeds = inputs_embeds.new_zeros(batch_size, extended_length, hidden_size)
            extended_attention = attention_mask.new_zeros(batch_size, extended_length)
            extended_positions = position_ids.new_zeros(3, batch_size, extended_length)
            extended_visual_masks = torch.zeros(
                batch_size,
                extended_length,
                dtype=torch.bool,
                device=embedding_device,
            )
            prefix_ids = torch.full(
                (batch_size,),
                prefix_token_id,
                dtype=torch.long,
                device=embedding_device,
            )
            prefix_embeds = embedding(prefix_ids)
            prefix_columns: list[int] = []
            for sample_index, length in enumerate(lengths):
                start = max_length - length if self.padding_side == "left" else 0
                stop = start + length
                extended_embeds[sample_index, start:stop] = compressed_embeds[sample_index]
                extended_embeds[sample_index, stop] = prefix_embeds[sample_index]
                extended_attention[sample_index, start : stop + 1] = 1
                extended_positions[:, sample_index, start:stop] = compressed_positions[sample_index]
                extended_positions[:, sample_index, stop] = (
                    compressed_positions[sample_index][:, -1] + 1
                )
                extended_visual_masks[sample_index, start:stop] = compressed_visual_masks[
                    sample_index
                ]
                prefix_columns.append(stop)

            continuation_kwargs = {
                "input_ids": None,
                "inputs_embeds": extended_embeds,
                "attention_mask": extended_attention,
                "position_ids": extended_positions,
                "use_cache": False,
                "return_dict": True,
            }
            if self.has_deepstack:
                continuation_kwargs.update(
                    {
                        "visual_pos_masks": extended_visual_masks,
                        "deepstack_visual_embeds": packed_deepstack,
                    }
                )
            continuation_outputs = self.model.model.language_model(**continuation_kwargs)
            output_device = continuation_outputs.last_hidden_state.device
            rows = torch.arange(batch_size, device=output_device)
            columns = torch.tensor(prefix_columns, device=output_device)
            return continuation_outputs.last_hidden_state[rows, columns]

        scores = score_candidate_sequences(
            last_hidden,
            self.model.lm_head,
            self.answer_labels,
            self.answer_token_sequences,
            candidate_labels=candidate_labels,
            continuation_hidden=continuation_hidden,
        )
        return scores, stats


class Qwen3VLMotionDropBatchAdapter(_QwenMotionDropBatchAdapter):
    """MotionDrop adapter for ``Qwen/Qwen3-VL-8B-Instruct``."""

    family = "qwen3_vl"
    padding_side = "right"
    has_deepstack = True
    thinking_mode = None


class Qwen35MotionDropBatchAdapter(_QwenMotionDropBatchAdapter):
    """MotionDrop adapter for dense ``Qwen/Qwen3.5-9B``."""

    family = "qwen3_5"
    padding_side = "left"
    has_deepstack = False
    thinking_mode = False


__all__ = ["Qwen3VLMotionDropBatchAdapter", "Qwen35MotionDropBatchAdapter"]
