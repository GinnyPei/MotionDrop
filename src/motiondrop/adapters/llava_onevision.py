"""Transformers-native LLaVA-OneVision MotionDrop adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, replace

import torch

from motiondrop.adapters.llava_vision import encode_video_features_with_attention
from motiondrop.compression import MotionDropConfig, select_motiondrop_tokens
from motiondrop.runtime import (
    label_token_sequences,
    module_device,
    score_candidate_sequences,
)


class LlavaOnevisionMotionDropBatchAdapter:
    """Post-SigLIP, pre-decoder MotionDrop for the converted OV-7B checkpoint."""

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
        self.processor.tokenizer.padding_side = "right"
        # (conversations, processor inputs, video features, spatial attention)
        # of the last batch; see _encoded_batch.
        self._last_batch: tuple | None = None

    def _video_features_with_attention(
        self,
        pixels: torch.Tensor,
        visual_device: torch.device,
        *,
        capture_attention: bool,
    ):
        del visual_device
        # Uncompressed features of a long clip outgrow GPU memory; keep them on
        # the host and move only the selected tokens to the language model.
        return encode_video_features_with_attention(
            self.model,
            pixels,
            capture_attention=capture_attention,
            feature_device="cpu",
        )

    def _processor_batch(self, conversations: Sequence[list[dict]]) -> dict[str, torch.Tensor]:
        prompts: list[str] = []
        videos: list[list] = []
        for conversation in conversations:
            sample_videos = [
                item["video"]
                for message in conversation
                for item in message["content"]
                if item.get("type") == "video"
            ]
            if len(sample_videos) != 1:
                raise ValueError("Each benchmark conversation must contain exactly one video")
            prompts.append(
                self.processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            videos.append(sample_videos[0])

        result = self.processor(
            text=prompts,
            videos=videos,
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        )
        required = {"input_ids", "attention_mask", "pixel_values_videos"}
        missing = sorted(required - set(result))
        if missing:
            raise ValueError(f"LLaVA-OneVision processor omitted required fields: {missing}")
        return result

    def _encoded_batch(self, conversations: Sequence[list[dict]], *, capture_attention: bool):
        """Processor inputs and encoded video, reused when the same batch repeats.

        Benchmarks score one batch at several compression rates; preprocessing
        and the vision encoder do not depend on the rate, so they run once.
        """

        cached = self._last_batch
        if (
            cached is not None
            and cached[0] is conversations
            and (cached[3] is not None or not capture_attention)
        ):
            return cached[1:]
        self._last_batch = None
        processor_inputs = self._processor_batch(conversations)
        batch_size = len(conversations)
        input_ids = processor_inputs["input_ids"]
        pixels = processor_inputs["pixel_values_videos"]
        if input_ids.shape[0] != batch_size or pixels.shape[0] != batch_size:
            raise ValueError(
                f"Expected one video per sample, batch={batch_size}, "
                f"input_ids={tuple(input_ids.shape)}, pixels={tuple(pixels.shape)}"
            )
        video_outputs, spatial_attention = self._video_features_with_attention(
            pixels,
            module_device(self.model.model.vision_tower),
            capture_attention=capture_attention,
        )
        video_features = video_outputs.pooler_output
        if video_features.ndim != 3 or int(video_features.shape[0]) != batch_size:
            raise ValueError(f"Unexpected LLaVA video feature shape: {tuple(video_features.shape)}")
        self._last_batch = (conversations, processor_inputs, video_features, spatial_attention)
        return self._last_batch[1:]

    @torch.inference_mode()
    def infer(
        self,
        conversations: Sequence[list[dict]],
        *,
        target_compression: float,
        sampled_frame_counts: Sequence[int],
        candidate_labels: Sequence[Sequence[str]] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        if len(conversations) != len(sampled_frame_counts):
            raise ValueError("conversations and sampled_frame_counts must have equal length")
        if not conversations:
            raise ValueError("Cannot infer an empty batch")

        config = replace(
            self.motiondrop,
            target_compression=target_compression,
            preserve_token_order=True,
        )
        processor_inputs, video_features, spatial_attention = self._encoded_batch(
            conversations, capture_attention=target_compression > 0.0
        )
        batch_size = len(conversations)

        embedding = self.model.model.get_input_embeddings()
        embedding_device = module_device(embedding)
        input_ids = processor_inputs["input_ids"].to(embedding_device)
        attention_mask = processor_inputs["attention_mask"].to(embedding_device)
        # Only the text around the video placeholders is embedded; the
        # placeholder block is replaced by the selected visual tokens.
        embed_dtype = embedding.weight.dtype
        hidden_size = int(embedding.weight.shape[-1])

        video_token_id = int(self.model.config.video_token_id)
        newline = self.model.model.image_newline
        compressed_embeds: list[torch.Tensor] = []
        stats: list[dict] = []
        for sample_index, sampled_frames in enumerate(sampled_frame_counts):
            sampled_frames = int(sampled_frames)
            feature_count = int(video_features[sample_index].shape[0])
            if sampled_frames < 1 or feature_count % sampled_frames:
                raise ValueError(
                    f"Cannot divide {feature_count} visual tokens across {sampled_frames} frames"
                )
            tokens_per_frame = feature_count // sampled_frames
            features = video_features[sample_index].reshape(sampled_frames, tokens_per_frame, -1)
            sample_spatial = (
                None
                if spatial_attention is None
                else spatial_attention[sample_index].to(features.device)
            )
            keep_visual, sample_stats = select_motiondrop_tokens(
                features,
                config,
                spatial_scores=sample_spatial,
            )
            selected_features = video_features[sample_index].index_select(
                0, keep_visual.to(video_features.device)
            )
            sample_stats.update(
                {
                    "schema": "llava_onevision_motiondrop_compression_v1",
                    "model_family": "llava_onevision",
                    "sampled_input_frames": sampled_frames,
                    "structural_visual_tokens": 1,
                    "structural_visual_token_type": "video_newline",
                    "synchronized_tensors": [
                        "projected_visual_embeddings",
                        "attention_mask",
                        "position_ids",
                    ],
                    "config": asdict(config),
                }
            )

            active = attention_mask[sample_index].bool()
            visual_positions = torch.nonzero(
                (input_ids[sample_index] == video_token_id) & active,
                as_tuple=False,
            ).flatten()
            expected_placeholders = feature_count + 1
            if int(visual_positions.numel()) != expected_placeholders:
                raise ValueError(
                    f"LLaVA video placeholder mismatch in sample {sample_index}: "
                    f"tokens={int(visual_positions.numel())}, expected={expected_placeholders}"
                )
            expected_positions = torch.arange(
                int(visual_positions[0]),
                int(visual_positions[0]) + expected_placeholders,
                device=visual_positions.device,
            )
            if not torch.equal(visual_positions, expected_positions):
                raise ValueError("LLaVA video placeholders must form one contiguous block")

            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            first_visual = int(visual_positions[0])
            last_visual = int(visual_positions[-1])
            prefix_indices = active_indices[active_indices < first_visual]
            suffix_indices = active_indices[active_indices > last_visual]
            visual_block = torch.cat(
                (
                    selected_features,
                    newline[None].to(selected_features.device, selected_features.dtype),
                ),
                dim=0,
            ).to(embedding_device, embed_dtype)
            compressed_embeds.append(
                torch.cat(
                    (
                        embedding(input_ids[sample_index, prefix_indices]),
                        visual_block,
                        embedding(input_ids[sample_index, suffix_indices]),
                    ),
                    dim=0,
                )
            )
            stats.append(sample_stats)

        lengths = [int(value.shape[0]) for value in compressed_embeds]
        max_length = max(lengths)
        packed_embeds = torch.zeros(
            batch_size, max_length, hidden_size, dtype=embed_dtype, device=embedding_device
        )
        packed_attention = attention_mask.new_zeros(batch_size, max_length)
        packed_positions = torch.zeros(
            batch_size, max_length, dtype=torch.long, device=embedding_device
        )
        for sample_index, length in enumerate(lengths):
            packed_embeds[sample_index, :length] = compressed_embeds[sample_index]
            packed_attention[sample_index, :length] = 1
            packed_positions[sample_index, :length] = torch.arange(length, device=embedding_device)

        outputs = self.model.model.language_model(
            input_ids=None,
            inputs_embeds=packed_embeds,
            attention_mask=packed_attention,
            position_ids=packed_positions,
            use_cache=False,
            return_dict=True,
        )
        output_device = outputs.last_hidden_state.device
        rows = torch.arange(batch_size, device=output_device)
        columns = torch.tensor(lengths, device=output_device) - 1
        last_hidden = outputs.last_hidden_state[rows, columns]

        def continuation_hidden(prefix_token_id: int) -> torch.Tensor:
            extended_length = max_length + 1
            extended_embeds = torch.zeros(
                batch_size, extended_length, hidden_size, dtype=embed_dtype, device=embedding_device
            )
            extended_attention = attention_mask.new_zeros(batch_size, extended_length)
            extended_positions = torch.zeros(
                batch_size,
                extended_length,
                dtype=torch.long,
                device=embedding_device,
            )
            prefix_ids = torch.full(
                (batch_size,),
                prefix_token_id,
                dtype=torch.long,
                device=embedding_device,
            )
            prefix_embeds = embedding(prefix_ids)
            for sample_index, length in enumerate(lengths):
                extended_embeds[sample_index, :length] = compressed_embeds[sample_index]
                extended_embeds[sample_index, length] = prefix_embeds[sample_index]
                extended_attention[sample_index, : length + 1] = 1
                extended_positions[sample_index, : length + 1] = torch.arange(
                    length + 1, device=embedding_device
                )
            continuation_outputs = self.model.model.language_model(
                input_ids=None,
                inputs_embeds=extended_embeds,
                attention_mask=extended_attention,
                position_ids=extended_positions,
                use_cache=False,
                return_dict=True,
            )
            output_device = continuation_outputs.last_hidden_state.device
            rows = torch.arange(batch_size, device=output_device)
            columns = torch.tensor(lengths, device=output_device)
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


__all__ = ["LlavaOnevisionMotionDropBatchAdapter"]
