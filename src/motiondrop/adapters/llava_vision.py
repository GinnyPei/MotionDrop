"""Shared LLaVA-OneVision feature and attention extraction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling

from motiondrop.runtime import module_device

# Frames per vision-tower call.
FRAME_CHUNK = 16


def encode_video_features_with_attention(
    model,
    pixels: torch.Tensor,
    *,
    capture_attention: bool,
):
    """Encode video and capture retained-layer mean-received patch attention."""

    captured: list[torch.Tensor] = []
    handle = None
    if capture_attention:
        try:
            layers = model.model.vision_tower.vision_model.encoder.layers
            layer_index = model.config.vision_feature_layer
            if isinstance(layer_index, list):
                layer_index = layer_index[-1]
            attention = layers[int(layer_index)].self_attn

            def capture(module, args, kwargs):
                hidden = kwargs.get("hidden_states")
                if hidden is None:
                    hidden = args[0]
                frame_scores = []
                for frame_hidden in hidden:
                    length = int(frame_hidden.shape[0])
                    queries = (
                        module.q_proj(frame_hidden)
                        .view(length, module.num_heads, module.head_dim)
                        .transpose(0, 1)
                    )
                    keys = (
                        module.k_proj(frame_hidden)
                        .view(length, module.num_heads, module.head_dim)
                        .transpose(0, 1)
                    )
                    weights = torch.softmax(
                        (queries.float() @ keys.float().transpose(-1, -2)) * module.scale,
                        dim=-1,
                    )
                    frame_scores.append(weights.mean(dim=(0, 1)).detach())
                captured.append(torch.stack(frame_scores))

            handle = attention.register_forward_pre_hook(capture, with_kwargs=True)
        except (AttributeError, IndexError, TypeError, ValueError):
            handle = None

    visual_device = module_device(model.model.vision_tower)
    try:
        # Frames are encoded independently, so chunking only bounds the memory
        # held by the vision tower's per-layer hidden states.
        samples = []
        for sample in pixels:
            chunks = [
                model.model.get_video_features(
                    sample[start : start + FRAME_CHUNK].unsqueeze(0).to(visual_device),
                    vision_feature_layer=model.config.vision_feature_layer,
                    vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                    return_dict=True,
                ).pooler_output
                for start in range(0, int(sample.shape[0]), FRAME_CHUNK)
            ]
            samples.append(torch.cat(chunks, dim=1))
        outputs = BaseModelOutputWithPooling(pooler_output=torch.cat(samples, dim=0))
    finally:
        if handle is not None:
            handle.remove()

    if not captured:
        return outputs, None
    attention_scores = torch.cat(captured, dim=0)
    if model.config.vision_feature_select_strategy == "default":
        attention_scores = attention_scores[:, 1:]

    batch_size, sampled_frames = pixels.shape[:2]
    target_tokens = int(outputs.pooler_output.shape[1]) // sampled_frames
    source_tokens = int(attention_scores.shape[1])
    source_side = math.isqrt(source_tokens)
    target_side = math.isqrt(target_tokens)
    if source_side * source_side != source_tokens or target_side * target_side != target_tokens:
        return outputs, None
    pooled = F.interpolate(
        attention_scores.float().reshape(-1, 1, source_side, source_side),
        size=(target_side, target_side),
        mode="bilinear",
        align_corners=False,
    ).flatten(1)
    return outputs, pooled.reshape(batch_size, sampled_frames, target_tokens)


__all__ = ["encode_video_features_with_attention"]
