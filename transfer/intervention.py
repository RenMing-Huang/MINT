"""MINT-Light next-scale decoding with demonstration-token intervention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TransferOutput:
    actions: Tensor
    indices: Tensor
    intention: Tensor
    confidence: Tensor


def _pad_last_dim(value: Tensor, size: int) -> Tensor:
    if value.shape[-1] > size:
        raise ValueError(f"Action dimension {value.shape[-1]} exceeds VQ-VAE dimension {size}")
    return F.pad(value, (0, size - value.shape[-1]))


@torch.no_grad()
def encode_demo_tokens(model: nn.Module, demo_actions: Tensor) -> list[Tensor]:
    """Encode a demonstration action chunk with the VQ-VAE paired to MINT-Light."""
    horizon = model.config.chunk_size
    if demo_actions.ndim != 3 or demo_actions.shape[1] != horizon:
        raise ValueError(
            f"demo_actions must have shape [batch, {horizon}, action_dim], "
            f"got {tuple(demo_actions.shape)}"
        )
    actions = _pad_last_dim(demo_actions, model.config.max_action_dim)
    return model.vqvae.inp_to_idxBl(actions)


def _prepare_forced_indices(
    forced_indices: Sequence[Tensor | None] | None,
    patch_nums: Sequence[int],
    batch: int,
    num_samples: int,
    device: torch.device,
) -> list[Tensor | None]:
    if forced_indices is None:
        return [None] * len(patch_nums)
    if len(forced_indices) != len(patch_nums):
        raise ValueError(
            f"Expected {len(patch_nums)} forced-index entries, got {len(forced_indices)}"
        )

    prepared: list[Tensor | None] = []
    for level, (indices, size) in enumerate(zip(forced_indices, patch_nums, strict=True)):
        if indices is None:
            prepared.append(None)
            continue
        if indices.ndim != 2 or tuple(indices.shape) != (batch, size):
            raise ValueError(
                f"Forced indices at level {level} must have shape {(batch, size)}, "
                f"got {tuple(indices.shape)}"
            )
        prepared.append(indices.to(device=device, dtype=torch.long).repeat_interleave(num_samples, dim=0))
    return prepared


def _merge_samples(value: Tensor, batch: int, num_samples: int) -> Tensor:
    return value.reshape(batch, num_samples, *value.shape[1:]).mean(dim=1)


@torch.no_grad()
def sample_with_token_transfer(
    model: nn.Module,
    images: list[Tensor],
    state: Tensor,
    tokens: Tensor | None,
    token_mask: Tensor | None,
    forced_indices: Sequence[Tensor | None] | None = None,
    num_samples: int = 1,
) -> TransferOutput:
    """Sample MINT-Light actions while replacing selected VQ scales.

    ``forced_indices[level]`` is either ``None`` (sample from the policy) or a
    ``[batch, patch_nums[level]]`` tensor encoded from the demonstration. Level
    zero is the coarse intent token used by the one-shot transfer experiment.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    model.eval()
    context, context_mask, condition, cross_context, cross_mask = model._context(
        images, state, tokens, token_mask
    )
    original_batch = context.shape[0]
    patch_nums = tuple(model.config.patch_nums)
    prepared_forced = _prepare_forced_indices(
        forced_indices, patch_nums, original_batch, num_samples, context.device
    )

    if num_samples > 1:
        context = context.repeat_interleave(num_samples, dim=0)
        context_mask = context_mask.repeat_interleave(num_samples, dim=0)
        condition = condition.repeat_interleave(num_samples, dim=0)
        if cross_context is not None:
            cross_context = cross_context.repeat_interleave(num_samples, dim=0)
        if cross_mask is not None:
            cross_mask = cross_mask.repeat_interleave(num_samples, dim=0)

    batch = context.shape[0]
    context_levels = torch.zeros(context.shape[1], dtype=torch.long, device=context.device)
    context = context + model.level_embedding(context_levels).unsqueeze(0)
    model.transformer.kv_caching(True)
    sampled_indices: list[Tensor] = []
    confidence_by_scale: list[Tensor] = []
    intention = None

    try:
        context_attention_mask = context_mask[:, None, :].expand(-1, context.shape[1], -1)
        model.transformer(
            context,
            cond=condition,
            mask=context_attention_mask,
            cross_context=cross_context,
            cross_mask=cross_mask,
        )
        f_hat = torch.zeros(
            batch,
            model.config.codebook_dim,
            patch_nums[-1],
            dtype=context.dtype,
            device=context.device,
        )
        generated = 0
        query = model._action_start_query(condition)

        for scale, size in enumerate(patch_nums):
            level = model.level_embedding.weight[scale + 1].view(1, 1, -1)
            if scale == 0:
                query = query.expand(-1, size, -1)
            positions = model.action_position_embedding(
                torch.arange(generated, generated + size, device=context.device)
            ).unsqueeze(0)
            query = query + level + positions
            cache_mask = torch.cat(
                [
                    context_mask,
                    torch.ones(
                        batch, generated + size, dtype=torch.bool, device=context.device
                    ),
                ],
                dim=1,
            )
            attention_mask = cache_mask[:, None, :].expand(-1, size, -1)
            hidden = model.transformer(
                query,
                cond=condition,
                mask=attention_mask,
                cross_context=cross_context,
                cross_mask=cross_mask,
            )
            logits = model.codebook_head(model.head_norm(hidden))
            probabilities = logits.softmax(dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
            confidence_by_scale.append(
                1.0 - entropy.mean(dim=1) / math.log(model.config.codebook_size)
            )

            indices = prepared_forced[scale]
            if indices is None:
                indices = model._sample_codebook_indices(logits)
            sampled_indices.append(indices)
            embeddings = model.vqvae.quantizer.embedding(indices).transpose(1, 2).contiguous()
            if scale == 0:
                intention = hidden.mean(dim=1)
            f_hat, next_scale = model.vqvae.quantizer.get_next_autoregressive_input(
                scale, len(patch_nums), f_hat, embeddings
            )
            generated += size
            if scale + 1 < len(patch_nums):
                query = model.vq_embedding(next_scale.transpose(1, 2))
    finally:
        model.transformer.kv_caching(False)

    if intention is None:
        raise RuntimeError("MINT-Light patch schedule is empty")
    actions = model.vqvae.decoder(model.vqvae.post_quant_conv(f_hat))
    if model.vqvae.patchwise_proj is not None:
        actions = model.vqvae.patchwise_proj(actions)
        gripper = (actions[..., 6:8].argmax(dim=-1, keepdim=True) * 2 - 1).to(actions.dtype)
        actions = torch.cat([actions[..., :6], gripper], dim=-1)

    indices = torch.cat(sampled_indices, dim=1)
    confidence = torch.stack(confidence_by_scale, dim=1).mean(dim=1, keepdim=True)
    if num_samples > 1:
        actions = _merge_samples(actions, original_batch, num_samples)
        # Patchwise VQ-VAE decodes the gripper to {-1, +1}; keep it discrete after voting.
        if model.vqvae.patchwise_proj is not None and actions.shape[-1] == 7:
            actions[..., 6] = torch.where(actions[..., 6] >= 0, 1.0, -1.0)
        intention = _merge_samples(intention, original_batch, num_samples)
        confidence = _merge_samples(confidence, original_batch, num_samples)
        indices = indices.reshape(original_batch, num_samples, -1)[:, 0]

    return TransferOutput(
        actions=actions,
        indices=indices,
        intention=intention,
        confidence=confidence,
    )
