#!/usr/bin/env python

from collections import deque
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .backbones import (
    DinoV3Backbone,
    LanguageEncoder,
    VisualFeatureProjector,
)
from .configuration_mint_light import MINTLightConfig
from .mint_utils import IntentionEnsembler, MultiScaleVQVAE
from .transformer import AdaLNTransformer


def _pad_last_dim(value: Tensor, size: int) -> Tensor:
    if value.shape[-1] > size:
        raise ValueError(f"Feature dimension {value.shape[-1]} exceeds configured size {size}")
    return F.pad(value, (0, size - value.shape[-1]))


class MINTLightModel(nn.Module):
    """LeRobot adaptation of the MINT-30M action chunk autoregressor."""

    def __init__(self, config: MINTLightConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        load_backbone_weights = config.pretrained_path is None

        self.vision_backbone = DinoV3Backbone(
            load_pretrained_weights=load_backbone_weights
        )
        self.vision_backbone.requires_grad_(False)
        self.visual_fusion = VisualFeatureProjector(self.vision_backbone.feature_dim, dim)
        self.language_encoder = LanguageEncoder(
            dim,
            load_pretrained_weights=load_backbone_weights,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(config.max_state_dim, config.state_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.state_hidden_dim, dim),
        )
        self.vq_embedding = nn.Linear(config.codebook_dim, dim)
        self.start_query = nn.Sequential(
            nn.Linear(dim, config.start_query_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.start_query_hidden_dim, dim),
        )
        # Modality order matches MINT-30M: visual, state, language.
        self.modality_embedding = nn.Embedding(3, dim)
        self.camera_embedding = nn.Embedding(2, dim)
        self.level_embedding = nn.Embedding(len(config.patch_nums) + 2, dim)
        grid_height, grid_width = self.vision_backbone.grid_size
        self.visual_row_embedding = nn.Embedding(grid_height, dim)
        self.visual_col_embedding = nn.Embedding(grid_width, dim)
        self.state_position_embedding = nn.Embedding(1, dim)
        self.language_position_embedding = nn.Embedding(config.tokenizer_max_length, dim)
        self.action_position_embedding = nn.Embedding(sum(config.patch_nums), dim)
        self.transformer = AdaLNTransformer(
            dim=dim,
            depth=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            mlp_hidden_dim=config.mlp_hidden_dim,
            attention_dropout=config.attention_dropout,
            ffn_dropout=config.ffn_dropout,
            drop_path_rate=config.drop_path_rate,
            cross_attention_every=config.cross_attention_every,
        )
        self.head_norm = nn.LayerNorm(dim, eps=1e-6)
        self.codebook_head = nn.Sequential(
            nn.Linear(dim, config.classifier_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.classifier_hidden_dim, config.codebook_size),
        )

        self.vqvae = MultiScaleVQVAE(
            seq_dim=config.max_action_dim,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            ch=config.vqvae_ch,
            ch_mult=config.vqvae_ch_mult,
            patch_nums=config.patch_nums,
            dropout=config.vqvae_dropout,
            beta=config.vqvae_beta,
            using_znorm=config.vqvae_using_znorm,
            quant_conv_ks=config.vqvae_quant_conv_ks,
            quant_resi=config.vqvae_quant_resi,
            share_quant_resi=config.vqvae_share_quant_resi,
            patchwise=config.vqvae_patchwise,
        )
        if config.vqvae_name_or_path:
            self.vqvae.load_vqvae_weights(str(config.vqvae_name_or_path))
        self.vqvae.requires_grad_(False)
        self.vqvae.eval()
        self._init_trainable_weights()
        self.transformer.reset_adaln_zero()

    def _init_trainable_weights(self) -> None:
        init_std = math.sqrt(1.0 / self.config.hidden_dim / 3.0)
        for module in self.modules():
            parameters = list(module.parameters(recurse=False))
            if parameters and all(not parameter.requires_grad for parameter in parameters):
                continue
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=init_std)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_backbone.eval()
        self.vqvae.eval()
        self.language_encoder.encoder.eval()
        return self

    def _visual_positions(self, device: torch.device) -> Tensor:
        grid_height, grid_width = self.vision_backbone.grid_size
        rows = torch.arange(grid_height, device=device).repeat_interleave(grid_width)
        cols = torch.arange(grid_width, device=device).repeat(grid_height)
        return (
            self.visual_row_embedding(rows) + self.visual_col_embedding(cols)
        ).unsqueeze(0)

    def _context(
        self, images: list[Tensor], state: Tensor, tokens: Tensor, token_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        visual_tokens = []
        visual_position = self._visual_positions(state.device)
        with torch.no_grad():
            backbone_features = [self.vision_backbone(image) for image in images]
        for camera, features in enumerate(backbone_features):
            visual = self.visual_fusion(features)
            visual = visual + visual_position
            visual = visual + self.camera_embedding.weight[camera].view(1, 1, -1)
            visual_tokens.append(visual)
        visual = torch.cat(visual_tokens, dim=1)
        state_token = self.state_encoder(
            _pad_last_dim(state, self.config.max_state_dim)
        ).unsqueeze(1)
        state_token = state_token + self.state_position_embedding.weight.view(1, 1, -1)
        language, language_tokens = self.language_encoder(tokens, token_mask)
        language_length = language_tokens.shape[1]
        if language_length > self.config.tokenizer_max_length:
            raise ValueError(
                f"Language sequence length {language_length} exceeds tokenizer_max_length "
                f"{self.config.tokenizer_max_length}"
            )
        language_positions = self.language_position_embedding(
            torch.arange(language_length, device=language_tokens.device)
        ).unsqueeze(0)
        language_tokens = language_tokens + language_positions

        visual = visual + self.modality_embedding.weight[0].view(1, 1, -1)
        state_token = state_token + self.modality_embedding.weight[1].view(1, 1, -1)
        language_tokens = language_tokens + self.modality_embedding.weight[2].view(1, 1, -1)
        context = torch.cat([visual, state_token, language_tokens], dim=1)
        batch = context.shape[0]
        prefix_mask = torch.ones(
            batch,
            visual.shape[1] + state_token.shape[1],
            dtype=torch.bool,
            device=context.device,
        )
        context_mask = torch.cat([prefix_mask, token_mask.bool()], dim=1)
        return context, context_mask, language, language_tokens, token_mask.bool()

    def _level_layout(self, context_length: int, device: torch.device) -> Tensor:
        return torch.cat(
            [
                torch.zeros(context_length, dtype=torch.long, device=device),
                *[
                    torch.full((size,), level + 1, dtype=torch.long, device=device)
                    for level, size in enumerate(self.config.patch_nums)
                ],
            ]
        )

    def _training_logits(
        self,
        context: Tensor,
        context_mask: Tensor,
        language: Tensor,
        language_tokens: Tensor,
        language_mask: Tensor,
        action_inputs: Tensor,
    ) -> Tensor:
        context_length = context.shape[1]
        action_length = action_inputs.shape[1]
        action_positions = self.action_position_embedding(
            torch.arange(action_length, device=action_inputs.device)
        ).unsqueeze(0)
        action_inputs = action_inputs + action_positions
        sequence = torch.cat([context, action_inputs], dim=1)
        levels = self._level_layout(context_length, sequence.device)
        sequence = sequence + self.level_embedding(levels).unsqueeze(0)
        # A query may see context and tokens from its own or earlier scale.
        level_mask = levels[:, None] >= levels[None, :]
        valid_keys = torch.cat(
            [
                context_mask,
                torch.ones(
                    sequence.shape[0], action_length, dtype=torch.bool, device=sequence.device
                ),
            ],
            dim=1,
        )
        mask = level_mask.unsqueeze(0) & valid_keys.unsqueeze(1)
        output = self.transformer(
            sequence,
            cond=language,
            mask=mask,
            cross_context=language_tokens,
            cross_mask=language_mask,
        )
        return self.codebook_head(self.head_norm(output[:, context_length:]))

    def _teacher_forcing_inputs(self, indices: list[Tensor], language: Tensor) -> Tensor:
        next_scale = self.vqvae.quantizer.idxBl_to_next_scale_input(indices)
        first = self.start_query(language).unsqueeze(1).expand(-1, self.config.patch_nums[0], -1)
        return torch.cat([first, self.vq_embedding(next_scale)], dim=1)

    def forward(
        self,
        images: list[Tensor],
        state: Tensor,
        tokens: Tensor,
        token_mask: Tensor,
        actions: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if actions is None:
            return self._sample_with_metadata(images, state, tokens, token_mask)
        with torch.no_grad():
            indices = self.vqvae.inp_to_idxBl(actions)
        targets = torch.cat(indices, dim=1)
        context, context_mask, language, language_tokens, language_mask = self._context(
            images, state, tokens, token_mask
        )
        logits = self._training_logits(
            context,
            context_mask,
            language,
            language_tokens,
            language_mask,
            self._teacher_forcing_inputs(indices, language),
        )
        return F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")

    @torch.no_grad()
    def sample(self, images: list[Tensor], state: Tensor, tokens: Tensor, token_mask: Tensor) -> Tensor:
        actions, _, _ = self._sample_with_metadata(images, state, tokens, token_mask)
        return actions

    @torch.no_grad()
    def sample_with_intention(
        self, images: list[Tensor], state: Tensor, tokens: Tensor, token_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        actions, intention, _ = self._sample_with_metadata(images, state, tokens, token_mask)
        return actions, intention

    def _sample_codebook_indices(self, logits: Tensor) -> Tensor:
        top_logits, top_indices = logits.topk(self.config.sample_top_k, dim=-1)
        if self.config.sample_top_k == 1:
            return top_indices.squeeze(-1)
        probabilities = top_logits.softmax(dim=-1)
        sampled = torch.multinomial(
            probabilities.reshape(-1, self.config.sample_top_k), num_samples=1
        ).view(*logits.shape[:-1], 1)
        return top_indices.gather(dim=-1, index=sampled).squeeze(-1)

    @torch.no_grad()
    def _sample_with_metadata(
        self, images: list[Tensor], state: Tensor, tokens: Tensor, token_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        context, context_mask, language, language_tokens, language_mask = self._context(
            images, state, tokens, token_mask
        )
        batch = context.shape[0]
        context_levels = torch.zeros(context.shape[1], dtype=torch.long, device=context.device)
        context = context + self.level_embedding(context_levels).unsqueeze(0)

        self.transformer.kv_caching(True)
        try:
            context_attention_mask = context_mask[:, None, :].expand(-1, context.shape[1], -1)
            self.transformer(
                context,
                cond=language,
                mask=context_attention_mask,
                cross_context=language_tokens,
                cross_mask=language_mask,
            )
            f_hat = torch.zeros(
                batch, self.config.codebook_dim, self.config.patch_nums[-1], device=context.device
            )
            generated = 0
            query = self.start_query(language).unsqueeze(1)
            intention = None
            confidence_by_scale = []
            for scale, size in enumerate(self.config.patch_nums):
                level = self.level_embedding.weight[scale + 1].view(1, 1, -1)
                if scale == 0:
                    query = query.expand(-1, size, -1)
                positions = self.action_position_embedding(
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
                hidden = self.transformer(
                    query,
                    cond=language,
                    mask=attention_mask,
                    cross_context=language_tokens,
                    cross_mask=language_mask,
                )
                logits = self.codebook_head(self.head_norm(hidden))
                probabilities = logits.softmax(dim=-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
                confidence_by_scale.append(
                    1.0 - entropy.mean(dim=1) / math.log(self.config.codebook_size)
                )
                indices = self._sample_codebook_indices(logits)
                embeddings = self.vqvae.quantizer.embedding(indices).transpose(1, 2).contiguous()
                if scale == 0:
                    intention = hidden.mean(dim=1)
                f_hat, next_scale = self.vqvae.quantizer.get_next_autoregressive_input(
                    scale, len(self.config.patch_nums), f_hat, embeddings
                )
                generated += size
                if scale + 1 < len(self.config.patch_nums):
                    query = self.vq_embedding(next_scale.transpose(1, 2))
        finally:
            self.transformer.kv_caching(False)

        actions = self.vqvae.decoder(self.vqvae.post_quant_conv(f_hat))
        if self.vqvae.patchwise_proj is not None:
            actions = self.vqvae.patchwise_proj(actions)
            gripper = (actions[..., 6:8].argmax(dim=-1, keepdim=True) * 2 - 1).to(actions.dtype)
            actions = torch.cat([actions[..., :6], gripper], dim=-1)
        if intention is None:
            raise RuntimeError("Cannot compute an intention from an empty patch schedule")
        confidence = torch.stack(confidence_by_scale, dim=1).mean(dim=1, keepdim=True)
        return actions, intention, confidence


class MINTLightPolicy(PreTrainedPolicy):
    config_class = MINTLightConfig
    name = "mint_light"

    def __init__(self, config: MINTLightConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = MINTLightModel(config).to(config.device)
        self.ema_parameters = nn.ParameterList(
            [
                nn.Parameter(parameter.detach().clone(), requires_grad=False)
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
        )
        self.ensembler = IntentionEnsembler(
            horizon=config.future_action_steps,
            n_action_steps=config.n_action_steps,
            temperature=config.intent_ensemble_temperature,
            temporal_decay=config.intent_ensemble_decay,
        )
        self.reset()

    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        if self.ensembler is not None:
            self.ensembler.reset()

    def get_optim_params(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @torch.no_grad()
    def update(self) -> None:
        for ema_parameter, parameter in zip(
            self.ema_parameters, self._trainable_parameters(), strict=True
        ):
            ema_parameter.mul_(self.config.ema_decay).add_(
                parameter, alpha=1.0 - self.config.ema_decay
            )

    def _trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def _ema_state(self) -> dict[str, Tensor]:
        names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        return dict(zip(names, self.ema_parameters, strict=True))

    def _inputs(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], Tensor, Tensor, Tensor]:
        missing = [key for key in self.config.image_features if key not in batch]
        if missing:
            raise ValueError(f"Configured visual inputs are missing from the batch: {missing}")
        images = [batch[key].to(self.config.device) for key in self.config.image_features]
        return (
            images,
            batch[OBS_STATE].to(self.config.device),
            batch[OBS_LANGUAGE_TOKENS].to(self.config.device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(self.config.device),
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        actions, _, _ = self._predict_action_chunk_with_intention(batch)
        return actions

    @torch.no_grad()
    def _predict_action_chunk_with_intention(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.eval()
        actions, intention, confidence = functional_call(
            self.model, self._ema_state(), self._inputs(batch)
        )
        action_dim = self.config.output_features[ACTION].shape[0]
        future = actions[:, self.config.action_history_steps :, :action_dim]
        return future, intention, confidence

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if not self._action_queue:
            chunk, intention, confidence = self._predict_action_chunk_with_intention(batch)
            if self.config.intent_ensemble:
                self.ensembler.add_chunk(chunk, intention, confidence)
                chunk = self.ensembler.get_ensembled_actions()
            else:
                chunk = chunk[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        actions = _pad_last_dim(batch[ACTION].to(self.config.device), self.config.max_action_dim)
        losses = self.model(*self._inputs(batch), actions)
        per_sample = losses.mean(dim=1)
        loss = per_sample if reduction == "none" else per_sample.mean()
        return loss, {"loss": per_sample.mean().item()}
