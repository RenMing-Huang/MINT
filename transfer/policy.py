"""Checkpoint-backed MINT-Light adapter for one-shot token transfer."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .intervention import TransferOutput, encode_demo_tokens, sample_with_token_transfer


def _resolve_vqvae_checkpoint(reference: str | Path) -> str:
    value = str(reference)
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    if path.is_dir():
        candidates = sorted(path.glob("*.pth"))
        if len(candidates) != 1:
            raise ValueError(
                f"VQ-VAE directory must contain exactly one .pth file, found {len(candidates)}"
            )
        if not (path / "config.yaml").is_file():
            raise FileNotFoundError(f"VQ-VAE config not found: {path / 'config.yaml'}")
        return str(candidates[0].resolve())
    if path.suffix == ".pth" or value.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {path}")

    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=value, filename="config.yaml")
    return hf_hub_download(repo_id=value, filename="ms_vqvae.pth")


class _InterventionModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        images: list[Tensor],
        state: Tensor,
        tokens: Tensor,
        token_mask: Tensor,
        forced_indices: Sequence[Tensor | None] | None,
        num_samples: int,
    ) -> TransferOutput:
        return sample_with_token_transfer(
            self.model,
            images,
            state,
            tokens,
            token_mask,
            forced_indices=forced_indices,
            num_samples=num_samples,
        )


class MINTLightTransferPolicy:
    """Uses the policy EMA weights and its paired VQ-VAE for intervention inference."""

    def __init__(self, policy: nn.Module, preprocessor, postprocessor):
        if getattr(policy, "name", None) != "mint_light":
            raise ValueError(f"Expected a MINT-Light policy, got {getattr(policy, 'name', None)!r}")
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.sampler = _InterventionModule(policy.model)
        self.action_queue: deque[Tensor] = deque()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        device: str = "cuda",
        vqvae_name_or_path: str | None = None,
        n_action_steps: int | None = None,
    ) -> "MINTLightTransferPolicy":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot_policy_mint_light import MINTLightConfig, MINTLightPolicy

        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, MINTLightConfig):
            raise ValueError(
                f"Checkpoint type must be mint_light, got {getattr(config, 'type', None)!r}"
            )
        if config.use_language:
            raise ValueError(
                "One-shot transfer requires a visual-only MINT-Light checkpoint "
                "trained with use_language=false."
            )
        config.device = device
        # The full checkpoint already contains both frozen backbones. This flag
        # prevents their constructors from fetching foundation weights first.
        config.pretrained_path = str(checkpoint)
        if n_action_steps is not None:
            if n_action_steps > config.future_action_steps:
                raise ValueError(
                    f"n_action_steps {n_action_steps} exceeds horizon {config.future_action_steps}"
                )
            config.n_action_steps = n_action_steps
        if vqvae_name_or_path is not None:
            vqvae_name_or_path = _resolve_vqvae_checkpoint(vqvae_name_or_path)
            architecture = (
                config.max_action_dim,
                config.codebook_size,
                config.codebook_dim,
                config.vqvae_ch,
                tuple(config.vqvae_ch_mult),
                tuple(config.patch_nums),
                config.vqvae_patchwise,
            )
            config.vqvae_name_or_path = vqvae_name_or_path
            config._load_vqvae_config()
            overridden_architecture = (
                config.max_action_dim,
                config.codebook_size,
                config.codebook_dim,
                config.vqvae_ch,
                tuple(config.vqvae_ch_mult),
                tuple(config.patch_nums),
                config.vqvae_patchwise,
            )
            if overridden_architecture != architecture:
                raise ValueError(
                    "The VQ-VAE override architecture differs from the policy checkpoint. "
                    "Use the paired tokenizer saved in config.json or a shape-compatible checkpoint."
                )
        policy = MINTLightPolicy.from_pretrained(checkpoint, config=config)
        if vqvae_name_or_path is not None:
            # The full policy checkpoint contains its original frozen VQ-VAE.
            # Apply an explicit compatible override after loading policy weights.
            policy.model.vqvae.load_vqvae_weights(config.vqvae_name_or_path)
            policy.model.vqvae.requires_grad_(False).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        return cls(policy, preprocessor, postprocessor)

    @property
    def config(self):
        return self.policy.config

    @property
    def model(self):
        return self.policy.model

    def reset(self) -> None:
        self.action_queue.clear()
        self.policy.reset()

    @torch.no_grad()
    def encode_demo(self, actions: Tensor) -> list[Tensor]:
        return encode_demo_tokens(self.model, actions.to(self.config.device))

    @torch.no_grad()
    def predict(
        self,
        batch: dict[str, Tensor],
        demo_actions: Tensor | None,
        force_levels: Sequence[int] = (0,),
        num_samples: int = 1,
    ) -> TransferOutput:
        forced: list[Tensor | None] | None = None
        if demo_actions is not None:
            demo_indices = self.encode_demo(demo_actions)
            forced = [None] * len(demo_indices)
            for level in force_levels:
                if level < 0 or level >= len(demo_indices):
                    raise ValueError(
                        f"force level {level} is outside [0, {len(demo_indices) - 1}]"
                    )
                forced[level] = demo_indices[level]

        ema_parameters = {
            f"model.{name}": parameter for name, parameter in self.policy._ema_state().items()
        }
        output = functional_call(
            self.sampler,
            ema_parameters,
            (*self.policy._inputs(batch), forced, num_samples),
            strict=False,
        )
        action_dim = self.config.output_features["action"].shape[0]
        output.actions = output.actions[
            :, self.config.action_history_steps :, :action_dim
        ]
        return output

    def ensemble(self, output: TransferOutput) -> Tensor:
        if self.config.intent_ensemble:
            self.policy.ensembler.add_chunk(
                output.actions, output.intention, output.confidence
            )
            return self.policy.ensembler.get_ensembled_actions()
        return output.actions[:, : self.config.n_action_steps]
