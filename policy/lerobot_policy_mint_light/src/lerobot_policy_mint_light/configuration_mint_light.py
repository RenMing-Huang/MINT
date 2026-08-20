#!/usr/bin/env python

import json
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub.constants import CONFIG_NAME
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from omegaconf import OmegaConf


@PreTrainedConfig.register_subclass("mint_light")
@dataclass
class MINTLightConfig(PreTrainedConfig):
    """Configuration for the compact MINT next-scale autoregressive policy."""

    VISION_BACKBONE_ID = "dinov3-vit-l-224px"
    LANGUAGE_MODEL_NAME = "google/siglip2-so400m-patch14-224"

    n_obs_steps: int = 1
    chunk_size: int = 16
    action_history_steps: int = 0
    n_action_steps: int = 4
    max_state_dim: int = 32
    max_action_dim: int = 7
    image_resolution: tuple[int, int] = (224, 224)

    hidden_dim: int = 384
    num_layers: int = 10
    num_heads: int = 6
    head_dim: int = 64
    mlp_hidden_dim: int = 1024
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.05
    drop_path_rate: float = 0.0
    cross_attention_every: int = 2
    vision_backbone_id: str = VISION_BACKBONE_ID
    vision_pretrained: bool = True
    language_model_name: str = LANGUAGE_MODEL_NAME
    language_pretrained: bool = True
    tokenizer_max_length: int = 64
    state_hidden_dim: int = 128
    start_query_hidden_dim: int = 384
    classifier_hidden_dim: int = 384
    ema_decay: float = 0.999
    sample_top_k: int = 3
    intent_ensemble: bool = True
    intent_ensemble_temperature: float = 0.5
    intent_ensemble_decay: float = 0.03

    codebook_size: int = 512
    codebook_dim: int = 32
    patch_nums: tuple[int, ...] = (1, 2, 4)
    vqvae_name_or_path: str | None = None
    # Decoder-only field for configs saved before 1.0; it has no runtime behavior.
    allow_random_vqvae: bool = field(default=False, init=False, repr=False)
    vqvae_ch: int = 32
    vqvae_ch_mult: tuple[int, ...] = (2, 4, 8)
    vqvae_dropout: float = 0.0
    vqvae_beta: float = 0.25
    vqvae_using_znorm: bool = True
    vqvae_quant_conv_ks: int = 3
    vqvae_quant_resi: float = 0.5
    vqvae_share_quant_resi: int = 0
    vqvae_patchwise: dict | None = None

    optimizer_lr: float = 3e-4
    optimizer_weight_decay: float = 0.005
    optimizer_grad_clip_norm: float = 2.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 9e-5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.vision_backbone_id != self.VISION_BACKBONE_ID:
            raise ValueError(
                "MINT-Light 1.0 only supports the DINOv3 ViT-L/16 visual encoder "
                f"({self.VISION_BACKBONE_ID}); got {self.vision_backbone_id}"
            )
        if not self.vision_pretrained:
            raise ValueError("MINT-Light 1.0 requires pretrained DINOv3 weights")
        if self.language_model_name != self.LANGUAGE_MODEL_NAME:
            raise ValueError(
                "MINT-Light 1.0 only supports the fixed SigLIP text encoder "
                f"({self.LANGUAGE_MODEL_NAME}); got {self.language_model_name}"
            )
        if not self.language_pretrained:
            raise ValueError("MINT-Light 1.0 requires pretrained SigLIP text weights")
        if tuple(self.image_resolution) != (224, 224):
            raise ValueError("MINT-Light 1.0 uses a fixed encoder resolution of 224x224")
        if self.vqvae_name_or_path:
            self._load_vqvae_config()
        if (
            self.hidden_dim <= 0
            or self.num_layers <= 0
            or self.num_heads <= 0
            or self.head_dim <= 0
            or self.mlp_hidden_dim <= 0
        ):
            raise ValueError("Transformer dimensions must be positive")
        if self.num_heads * self.head_dim != self.hidden_dim:
            raise ValueError("num_heads * head_dim must equal hidden_dim")
        if self.cross_attention_every <= 0:
            raise ValueError("cross_attention_every must be positive")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0.0 <= self.ffn_dropout < 1.0:
            raise ValueError("ffn_dropout must be in [0, 1)")
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if self.start_query_hidden_dim <= 0 or self.classifier_hidden_dim <= 0:
            raise ValueError("Projection dimensions must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 1 <= self.sample_top_k <= self.codebook_size:
            raise ValueError("sample_top_k must be in [1, codebook_size]")
        if self.intent_ensemble_temperature <= 0.0:
            raise ValueError("intent_ensemble_temperature must be positive")
        if self.intent_ensemble_decay < 0.0:
            raise ValueError("intent_ensemble_decay must be non-negative")
        if not 0 <= self.action_history_steps < self.chunk_size:
            raise ValueError("action_history_steps must be in [0, chunk_size)")
        if self.n_action_steps > self.future_action_steps:
            raise ValueError("n_action_steps cannot exceed the future action horizon")
        if not self.vqvae_name_or_path and self.pretrained_path is None:
            raise ValueError(
                "vqvae_name_or_path is required when training MINT-Light from scratch"
            )
        latent_horizon = self.chunk_size // (2 ** (len(self.vqvae_ch_mult) - 1))
        if not self.patch_nums or self.patch_nums[-1] != latent_horizon:
            raise ValueError(
                "patch_nums[-1] must match the VQ-VAE latent horizon: "
                "chunk_size / 2**(len(vqvae_ch_mult) - 1)"
            )

    def _load_vqvae_config(self) -> None:
        checkpoint = Path(self.vqvae_name_or_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint}")
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"VQ-VAE config not found: {config_path}")

        runtime = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        model_cfg = runtime.get("model_cfg", {})
        quant_cfg = runtime.get("quant_cfg", {})
        self.chunk_size = int(runtime["horizon"])
        self.max_action_dim = int(model_cfg.get("action_dim", self.max_action_dim))
        self.codebook_size = int(model_cfg.get("codebook_size", self.codebook_size))
        self.codebook_dim = int(model_cfg.get("codebook_dim", self.codebook_dim))
        self.vqvae_ch = int(model_cfg.get("ch", self.vqvae_ch))
        self.vqvae_ch_mult = tuple(model_cfg.get("ch_mult", self.vqvae_ch_mult))
        self.vqvae_dropout = float(model_cfg.get("dropout", self.vqvae_dropout))
        self.vqvae_patchwise = model_cfg.get("patchwise", self.vqvae_patchwise)
        self.patch_nums = tuple(quant_cfg.get("patch_nums", self.patch_nums))
        self.vqvae_beta = float(quant_cfg.get("beta", self.vqvae_beta))
        self.vqvae_using_znorm = bool(
            quant_cfg.get("using_znorm", quant_cfg.get("znorm", self.vqvae_using_znorm))
        )
        self.vqvae_quant_conv_ks = int(
            quant_cfg.get("quant_conv_ks", self.vqvae_quant_conv_ks)
        )
        self.vqvae_quant_resi = float(quant_cfg.get("quant_resi", self.vqvae_quant_resi))
        self.vqvae_share_quant_resi = int(
            quant_cfg.get("share_quant_resi", self.vqvae_share_quant_resi)
        )
        self.vqvae_name_or_path = str(checkpoint)

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}
        self.input_features.setdefault(
            "observation.state", PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,))
        )
        self.output_features.setdefault(
            "action", PolicyFeature(type=FeatureType.ACTION, shape=(self.max_action_dim,))
        )
        if len(self.image_features) != 2:
            raise ValueError(
                f"MINT-Light follows MINT-30M and requires exactly two visual inputs; got {len(self.image_features)}"
            )

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        config_path = save_directory / CONFIG_NAME
        payload = json.loads(config_path.read_text())
        payload.pop("allow_random_vqvae", None)
        config_path.write_text(json.dumps(payload, indent=4) + "\n")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(-self.action_history_steps, self.future_action_steps))

    @property
    def future_action_steps(self) -> int:
        return self.chunk_size - self.action_history_steps

    @property
    def reward_delta_indices(self) -> None:
        return None
