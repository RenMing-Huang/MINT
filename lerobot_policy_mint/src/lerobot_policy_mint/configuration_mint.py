#!/usr/bin/env python

import os
from dataclasses import dataclass, field


from omegaconf import OmegaConf
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("mint")
@dataclass
class MINTConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    train_expert_only: bool = False  # If True, only train the action expert, keep paligemma frozen
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 16  # Number of action steps to predict, alignment with multi-scale vqvae
    n_action_steps: int = 1  # Number of action steps to execute
    label_smooth: float = 0.0  # Label smoothing for action prediction loss

    max_state_dim: int = 32
    max_action_dim: int = 7

    # Provide VQVAE checkpoint path. Runtime params are loaded from ckpt_dir/config.yaml.
    vqvae_name_or_path: str | None = ""

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # mint uses quantiles for state
            "ACTION": NormalizationMode.IDENTITY,  # mint uses identity for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Optimizer settings
    optimizer_lr: float = 2.5e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-5

    def __post_init__(self):
        super().__post_init__()

        self._load_vqvae_runtime_config()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def _load_vqvae_runtime_config(self) -> None:
        if not self.vqvae_name_or_path:
            raise ValueError("`vqvae_name_or_path` is required.")
        if OmegaConf is None:
            raise ImportError("OmegaConf is required to load VQVAE runtime config.")

        checkpoint_path, config_path = self._resolve_vqvae_paths(self.vqvae_name_or_path)

        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        model_cfg = cfg.get("model_cfg", {})
        quant_cfg = cfg.get("quant_cfg", {})

        for key, value in {**model_cfg, **quant_cfg}.items():
            setattr(self, key, value)

        assert self.chunk_size == cfg.get("horizon"), "chunk size should align with vqvae horizon"
        self.vqvae_name_or_path = checkpoint_path

    def _resolve_vqvae_paths(self, input_path: str) -> tuple[str, str]:
        """Resolve checkpoint and config paths from a checkpoint file or directory.

        Rules:
        - If input is a checkpoint file, load config from its parent directory.
        - If input is a directory:
          - choose the only .pth file if exactly one exists;
          - choose latest.pth if multiple .pth files exist and latest.pth exists;
          - otherwise raise an error.
        """
        resolved_input = os.path.abspath(os.path.expanduser(input_path))

        if os.path.isfile(resolved_input):
            ckpt_path = resolved_input
            ckpt_dir = os.path.dirname(ckpt_path)
        elif os.path.isdir(resolved_input):
            ckpt_dir = resolved_input
            pth_files = sorted(
                [os.path.join(ckpt_dir, fn) for fn in os.listdir(ckpt_dir) if fn.endswith(".pth")]
            )

            if len(pth_files) == 0:
                raise FileNotFoundError(f"No .pth checkpoint found in directory: {ckpt_dir}")
            if len(pth_files) == 1:
                ckpt_path = pth_files[0]
            else:
                latest_path = os.path.join(ckpt_dir, "latest.pth")
                if os.path.isfile(latest_path):
                    ckpt_path = latest_path
                else:
                    raise ValueError(
                        f"Multiple .pth checkpoints found in {ckpt_dir}, but latest.pth is missing. "
                        "Please provide a checkpoint file directly or create latest.pth."
                    )
        else:
            raise FileNotFoundError(f"Invalid checkpoint path: {input_path}")

        config_path = os.path.join(ckpt_dir, "config.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"VQVAE runtime config not found: {config_path}")

        return ckpt_path, config_path

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
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
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
