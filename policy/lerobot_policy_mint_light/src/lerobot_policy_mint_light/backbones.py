"""Frozen production encoders for MINT-Light."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DinoV3Backbone(nn.Module):
    """Frozen DINOv3 ViT-L/16 patch-token encoder."""

    MODEL_NAME = "vit_large_patch16_dinov3.lvd1689m"

    def __init__(self, load_pretrained_weights: bool = True):
        super().__init__()
        import timm

        self.dino = timm.create_model(
            self.MODEL_NAME,
            pretrained=load_pretrained_weights,
            num_classes=0,
            img_size=224,
        )
        self.feature_dim = self.dino.embed_dim
        self.grid_size = tuple(self.dino.patch_embed.grid_size)
        self.num_patches = math.prod(self.grid_size)
        config = getattr(self.dino, "pretrained_cfg", {})
        mean = config.get("mean", (0.485, 0.456, 0.406))
        std = config.get("std", (0.229, 0.224, 0.225))
        self.register_buffer(
            "dino_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "dino_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: Tensor) -> Tensor:
        images = F.interpolate(
            images, (224, 224), mode="bicubic", align_corners=False, antialias=True
        )
        hidden = self.dino.forward_features((images - self.dino_mean) / self.dino_std)
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise RuntimeError(
                f"Expected {type(self.dino).__name__}.forward_features to return [B, N, C] tokens"
            )
        if hidden.shape[1] < self.num_patches:
            raise RuntimeError(
                f"DINOv3 returned {hidden.shape[1]} tokens for a {self.grid_size} patch grid"
            )
        return hidden[:, -self.num_patches :]


class VisualFeatureProjector(nn.Module):
    """Project DINOv3 patch tokens into the policy width."""

    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        # Keep the indexed module layout used by released single-DINO checkpoints.
        self.norms = nn.ModuleList([nn.LayerNorm(feature_dim, eps=1e-6)])
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, output_dim),
                    nn.SiLU(),
                    nn.Linear(output_dim, output_dim),
                )
            ]
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.projectors[0](self.norms[0](features))


class LanguageEncoder(nn.Module):
    """Frozen SigLIP text tower with pooled and token-level outputs."""

    MODEL_NAME = "google/siglip2-so400m-patch14-224"

    def __init__(
        self,
        output_dim: int,
        load_pretrained_weights: bool = True,
    ):
        super().__init__()
        from transformers import SiglipTextModel

        if load_pretrained_weights:
            self.encoder = SiglipTextModel.from_pretrained(self.MODEL_NAME)
        else:
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(
                self.MODEL_NAME, local_files_only=True
            )
            text_config = getattr(model_config, "text_config", None)
            if getattr(text_config, "model_type", None) != "siglip_text_model":
                raise RuntimeError(
                    f"{self.MODEL_NAME} no longer resolves to a SigLIP text configuration"
                )
            self.encoder = SiglipTextModel(text_config)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        token_dim = self.encoder.config.hidden_size
        self.token_projector = nn.Sequential(
            nn.LayerNorm(token_dim, eps=1e-6), nn.Linear(token_dim, output_dim)
        )
        self.pooled_projector = nn.Sequential(
            nn.LayerNorm(token_dim, eps=1e-6), nn.Linear(token_dim, output_dim)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self, tokens: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            output = self.encoder(input_ids=tokens, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        pooled = output.pooler_output
        return self.pooled_projector(pooled), self.token_projector(hidden)
