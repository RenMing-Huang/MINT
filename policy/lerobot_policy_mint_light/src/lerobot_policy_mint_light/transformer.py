"""Modern AdaLN transformer used by the MINT-Light action-token policy."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DropPath(nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = x.new_empty(shape).bernoulli_(keep_probability)
        return x * keep / keep_probability


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.dropout = dropout
        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.output = nn.Linear(self.inner_dim, dim)
        self.output_dropout = nn.Dropout(dropout)
        self._caching = False
        self.kv_cache: tuple[Tensor, Tensor] | None = None

    def kv_caching(self, enabled: bool) -> None:
        self._caching = enabled
        if not enabled:
            self.kv_cache = None

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if self._caching:
            if self.kv_cache is not None:
                old_k, old_v = self.kv_cache
                k = torch.cat([old_k, k], dim=2)
                v = torch.cat([old_v, v], dim=2)
            self.kv_cache = (k, v)

        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(1)
        hidden = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        hidden = hidden.transpose(1, 2).reshape(batch, length, self.inner_dim)
        return self.output_dropout(self.output(hidden))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.dropout = dropout
        self.query = nn.Linear(dim, self.inner_dim, bias=False)
        self.key_value = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.output = nn.Linear(self.inner_dim, dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        batch, query_length, _ = x.shape
        context_length = context.shape[1]
        q = self.query(x).reshape(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k, v = self.key_value(context).reshape(
            batch, context_length, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        attention_mask = None
        if context_mask is not None:
            attention_mask = context_mask[:, None, None, :].bool()
        hidden = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        hidden = hidden.transpose(1, 2).reshape(batch, query_length, self.inner_dim)
        return self.output_dropout(self.output(hidden))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(dim, hidden_dim * 2)
        self.output = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.dropout(self.output(value * F.silu(gate)))


class AdaLNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_hidden_dim: int,
        attention_dropout: float,
        ffn_dropout: float,
        drop_path: float,
        use_cross_attention: bool,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.self_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attention = SelfAttention(dim, num_heads, head_dim, attention_dropout)
        if use_cross_attention:
            self.cross_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.cross_attention = CrossAttention(dim, num_heads, head_dim, attention_dropout)
        else:
            self.cross_norm = None
            self.cross_attention = None
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = SwiGLU(dim, mlp_hidden_dim, ffn_dropout)
        self.drop_path = DropPath(drop_path)
        modulation_count = 9 if use_cross_attention else 6
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, modulation_count * dim))

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale) + shift

    def reset_adaln_zero(self) -> None:
        modulation = self.modulation[-1]
        nn.init.zeros_(modulation.weight)
        nn.init.zeros_(modulation.bias)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        mask: Tensor | None,
        cross_context: Tensor | None,
        cross_mask: Tensor | None,
    ) -> Tensor:
        modulation = self.modulation(cond).unsqueeze(1)
        if self.use_cross_attention:
            (
                shift_self,
                scale_self,
                gate_self,
                shift_cross,
                scale_cross,
                gate_cross,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = modulation.chunk(9, dim=-1)
        else:
            shift_self, scale_self, gate_self, shift_ffn, scale_ffn, gate_ffn = (
                modulation.chunk(6, dim=-1)
            )

        hidden = self._modulate(self.self_norm(x), shift_self, scale_self)
        x = x + self.drop_path(self.self_attention(hidden, mask)) * gate_self
        if self.use_cross_attention:
            if cross_context is None:
                raise ValueError("Token-level language context is required for cross-attention")
            hidden = self._modulate(self.cross_norm(x), shift_cross, scale_cross)
            x = x + self.drop_path(
                self.cross_attention(hidden, cross_context, cross_mask)
            ) * gate_cross
        hidden = self._modulate(self.ffn_norm(x), shift_ffn, scale_ffn)
        return x + self.drop_path(self.ffn(hidden)) * gate_ffn


class AdaLNTransformer(nn.Module):
    """AdaLN-Zero transformer with SDPA, SwiGLU, and token-level language fusion."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        mlp_hidden_dim: int,
        attention_dropout: float,
        ffn_dropout: float,
        drop_path_rate: float,
        cross_attention_every: int,
    ):
        super().__init__()
        if cross_attention_every <= 0:
            raise ValueError("cross_attention_every must be positive")
        drop_path_rates = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.layers = nn.ModuleList(
            [
                AdaLNBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_hidden_dim=mlp_hidden_dim,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    drop_path=drop_path_rates[layer],
                    use_cross_attention=(layer + 1) % cross_attention_every == 0,
                )
                for layer in range(depth)
            ]
        )

    def reset_adaln_zero(self) -> None:
        for layer in self.layers:
            layer.reset_adaln_zero()

    def kv_caching(self, enabled: bool) -> None:
        for layer in self.layers:
            layer.self_attention.kv_caching(enabled)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        mask: Tensor | None = None,
        cross_context: Tensor | None = None,
        cross_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, cond, mask, cross_context, cross_mask)
        return x
