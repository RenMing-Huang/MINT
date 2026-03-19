import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.models as models

from safetensors.torch import load_file as load_safetensors_file
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

import numpy as np
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any
from collections import deque
import einops
from einops.layers.torch import Rearrange

# ========= quantinizer begin =========
class VectorQuantizer2(nn.Module):
    def __init__(
        self, codebook_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, patch_nums=None, quant_resi=0.5, share_quant_resi=4,
        upsample_mode='bilinear', downsample_mode='area',
    ):
        super().__init__()
        self.codebook_size: int = codebook_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.patch_nums: Tuple[int] = patch_nums
        self.upsample_mode = upsample_mode
        self.downsample_mode = downsample_mode

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))

        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.patch_nums), self.codebook_size), fill_value=0.0))
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.codebook_size, self.Cvae)

        self.prog_si = -1

    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.codebook_size, abs(eini) / self.codebook_size)

    def extra_repr(self) -> str:
        return f'{self.patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.patch_nums)}, quant_resi={self.quant_resi_ratio}'

    # ===================== `forward` is only used in VAE training =====================
    def forward(
        self,
        f_BCH: torch.Tensor,
        ret_usages: bool = False,
        ret_fhat_scales: bool = False,
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor] | Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]]:
        dtype = f_BCH.dtype
        if dtype != torch.float32: f_BCH = f_BCH.float()
        B, C, H = f_BCH.shape
        f_no_grad = f_BCH.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        f_hat_scales: List[torch.Tensor] = []


        mean_vq_loss: torch.Tensor = 0.0
        vocab_hit_V = torch.zeros(self.codebook_size, dtype=torch.float, device=f_BCH.device)
        SN = len(self.patch_nums)
        for si, pn in enumerate(self.patch_nums): # from small to large
            # find the nearest embedding
            if self.using_znorm:
                rest_NC = F.interpolate(f_rest, size=pn, mode=self.downsample_mode).permute(0, 2, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 1).reshape(-1, C)
                rest_NC = F.normalize(rest_NC, dim=-1)
                idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                rest_NC = F.interpolate(f_rest, size=pn, mode=self.downsample_mode).permute(0, 2, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 1).reshape(-1, C)
                d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)
            hit_V = idx_N.bincount(minlength=self.codebook_size).float()

            # calc loss
            idx_BH = idx_N.view(B, pn)
            h_BCH = F.interpolate(self.embedding(idx_BH).permute(0, 2, 1), size=H, mode=self.upsample_mode).contiguous() if (si != SN-1) else self.embedding(idx_BH).permute(0, 2, 1).contiguous()
            h_BCH = self.quant_resi[si/(SN-1)](h_BCH)

            f_hat = f_hat + h_BCH
            f_rest = f_rest - h_BCH

            if ret_fhat_scales:
                f_hat_diff = f_hat + (f_BCH - f_no_grad)
                f_hat_scales.append(f_hat_diff)


            if self.training:
                if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                self.record_hit += 1
            vocab_hit_V.add_(hit_V)
            
            curr_vq_loss = F.mse_loss(f_hat.detach(), f_BCH).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            mean_vq_loss += curr_vq_loss

        mean_vq_loss *= 1. / SN
        f_hat = f_hat.detach() + (f_BCH - f_no_grad)
        margin = (f_BCH.numel() / f_BCH.shape[1]) / self.codebook_size * 0.08
        if ret_usages:
            usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.patch_nums)]
        else:
            usages = None
        if ret_fhat_scales:
            return f_hat, usages, mean_vq_loss, f_hat_scales
        return f_hat, usages, mean_vq_loss

    def embed_to_fhat(self, ms_h_BCH: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BCH = []
        B = ms_h_BCH[0].shape[0]
        H = self.patch_nums[-1]
        SN = len(self.patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BCH[0].new_zeros(B, self.Cvae, H, dtype=torch.float32)
            for si, pn in enumerate(self.patch_nums):
                h_BCH = ms_h_BCH[si]
                if si < len(self.patch_nums) - 1:
                    h_BCH = F.interpolate(h_BCH, size=H, mode=self.upsample_mode)
                h_BCH = self.quant_resi[si/(SN-1)](h_BCH)
                f_hat.add_(h_BCH)
                if last_one: ls_f_hat_BCH = f_hat
                else: ls_f_hat_BCH.append(f_hat.clone())
        else:
            f_hat = ms_h_BCH[0].new_zeros(B, self.Cvae, self.patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.patch_nums):
                f_hat = F.interpolate(f_hat, size=pn, mode=self.downsample_mode)
                h_BCH = self.quant_resi[si/(SN-1)](ms_h_BCH[si])
                f_hat.add_(h_BCH)
                if last_one: ls_f_hat_BCH = f_hat
                else: ls_f_hat_BCH.append(f_hat)

        return ls_f_hat_BCH

    def f_to_idxBl_or_fhat(self, f_BCH: torch.Tensor, to_fhat: bool, patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:
        B, C, H = f_BCH.shape
        f_no_grad = f_BCH.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_hs = [pn for pn in (patch_nums or self.patch_nums)]
        assert patch_hs[-1] == H, f'{patch_hs[-1]=} != ({H=})'

        SN = len(patch_hs)
        for si, ph in enumerate(patch_hs):
            if 0 <= self.prog_si < si: break
            z_NC = F.interpolate(f_rest, size=ph, mode=self.downsample_mode).permute(0, 2, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)

            idx_BH = idx_N.view(B, ph)
            h_BCH = F.interpolate(self.embedding(idx_BH).permute(0, 2, 1), size=H, mode=self.upsample_mode).contiguous() if (si != SN-1) else self.embedding(idx_BH).permute(0, 2, 1).contiguous()
            h_BCH = self.quant_resi[si/(SN-1)](h_BCH)
            f_hat.add_(h_BCH)
            f_rest.sub_(h_BCH)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph))

        return f_hat_or_idx_Bl

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BCH: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        H = self.patch_nums[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BCH, size=H, mode=self.upsample_mode))
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=self.patch_nums[si+1], mode=self.downsample_mode)
        else:
            h = self.quant_resi[si/(SN-1)](h_BCH)
            f_hat.add_(h)
            return f_hat, f_hat

    def idxBl_to_next_scale_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales_inputs = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H_max = self.patch_nums[-1]
        SN = len(self.patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H_max, dtype=torch.float32)
        pn_next = self.patch_nums[0]
        for si in range(SN - 1): 
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break
            h_BCH = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next), size=H_max, mode=self.upsample_mode)
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BCH))
            pn_next = self.patch_nums[si+1]
            next_scales_inputs.append(F.interpolate(f_hat, size=pn_next, mode=self.downsample_mode).view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales_inputs, dim=1) if len(next_scales_inputs) else None


class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BCH):
        return h_BCH.mul(1-self.resi_ratio) + super().forward(h_BCH) * self.resi_ratio


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
# ========= quantinizer end =========

# ========= blocks begin ============
class PatchwiseEmbedding1D(nn.Module):
    """Split 7-dim action into 3 modality branches and embed each to D_embed dimensions.
    
    Input: (B, T, 7) where 7 = [pos(3), rot(3), grip(1)]
    Output: (B, 3*D_embed, T) ready for grouped Conv1d
    """
    def __init__(self, d_embed: int = 64, dropout: float = 0.0, norm_type: str = 'layer'):
        super().__init__()
        self.d_embed = d_embed
        
        # Per-branch MLPs
        self.pos_embed = nn.Sequential(
            nn.Linear(3, d_embed),
            nn.GELU(),
            nn.LayerNorm(d_embed) if norm_type == 'layer' else nn.Identity(),
            nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed, d_embed),
        )
        self.rot_embed = nn.Sequential(
            nn.Linear(3, d_embed),
            nn.GELU(),
            nn.LayerNorm(d_embed) if norm_type == 'layer' else nn.Identity(),
            nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed, d_embed),
        )
        # Change to Embedding for gripper classification
        self.grip_embed = nn.Embedding(2, d_embed)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        """x: (B, T, 7)"""


        x_pos = x[..., :3]    # (B, T, 3)
        x_rot = x[..., 3:6]   # (B, T, 3)
        x_grip = x[..., 6]    # (B, T)
        
        e_pos = self.pos_embed(x_pos)    # (B, T, D)

        e_rot = self.rot_embed(x_rot)    # (B, T, D)

        
        # Gripper: Discretize > 0.5 -> 1, else 0
        x_grip_int = (x_grip > 0).long()
        e_grip = self.grip_embed(x_grip_int) # (B, T, D)

        
        # Stack and transpose to (B, 3*D, T)
        e_all = torch.cat([e_pos, e_rot, e_grip], dim=-1)  # (B, T, 3*D)
        return e_all.transpose(1, 2)  # (B, 3*D, T)

class PatchwiseProjection1D(nn.Module):
    """Mirror of PatchwiseEmbedding - project 3*D_embed back to 7-dim action.
    
    Input: (B, 3*D_embed, T) from decoder
    Output: (B, T, 8) -> [pos(3), rot(3), grip_logits(2)]
    """
    def __init__(self, d_embed: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_embed = d_embed
        
        # Per-branch projection heads
        self.pos_head = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed, 3),
        )
        self.rot_head = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed, 3),
        )
        self.grip_head = nn.Sequential(
            nn.Linear(d_embed, d_embed // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed // 2, 2), # Output logits for 2 classes
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, z):
        """z: (B, 3*D, T)"""
        z = z.transpose(1, 2)  # (B, T, 3*D)
        D = self.d_embed
        
        z_pos = z[..., :D]
        z_rot = z[..., D:2*D]
        z_grip = z[..., 2*D:]
        
        out_pos = self.pos_head(z_pos)    # (B, T, 3)
        out_rot = self.rot_head(z_rot)    # (B, T, 3)
        out_grip = self.grip_head(z_grip) # (B, T, 2) - Logits
        
        return torch.cat([out_pos, out_rot, out_grip], dim=-1)  # (B, T, 8)

def nonlinearity(x):
    return x * torch.sigmoid(x)

def Normalize(in_dims, num_groups=8):
    if num_groups > in_dims:
        num_groups = 1
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_dims, eps=1e-6, affine=True)

class Upsample1D_2x(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_dims, in_dims, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Downsample1D_2x(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_dims, in_dims, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_dims, out_channels=None, dropout):
        super().__init__()
        self.in_dims = in_dims
        out_channels = in_dims if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_dims)
        self.conv1 = torch.nn.Conv1d(in_dims, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_dims != self.out_channels:
            self.nin_shortcut = torch.nn.Conv1d(in_dims, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h

class GroupedResnetBlock(nn.Module):
    """ResnetBlock with grouped convolutions for branch isolation."""
    def __init__(self, *, in_dims, out_channels=None, dropout, groups=3):
        super().__init__()
        self.in_dims = in_dims
        out_channels = in_dims if out_channels is None else out_channels
        self.out_channels = out_channels
        self.groups = groups

        self.norm1 = Normalize(in_dims)
        self.conv1 = torch.nn.Conv1d(in_dims, out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        if self.in_dims != self.out_channels:
            self.nin_shortcut = torch.nn.Conv1d(in_dims, out_channels, kernel_size=1, stride=1, padding=0, groups=groups)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.C = in_dims
        self.norm = Normalize(in_dims)
        self.qkv = torch.nn.Conv1d(in_dims, 3 * in_dims, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_dims) ** (-0.5)
        self.proj_out = torch.nn.Conv1d(in_dims, in_dims, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=1), qkv)
        q = q * self.w_ratio
        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return x + self.proj_out(out)

def make_attn(in_dims, using_sa=True):
    return AttnBlock(in_dims) if using_sa else nn.Identity()

class Encoder(nn.Module):
    def __init__(
        self, *, ch=32, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_dims=3, z_channels, double_z=False,
        using_sa=True, using_mid_sa=True,
        patchwise_cfg: dict | None = None
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_dims = in_dims
        self.patchwise_cfg = patchwise_cfg or {'enable': False}
        
        # Grouped conv setup
        self.use_grouped = self.patchwise_cfg.get('enable', False)
        self.grouped_depth = self.patchwise_cfg.get('grouped_depth', 2) if self.use_grouped else 0
        
        # conv_in: if grouped, input is 3*D_embed; else original in_dims
        actual_in_dims = in_dims if not self.use_grouped else (3 * self.patchwise_cfg.get('d_embed', 64))
        self.conv_in = torch.nn.Conv1d(
            actual_in_dims, self.ch, 
            kernel_size=3, stride=1, padding=1,
            groups=3 if self.use_grouped else 1
        )

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                # Use grouped conv for early layers if patchwise enabled
                use_groups_here = self.use_grouped and i_level < self.grouped_depth
                if use_groups_here:
                    # Custom grouped ResnetBlock - simplified, just group the convs
                    block.append(GroupedResnetBlock(
                        in_dims=block_in, out_channels=block_out, 
                        dropout=dropout, groups=3
                    ))
                else:
                    block.append(ResnetBlock(in_dims=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample1D_2x(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_dims=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_dims=block_in, out_channels=block_in, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # If patchwise, x is already (B, 3*D, T); else (B, T, C) -> (B, C, T)
        if not self.use_grouped:
            x = einops.rearrange(x, 'b h d -> b d h')

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):

                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        return h

class Decoder(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_dims=3, z_channels,
        using_sa=True, using_mid_sa=True,
        patchwise_cfg: dict | None = None
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_dims = in_dims
        self.patchwise_cfg = patchwise_cfg or {'enable': False}

        # Mirror encoder: grouped conv for early up-blocks
        self.use_grouped = self.patchwise_cfg.get('enable', False)
        self.grouped_depth = self.patchwise_cfg.get('grouped_depth', 2) if self.use_grouped else 0

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = torch.nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_dims=block_in, out_channels=block_in, dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(
            in_dims=block_in, out_channels=block_in, dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                # Mirror encoder: use grouped conv for LATE layers (i_level < grouped_depth)
                use_groups_here = self.use_grouped and i_level < self.grouped_depth
                if use_groups_here:
                    block.append(GroupedResnetBlock(
                        in_dims=block_in, out_channels=block_out, 
                        dropout=dropout, groups=3
                    ))
                else:
                    block.append(ResnetBlock(in_dims=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions-1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample1D_2x(block_in)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        # conv_out: if grouped, output 3*D_embed; else in_dims
        actual_out_dims = in_dims if not self.use_grouped else (3 * self.patchwise_cfg.get('d_embed', 64))
        self.conv_out = torch.nn.Conv1d(
            block_in, actual_out_dims, 
            kernel_size=3, stride=1, padding=1,
            groups=3 if self.use_grouped else 1
        )

    def forward(self, z):
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        # If patchwise, return (B, 3*D, T); else (B, T, C)
        if not self.use_grouped:
            h = einops.rearrange(h, 'b d h -> b h d')
        return h
# =========== end blocks ============

class MultiScaleVQVAE(nn.Module):
    def __init__(
        self,
        seq_dim,
        codebook_size=512,
        codebook_dim=16,
        ch=32,
        patch_nums=(1, 2, 4),
        ch_mult=(2, 4, 8),
        dropout=0.0,
        beta=0.25,
        using_znorm=True,
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=0,
        default_qresi_counts=0,
        upsample_mode='linear',
        downsample_mode='area',
        patchwise: dict | None = None,

    ):
        super().__init__()
        self.V, self.Cvae = codebook_size, codebook_dim
        self.patch_nums = patch_nums
        self.patchwise_cfg = patchwise or {'enable': False}

        ddconfig = dict(
            dropout=dropout,
            ch=ch,
            z_channels=codebook_dim,
            in_dims=seq_dim,
            ch_mult=ch_mult,
            num_res_blocks=2,
            using_sa=True,
            using_mid_sa=True,
            patchwise_cfg=self.patchwise_cfg,
        )

        # Patchwise embedding (if enabled)
        if self.patchwise_cfg.get('enable', False):
            self.patchwise_embed = PatchwiseEmbedding1D(
                d_embed=self.patchwise_cfg.get('d_embed'),
                dropout=dropout,
                norm_type=self.patchwise_cfg.get('norm', 'layer'),
            )
            self.patchwise_proj = PatchwiseProjection1D(
                d_embed=self.patchwise_cfg.get('d_embed'),
                dropout=dropout,
            )
        else:
            self.patchwise_embed = None
            self.patchwise_proj = None

        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.codebook_size = codebook_size
        self.quantizer = VectorQuantizer2(
            codebook_size=codebook_size,
            Cvae=self.Cvae,
            using_znorm=using_znorm,
            beta=beta,
            default_qresi_counts=default_qresi_counts,
            patch_nums=patch_nums,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
            upsample_mode=upsample_mode,
            downsample_mode=downsample_mode,
        )
        self.quant_conv = nn.Conv1d(self.Cvae, self.Cvae, quant_conv_ks, padding=quant_conv_ks // 2)
        self.post_quant_conv = nn.Conv1d(self.Cvae, self.Cvae, quant_conv_ks, padding=quant_conv_ks // 2)


    def forward(self, inp, ret_usages: bool = False, ret_ms_l1: bool = False):
        # Patchwise embedding if enabled
        if self.patchwise_embed is not None:
            inp = self.patchwise_embed(inp)  # (B, T, 7) -> (B, 3*D, T)

        f = self.quant_conv(self.encoder(inp))
        SN = len(self.patch_nums)

        q_out = self.quantizer(
            f,
            ret_usages=ret_usages,
            ret_fhat_scales=ret_ms_l1,
        )
        if ret_ms_l1:
            # quantizer returns 4-tuple when ret_fhat_scales=True
            f_hat, usages, vq_loss, fhat_scales = q_out
        else:
            f_hat, usages, vq_loss = q_out

        rec = self.decoder(self.post_quant_conv(f_hat))
        
        # Patchwise projection if enabled
        if self.patchwise_proj is not None:
            rec = self.patchwise_proj(rec)  # (B, 3*D, T) -> (B, T, 8)
        
        if ret_ms_l1:
            rec_scales: List[torch.Tensor] = []
            for fh in fhat_scales:
                rec_s = self.decoder(self.post_quant_conv(fh))
                if self.patchwise_proj is not None:
                    rec_s = self.patchwise_proj(rec_s)
                rec_scales.append(rec_s)
            return rec, usages, vq_loss, rec_scales
        return rec, usages, vq_loss

    def inp_to_idxBl(self, inp_seq_no_grad: torch.Tensor, patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:
        # Patchwise embedding if enabled
        if self.patchwise_embed is not None:
            inp_seq_no_grad = self.patchwise_embed(inp_seq_no_grad)
        f = self.quant_conv(self.encoder(inp_seq_no_grad))
        return self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=False, patch_nums=patch_nums)

    def load_vqvae_weights(self, load_path: str):
        """Load VQVAE weights from resume_path or huggingface hub."""
        if os.path.isdir(load_path):
            pt_files = []
            for root, _, files in os.walk(load_path):
                for file_name in files:
                    if file_name.endswith(".pth"):
                        pt_files.append(os.path.join(root, file_name))

            if len(pt_files) == 0:
                logging.warning(f"VQVAE load path is a directory but no .pth file was found: {load_path}")
                return

            if len(pt_files) > 1:
                pt_files.sort(key=os.path.getmtime, reverse=True)
                logging.warning(
                    f"Multiple .pth files found in {load_path}. "
                    f"Using the latest modified file: {pt_files[0]}"
                )
            load_path = pt_files[0]
        elif not os.path.exists(load_path):
            try:
                load_path = hf_hub_download(
                    repo_id=load_path,
                    filename=SAFETENSORS_SINGLE_FILE,
                )
            except Exception:
                pass
        
        if not os.path.exists(load_path):
             logging.warning(f"VQVAE load path provided but file not found locally or on HF: {load_path}")
             return

        print(f"Loading VQVAE weights from {load_path}")
        
        if load_path.endswith(".safetensors"):
            state = load_safetensors_file(load_path)
        else:
            state = torch.load(load_path, map_location="cpu")
            
        sd = None
        if isinstance(state, dict) and "trainer" in state and isinstance(state["trainer"], dict):
            sd = state["trainer"].get("vae_wo_ddp", None)
        if sd is None and isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        if sd is None and isinstance(state, dict):
            sd = state
            
        target_keys = set(self.state_dict().keys())
        
        def strip_prefix(key, prefix):
            return key[len(prefix):] if key.startswith(prefix) else key
            
        new_sd = {}
        for k, v in sd.items():
            new_k = k
            if k.startswith("model.multi_scale_vqvae."):
                new_k = strip_prefix(k, "model.multi_scale_vqvae.")
            elif k.startswith("multi_scale_vqvae."):
                new_k = strip_prefix(k, "multi_scale_vqvae.")
            elif k.startswith("model."):
                 new_k = strip_prefix(k, "model.")
            
            if new_k in target_keys:
                new_sd[new_k] = v
            elif k in target_keys:
                new_sd[k] = v
                
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        
        if len(missing) > 0:
            logging.warning(f"Missing keys when loading VQVAE: {missing} ...")
        if len(unexpected) > 0:
             logging.warning(f"Unexpected keys when loading VQVAE: {unexpected} ...")
        print(f"Loaded VQVAE weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # multi-scale decode utilities removed for simplicity

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantizer.ema_vocab_hit_SV' in state_dict and state_dict['quantizer.ema_vocab_hit_SV'].shape[0] != self.quantizer.ema_vocab_hit_SV.shape[0]:
            state_dict['quantizer.ema_vocab_hit_SV'] = self.quantizer.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)


# ==== similarity ensemble ======

class TemporalEnsembler:
    def __init__(self, horizon: int, n_action_steps: int, ensemble_coef: float = 0.5):
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.ensemble_coef = ensemble_coef
        self.chunk_queue = deque()
        self.global_step = 0

    def reset(self):
        self.chunk_queue.clear()
        self.global_step = 0

    def add_chunk(self, chunk: torch.Tensor, intention=None, **kwargs):
        self.chunk_queue.append((chunk, self.global_step))
        while self.chunk_queue and self.chunk_queue[0][1] + self.horizon <= self.global_step:
            self.chunk_queue.popleft()

    def get_ensembled_actions(self) -> torch.Tensor:
        batch_size, _, action_dim = self.chunk_queue[0][0].shape
        device = self.chunk_queue[0][0].device
        ensembled_chunk = torch.zeros((batch_size, self.n_action_steps, action_dim), device=device)

        for i in range(self.n_action_steps):
            t_curr = self.global_step + i
            combined_act = 0.0
            weight_sum = 1e-8
            
            for chunk, start_time in self.chunk_queue:
                k = t_curr - start_time
                if 0 <= k < self.horizon:
                    weight = np.exp(-self.ensemble_coef * k)
                    combined_act += chunk[:, k] * weight
                    weight_sum += weight
                    
            ensembled_chunk[:, i] = combined_act / weight_sum
        
        self.global_step += self.n_action_steps
        return ensembled_chunk


class IntentionEnsembler:
    def __init__(self, horizon: int, n_action_steps: int, temperature: float = 0.05):
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.temperature = temperature
        self.chunk_queue = deque()
        self.global_step = 0

    def reset(self):
        self.chunk_queue.clear()
        self.global_step = 0

    def add_chunk(self, chunk: torch.Tensor, intention_vector: torch.Tensor):
        self.chunk_queue.append({
            "chunk": chunk.detach(),
            "intent": F.normalize(intention_vector.detach(), dim=-1),
            "start_time": self.global_step
        })
        while self.chunk_queue and self.chunk_queue[0]["start_time"] + self.horizon <= self.global_step:
            self.chunk_queue.popleft()

    def get_ensembled_actions(self) -> torch.Tensor:
        if not self.chunk_queue:
            return None
        
        ref_intent = self.chunk_queue[-1]["intent"]
        batch_size, _, action_dim = self.chunk_queue[0]["chunk"].shape
        ensembled_out = torch.zeros((batch_size, self.n_action_steps, action_dim), device=ref_intent.device)
        
        for i in range(self.n_action_steps):
            t_curr = self.global_step + i
            combined_act = 0.0
            total_weight = 1e-8
            
            for item in self.chunk_queue:
                k = t_curr - item["start_time"]
                if 0 <= k < item["chunk"].shape[1]:
                    similarity = torch.sum(ref_intent * item["intent"], dim=-1, keepdim=True)
                    weight = torch.exp(similarity / self.temperature)
                    combined_act += item["chunk"][:, k] * weight
                    total_weight += weight
                    
            ensembled_out[:, i] = combined_act / total_weight
        
        self.global_step += self.n_action_steps
        return ensembled_out