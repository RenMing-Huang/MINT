import torch.distributed as dist
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]
class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, codebook_size, Cvae, znorm, beta: float = 0.25,
        default_qresi_counts=0, patch_nums=None, quant_resi=0.5, share_quant_resi=4,
        upsample_mode='bilinear', downsample_mode='area',
    ):
        super().__init__()
        self.codebook_size: int = codebook_size
        self.Cvae: int = Cvae
        self.znorm: bool = znorm
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
        return f'{self.patch_nums}, znorm={self.znorm}, beta={self.beta}  |  S={len(self.patch_nums)}, quant_resi={self.quant_resi_ratio}'

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

        with torch.amp.autocast('cuda', enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.codebook_size, dtype=torch.float, device=f_BCH.device)
            SN = len(self.patch_nums)
            for si, pn in enumerate(self.patch_nums): # from small to large
                # find the nearest embedding
                if self.znorm:
                    rest_NC = F.interpolate(f_rest, size=pn, mode=self.downsample_mode).permute(0, 2, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=pn, mode=self.downsample_mode).permute(0, 2, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                hit_V = idx_N.bincount(minlength=self.codebook_size).float()

                if self.training and tdist.is_available() and tdist.is_initialized():
                    try:
                        tdist.all_reduce(hit_V, op=tdist.ReduceOp.SUM)
                    except Exception:
                        pass
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
            # Apply straight-through estimator to final only
            f_hat = f_hat.detach() + (f_BCH - f_no_grad)
            

        margin = (f_BCH.numel() / f_BCH.shape[1]) / self.codebook_size * 0.08
        if ret_usages:
            usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.patch_nums)]
        else:
            usages = None
        if ret_fhat_scales:
            return f_hat, usages, mean_vq_loss, f_hat_scales
        return f_hat, usages, mean_vq_loss

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
            if self.znorm:
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
