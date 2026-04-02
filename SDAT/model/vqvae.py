from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_vae import Decoder, Encoder, PatchwiseEmbedding1D, PatchwiseProjection1D
from .quant import VectorQuantizer2
class VQVAE(nn.Module):
    def __init__(
        self,
        seq_dim,
        codebook_size=4096,
        z_channels=32,
        ch=128,
        dropout=0.0,
        beta=0.25,
        znorm=False,
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=0,
        default_qresi_counts=0,
        patch_nums=(4, 8, 16, 32),
        ch_mult=(1, 2, 2, 4),
        upsample_mode='linear',
        downsample_mode='area',
        test_mode=True,
        # NEW: patchwise config
        patchwise: dict | None = None,

    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = codebook_size, z_channels
        self.patch_nums = patch_nums
        # store training configs on the module for trainer access

        self.patchwise_cfg = patchwise or {'enable': False}


        ddconfig = dict(
            dropout=dropout,
            ch=ch,
            z_channels=z_channels,
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
                d_embed=self.patchwise_cfg.get('d_embed', 64),
                dropout=dropout,
                norm_type=self.patchwise_cfg.get('norm', 'layer'),
            )
            self.patchwise_proj = PatchwiseProjection1D(
                d_embed=self.patchwise_cfg.get('d_embed', 64),
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
            znorm=znorm,
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

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantizer.ema_vocab_hit_SV' in state_dict and state_dict['quantizer.ema_vocab_hit_SV'].shape[0] != self.quantizer.ema_vocab_hit_SV.shape[0]:
            state_dict['quantizer.ema_vocab_hit_SV'] = self.quantizer.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
