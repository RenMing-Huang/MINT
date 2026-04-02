from typing import Optional, Tuple, Union, Any
import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SDAT.optim.amp_opt import AmpOptimizer
from SDAT.utils.misc import MetricLogger

from SDAT.model import VQVAE, VectorQuantizer2

Ten = torch.Tensor

class VQVAETrainer(object):
    def __init__(
        self,
        vae_wo_ddp: VQVAE,
        vae: nn.Module,
        vae_opt: AmpOptimizer,
        ema_ratio: float,
        is_ema: bool,
        dct_loss_weight: float = 0.05,
    ):

        super(VQVAETrainer, self).__init__()

        ### models - vae
        self.vae, self.vae_opt = vae, vae_opt
        self.vae_wo_ddp: VQVAE = vae_wo_ddp
        self.vae_params: Tuple[nn.Parameter] = tuple(self.vae_wo_ddp.parameters())

        ### ema for vae
        self.ema_ratio = ema_ratio
        self.is_ema = is_ema
        if self.is_ema:
            self.vae_ema: VQVAE = deepcopy(vae_wo_ddp).eval()
        else:
            self.vae_ema: VQVAE = None

        ### params - vae
        self.w_l1=1.0
        self.w_l2=1.0 
        self.w_vq=1 
        self.w_dct = dct_loss_weight # set from config
        self.SN = len(getattr(self.vae, 'patch_nums', [])) if hasattr(self.vae, 'patch_nums') else 0
        


        self._dct_cache: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader, max_batches: int = 10) -> dict:
        """Run a lightweight eval over up to max_batches from ld_val.
        Returns averaged metrics: loss_rec_l1, loss_rec_l2, loss_vq, loss_vae.
        """
        self.vae.eval()
        n = 0
        sum_rec_l1 = 0.0
        sum_rec_l2 = 0.0
        sum_vq = 0.0
        sum_total = 0.0
        for i, obj in enumerate(ld_val):
            if i >= max_batches:
                break
            inp = obj['actions'] if isinstance(obj, dict) else obj
            if isinstance(inp, torch.Tensor) and inp.dim() == 2:
                # ensure shape [B, L, C] if dataset returns [B, L] for actions
                inp = inp.unsqueeze(-1)
            out = self.vae(inp=inp, ret_usages=False, ret_ms_l1=False)

            if isinstance(out, (list, tuple)) and len(out) >= 3:
                rec_inp, usages, loss_vq = out[0], out[1], out[2]
            else:
                # Fallback: some implementations may return dict
                rec_inp = out["rec_inp"] if isinstance(out, dict) else out
                usages = None
                loss_vq = torch.tensor(0.0, device=rec_inp.device)
            
            # Helper to check patchwise status
            is_patchwise = getattr(self.vae_wo_ddp, 'patchwise_cfg', {}).get('enable', False)
            
            if is_patchwise and rec_inp.shape[-1] == 8:
                # Patchwise with gripper classification
                l1_pos = F.l1_loss(rec_inp[..., :3], inp[..., :3])
                l1_rot = F.l1_loss(rec_inp[..., 3:6], inp[..., 3:6])
                # Gripper Classification
                rec_grip_logits = rec_inp[..., 6:8].reshape(-1, 2)
                inp_grip = (inp[..., 6] > 0).long().reshape(-1)
                l1_grip = F.cross_entropy(rec_grip_logits, inp_grip)
                
                loss_rec_l1 = (l1_pos + l1_rot) / 2.0  # Just continuous parts
                loss_rec_l2 = F.mse_loss(rec_inp[..., :6], inp[..., :6])
                
                # Combined metric for VQ
                loss_vae = (self.w_vq * loss_vq +
                            self.w_l1 * (l1_pos + l1_rot + l1_grip) + 
                            self.w_l2 * loss_rec_l2)
            else:
                loss_rec_l1 = F.l1_loss(input=rec_inp, target=inp)
                loss_rec_l2 = F.mse_loss(input=rec_inp, target=inp)
                loss_vae = (self.w_vq * loss_vq +
                            self.w_l1 * loss_rec_l1 +
                            self.w_l2 * loss_rec_l2)
            
            bsz = inp.shape[0]
            n += bsz
            sum_rec_l1 += float(loss_rec_l1.item()) * bsz
            sum_rec_l2 += float(loss_rec_l2.item()) * bsz
            sum_vq += float(loss_vq.item()) * bsz
            sum_total += float(loss_vae.item()) * bsz
        if n == 0:
            return {}
        return {
            'loss_rec_l1': sum_rec_l1 / n,
            'loss_rec_l2': sum_rec_l2 / n,
            'loss_vq': sum_vq / n,
            'loss_vae': sum_total / n,
        }

    def train_step(
        self,
        it: int,
        g_it: int,
        stepping: bool,
        me_lg: MetricLogger,
        inp: torch.Tensor,  # input batch (e.g., images) from VAEDataset
    ) -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float], Optional[dict]]:
        is_loggable = (it in me_lg.log_iters and it > 0)

        ### VAE
        # automatic mixed precision
        with self.vae_opt.amp_ctx:
            # ensure training mode
            self.vae.train(True)
            out = self.vae(inp=inp, ret_usages=is_loggable, ret_ms_l1=True)
            if isinstance(out, (list, tuple)) and len(out) == 4:
                rec_inp, usages, loss_vq, ms_l1_list = out
            else:
                rec_inp, usages, loss_vq = out
                ms_l1_list = None
            
            # Detect patchwise mode
            is_patchwise = hasattr(self.vae_wo_ddp, 'patchwise_cfg') and self.vae_wo_ddp.patchwise_cfg.get('enable', False)

            if is_patchwise and rec_inp.shape[-1] == 8:
                # We calculate specific losses later for optimization
                loss_rec_l1 = F.l1_loss(rec_inp[..., :6], inp[..., :6]) 
                loss_rec_l2 = F.mse_loss(rec_inp[..., :6], inp[..., :6])
            else:
                loss_rec_l1 = F.l1_loss(input=rec_inp, target=inp)
                loss_rec_l2 = F.mse_loss(input=rec_inp, target=inp)

            B, L, C = inp.shape
            
            # --- Multi-scale DCT Loss ---
            loss_ms_dct = torch.tensor(0.0, device=inp.device)
            
            if ms_l1_list is not None:
                # Prepare DCT matrix
                D = self._get_dct_ortho_matrix(L, inp.device, torch.float32)
                # Target DCT
                inp_perm = inp.transpose(1, 2).to(torch.float32) # (B, C, L)
                target_dct = torch.matmul(inp_perm, D.transpose(0, 1)) # (B, C, L)
                
                for rec_s in ms_l1_list:
                    # rec_s: (B, L, C) -> (B, C, L)
                    rec_s_perm = rec_s.transpose(1, 2).to(torch.float32)
                    rec_s_dct = torch.matmul(rec_s_perm, D.transpose(0, 1))
                    
                    # L1 loss in frequency domain
                    # We can optionally mask out some dims (like gripper)
                    # For now apply to all or match rec logic
                    if is_patchwise and rec_s.shape[-1] == 8:
                        # Only constrain continuous dims (0-6)
                        loss_ms_dct += F.mse_loss(rec_s_dct[:, :6, :], target_dct[:, :6, :])
                    else:
                        loss_ms_dct += F.mse_loss(rec_s_dct[:, :6, :], target_dct[:, :6, :])
                
                loss_ms_dct = loss_ms_dct / len(ms_l1_list)

            # Combine losses
            # Optional per-branch weights from config
            w_pos = w_rot = w_grip = 1.0
            if hasattr(self.vae_wo_ddp, 'patchwise_cfg') and self.vae_wo_ddp.patchwise_cfg.get('enable', False):
                pbw = self.vae_wo_ddp.patchwise_cfg.get('per_branch_weights', None)
                if isinstance(pbw, (list, tuple)) and len(pbw) == 3:
                    w_pos, w_rot, w_grip = float(pbw[0]), float(pbw[1]), float(pbw[2])

            # Branch-wise Loss
            if is_patchwise and rec_inp.shape[-1] == 8:
                l1_pos = F.l1_loss(rec_inp[..., :3], inp[..., :3])
                l1_rot = F.l1_loss(rec_inp[..., 3:6], inp[..., 3:6])
                
                # Gripper Classification Loss
                rec_grip_logits = rec_inp[..., 6:8].reshape(-1, 2)
                inp_grip = (inp[..., 6] > 0).long().reshape(-1)
                l1_grip = F.cross_entropy(rec_grip_logits, inp_grip)
                l1_rec = (w_pos * l1_pos + w_rot * l1_rot + w_grip * l1_grip)
            else:
                l1_rec = F.l1_loss(rec_inp, inp)

            loss_vae = (self.w_vq * loss_vq + 
                       self.w_l1 * l1_rec +
                       self.w_dct * loss_ms_dct)


        # Optimizer step when accumulation boundary reached
        if stepping:
            grad_norm, scale_log2 = self.vae_opt.backward_clip_step(loss=loss_vae, stepping=True)

        ### UPDATE EMA
        if stepping and self.is_ema:
            self.ema_update(g_it)

        ### LOG to metric
        if it in me_lg.log_iters and it > 0:
            me_lg.update(
                loss_rec_l1=loss_rec_l1.item(),
                loss_rec_l2=loss_rec_l2.item(),
                loss_vq=loss_vq.item(),
                loss_vae=loss_vae.item(),
                loss_ms_dct=loss_ms_dct.item()
                )
            
            # Branch-wise logging if patchwise enabled
            if hasattr(self.vae_wo_ddp, 'patchwise_cfg') and self.vae_wo_ddp.patchwise_cfg.get('enable', False):
                with torch.no_grad():
                    me_lg.update(l1_pos=l1_pos.item(), l1_rot=l1_rot.item(), l1_grip=l1_grip.item())
            
            if usages is not None:
                if isinstance(usages, (list, tuple)):
                    for i, usage in enumerate(usages):
                        val = usage.detach().item() if torch.is_tensor(usage) else float(usage)
                        me_lg.update(**{f'vae_usage_{i}': val})
                else: 
                    me_lg.update(vae_usage=float(usages.item()))
            # Disable ms-scale logging for stability


        log_dict = None
        if is_loggable:
            log_dict = {
                'loss_rec_l1': float(loss_rec_l1.item()),
                'loss_rec_l2': float(loss_rec_l2.item()),
                'loss_vq': float(loss_vq.item()),
                'loss_vae': float(loss_vae.item()),
            }
            
            # Branch-wise logging for wandb
            if hasattr(self.vae_wo_ddp, 'patchwise_cfg') and self.vae_wo_ddp.patchwise_cfg.get('enable', False):
                with torch.no_grad():
                    log_dict['l1_pos'] = float(l1_pos.item())
                    log_dict['l1_rot'] = float(l1_rot.item())
                    log_dict['l1_grip'] = float(l1_grip.item())
            
            if 'loss_ms_dct' in locals():
                log_dict['loss_ms_dct'] = float(loss_ms_dct.item())
            

            if usages is not None:
                if isinstance(usages, (list, tuple)):
                    for i, u in enumerate(usages):
                        val = u.detach().item() if torch.is_tensor(u) else float(u)
                        log_dict[f'vae_usage_{i}'] = val
                else:
                    log_dict['vae_usage'] = float(usages.item())


        gn = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else (float(grad_norm) if grad_norm is not None else None)
        sc = float(scale_log2) if isinstance(scale_log2, torch.Tensor) else (float(scale_log2) if scale_log2 is not None else None)

        return gn, sc, log_dict

    def _get_dct_ortho_matrix(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build and cache orthonormal DCT-II matrix of size (L, L).
        For norm='ortho', inverse is its transpose (IDCT-III).
        """
        key = (L, device, dtype)
        # Simple cache by L only; if device/dtype differs, rebuild
        mat = self._dct_cache.get(L, None)
        if mat is None or mat.device != device or mat.dtype != dtype:
            n = torch.arange(L, device=device, dtype=dtype)
            k = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # (L,1)
            # Cosine kernel: cos(pi/N * (n + 0.5) * k)
            cos_arg = (math.pi / L) * (n + 0.5)  # (L,)
            mat = torch.cos(k * cos_arg)  # (L, L)
            # Orthonormal scaling: sqrt(2/N) * alpha(k)
            scale = torch.ones(L, device=device, dtype=dtype)
            scale[0] = 1.0 / math.sqrt(2.0)
            mat = mat * (math.sqrt(2.0 / L) * scale).unsqueeze(1)  # scale per-row k
            self._dct_cache[L] = mat
        return self._dct_cache[L]


    def ema_update(self, g_it):
        """
        @func:
        ema update in order to get a more stable version
        """
        ## init
        ema_ratio = min(self.ema_ratio, (g_it//2 + 1) / (g_it//2 + 10))
        ## params
        for p_ema, p in zip(self.vae_ema.parameters(), self.vae_wo_ddp.parameters()):
            if p.requires_grad:
                p_ema.data.mul_(ema_ratio).add_(p.data, alpha=1-ema_ratio)
        ## buffer
        for p_ema, p in zip(self.vae_ema.buffers(), self.vae_wo_ddp.buffers()):
            p_ema.data.copy_(p.data)
        ## codebook
        quant, quant_ema = self.vae_wo_ddp.quantizer, self.vae_ema.quantizer
        quant: VectorQuantizer2
        if hasattr(quant, 'using_ema') and quant.using_ema:
            if hasattr(quant, 'using_restart') and quant.using_restart:
                quant_ema.embedding.weight.data.copy_(quant.embedding.weight.data)
            else:
                quant_ema.embedding.weight.data.mul_(ema_ratio).add_(quant.embedding.weight.data, alpha=1-ema_ratio)

    def get_config(self):
        """
        @func:
        get the loss and ema config of the model
        """
        cfg = {
            'ema_ratio': self.ema_ratio,
            'w_l1': self.w_l1,
            'w_l2': self.w_l2,
            'w_vq': self.w_vq,
        }
        # multiscale removed
        return cfg

    def state_dict(self):
        """
        @func:
        fetch the models needed to reserve
        """
        state = {'config': self.get_config()}
        for k in ('vae_wo_ddp', 'vae_ema', 'vae_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True):
        """
        @func:
        load the models needed
        """
        for k in ('vae_wo_ddp', 'vae_ema', 'vae_opt'):
            m = getattr(self, k)
            if m is not None:

                # model
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod

                # load_state_dict
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VAETr.load_state_dict] {k} missing:  {missing}')
                    print(f'[VAETr.load_state_dict] {k} unexpected:  {unexpected}')

        config: dict = state.pop('config', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAETr.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
