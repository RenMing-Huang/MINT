import gc
import os
import time
from datetime import datetime
from functools import partial
from typing import Callable, Dict

import lightning
import torch
torch.set_float32_matmul_precision('medium')
import torch.distributed as tdist
import wandb
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from SDAT.trainer import VQVAETrainer
from SDAT.model import build_vqvae_model
from SDAT.utils.normalizer import build_normalizer_from_stats
from SDAT.optim.amp_opt import AmpOptimizer
from SDAT.optim.lr_control import filter_params, lr_wd_annealing
from SDAT.utils import misc


def build_runtime_vqvae_config(cfg: DictConfig) -> DictConfig:
    """Convert flat lerobot training cfg to runtime cfg consumed by policy."""
    runtime_cfg = {
        "horizon": int(cfg.get("horizon", 16)),
        "patch_size": int(cfg.get("patch_size", 1)),
        "model_cfg": {
            "codebook_size": int(cfg.get("codebook_size", 512)),
            "codebook_dim": int(cfg.get("codebook_dim", 32)),
            "ch": int(cfg.get("ch", 48)),
            "ch_mult": list(cfg.get("ch_mult", [2, 4, 8])),
            "action_dim": int(cfg.get("action_dim", 7)),
            "dropout": float(cfg.get("dropout", 0.1)),
            "patchwise": cfg.get("patchwise", None),
        },
        "quant_cfg": {
            "beta": float(cfg.get("beta", 0.25)),
            "znorm": bool(cfg.get("znorm", True)),
            "quant_conv_ks": int(cfg.get("quant_conv_ks", 3)),
            "quant_resi": float(cfg.get("quant_resi", 0.5)),
            "share_quant_resi": int(cfg.get("share_quant_resi", 0)),
            "patch_nums": list(cfg.get("patch_nums", [1, 2, 4])),
            "vae_init": float(cfg.get("vae_init", -0.1)),
            "vocab_init": float(cfg.get("vocab_init", -1.0)),
            "codebook_reset": cfg.get("codebook_reset", None),
        },
    }
    return OmegaConf.create(runtime_cfg)


def is_runtime_vqvae_config(cfg: DictConfig) -> bool:
    return bool("model_cfg" in cfg and "quant_cfg" in cfg)


def save_vqvae_configs(work_dir: str, train_cfg: DictConfig) -> None:
    """Save both training cfg and runtime cfg automatically.

    - train_config.yaml: full training config for reproduction
    - config.yaml: runtime config for policy-side VQVAE loading
    """
    os.makedirs(work_dir, exist_ok=True)
    OmegaConf.save(config=train_cfg, f=os.path.join(work_dir, "train_config.yaml"))

    runtime_cfg = train_cfg if is_runtime_vqvae_config(train_cfg) else build_runtime_vqvae_config(train_cfg)
    OmegaConf.save(config=runtime_cfg, f=os.path.join(work_dir, "config.yaml"))


def build_experiment_suffix(cfg: DictConfig) -> str:
    return "_".join(
        [
            f"h_{int(cfg.get('horizon', 16))}",
            f"bs_{int(cfg.get('batch_size', 256))}",
            f"cbs_{int(cfg.get('codebook_size', 512))}",
            f"cbd_{int(cfg.get('codebook_dim', 32))}",
            f"ch_{int(cfg.get('ch', 48))}",
        ]
    )


def resolve_shared_work_dir(fabric: Fabric, cfg: DictConfig) -> str:
    work_dir = ""
    if fabric.is_global_zero:
        current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        work_dir = os.path.join("outputs/sdat", build_experiment_suffix(cfg), current_date)
        os.makedirs(work_dir, exist_ok=True)

    if tdist.is_available() and tdist.is_initialized():
        payload = [work_dir]
        tdist.broadcast_object_list(payload, src=0)
        work_dir = str(payload[0])

    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def build_action_preprocess_fn(dataset: LeRobotDataset, cfg: DictConfig, horizon: int) -> Callable:
    mode = str(cfg.get("action_norm_mode", "quantile")).lower().strip()

    normalizer = None
    if mode != "none":
        stats = getattr(getattr(dataset, "meta", None), "stats", None)
        action_stats = stats.get("action", None) if isinstance(stats, dict) else None
        if not isinstance(action_stats, dict):
            action_stats = stats.get("actions", None) if isinstance(stats, dict) else None

        if isinstance(action_stats, dict):
            if "count" in action_stats:
                count_v = torch.as_tensor(action_stats["count"]).reshape(-1)[0].item()
            else:
                count_v = len(dataset)
            norm_stats = {
                "action_dim": int(torch.as_tensor(action_stats["mean" if "mean" in action_stats else "min"]).numel()),
                "count": int(count_v),
                "mean": torch.as_tensor(action_stats.get("mean"), dtype=torch.float32).view(-1)
                if action_stats.get("mean") is not None
                else None,
                "std": torch.as_tensor(action_stats.get("std"), dtype=torch.float32).view(-1)
                if action_stats.get("std") is not None
                else None,
                "min": torch.as_tensor(action_stats.get("min"), dtype=torch.float32).view(-1)
                if action_stats.get("min") is not None
                else None,
                "max": torch.as_tensor(action_stats.get("max"), dtype=torch.float32).view(-1)
                if action_stats.get("max") is not None
                else None,
                "q01": torch.as_tensor(action_stats.get("q01"), dtype=torch.float32).view(-1)
                if action_stats.get("q01") is not None
                else None,
                "q99": torch.as_tensor(action_stats.get("q99"), dtype=torch.float32).view(-1)
                if action_stats.get("q99") is not None
                else None,
            }
            try:
                normalizer = build_normalizer_from_stats(mode, norm_stats)
            except Exception:
                normalizer = None

    if normalizer is None and mode != "none":
        print(f"[warn] cannot build normalizer from dataset.meta.stats, fallback to no normalization (mode={mode})")

    def preprocess(batch: Dict, device: torch.device) -> torch.Tensor:
        if "action" in batch:
            action_seq = batch["action"]
        elif "actions" in batch:
            action_seq = batch["actions"]
        else:
            raise KeyError("Batch must contain 'action' or 'actions'.")

        action_seq = torch.as_tensor(action_seq, dtype=torch.float32, device=device)
        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(1)

        valid_len = int(action_seq.shape[1])
        if valid_len > horizon:
            action_seq = action_seq[:, :horizon]
        elif valid_len < horizon:
            pad_len = horizon - valid_len
            pad_tensor = action_seq[:, -1:, :].repeat(1, pad_len, 1)
            action_seq = torch.cat([action_seq, pad_tensor], dim=1)

        if normalizer is not None:
            action_seq = normalizer.normalize(action_seq)
        return action_seq

    return preprocess


def build_vqvae_model_from_config(device: torch.device, cfg: DictConfig):
    runtime_cfg = build_runtime_vqvae_config(cfg)
    vae_wo_ddp = build_vqvae_model(
        device=device,
        **runtime_cfg.model_cfg,
        **runtime_cfg.quant_cfg,
    )
    return vae_wo_ddp


def build_lerobot_base_dataset(cfg: DictConfig) -> LeRobotDataset:
    repo_id = cfg.get("repo_id", None)
    if repo_id is None or str(repo_id).strip() in {"", "???"}:
        raise ValueError("cfg.repo_id must be set for LeRobot dataset training.")

    kwargs = {"repo_id": str(repo_id)}
    root = cfg.get("root", None)
    revision = cfg.get("revision", None)

    if root is not None and str(root).strip() not in {"", "???"}:
        kwargs["root"] = str(root)
    if revision is not None and str(revision).strip() not in {"", "???"}:
        kwargs["revision"] = str(revision)

    horizon = int(cfg.get("horizon", 16))
    fps = cfg.get("fps", None)
    if fps is None:
        fps_probe = LeRobotDataset(**kwargs)
        fps = float(getattr(fps_probe, "fps", fps_probe.meta.fps))

    kwargs["delta_timestamps"] = {
        "action": [t / float(fps) for t in range(horizon)],
    }

    dataset = LeRobotDataset(**kwargs)
    if hasattr(dataset, "select_columns"):
        selected = dataset.select_columns(["action"])
        if selected is not None:
            dataset = selected

    return dataset


def build_training_components(cfg: DictConfig):
    precision = None
    amp_dtype = str(cfg.get("amp_dtype", "bf16")).lower()
    mix_precision = bool(cfg.get("mix_precision", True))
    if mix_precision:
        precision = "bf16-mixed" if amp_dtype == "bf16" else "16-mixed"

    devices = cfg.get("train_gpus", "auto")
    if isinstance(devices, (list, tuple)):
        devices = list(devices)
    strategy = "ddp_find_unused_parameters_true" if isinstance(devices, list) and len(devices) > 1 else "auto"

    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=devices,
        precision=precision,
        strategy=strategy,
    )
    fabric.launch()

    work_dir = resolve_shared_work_dir(fabric=fabric, cfg=cfg)
    if fabric.is_global_zero:
        save_vqvae_configs(work_dir=work_dir, train_cfg=cfg)
    fabric.barrier()

    if not bool(cfg.get("dry", False)) and fabric.is_global_zero:
        wandb.init(
            project=str(cfg.get("project", "MINT-Tokenizer")),
            name=str(cfg.get("name", "lerobot-vae")),
            group=str(cfg.get("group", "lerobot-vae")),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    base_dataset = build_lerobot_base_dataset(cfg)
    horizon = int(cfg.get("horizon", 16))
    preprocess_fn = build_action_preprocess_fn(dataset=base_dataset, cfg=cfg, horizon=horizon)

    ld_train = DataLoader(
        base_dataset,
        batch_size=int(cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 8)),
        pin_memory=(fabric.device.type == "cuda"),
    )

    vae_wo_ddp = build_vqvae_model_from_config(device=fabric.device, cfg=cfg)

    names, paras, para_groups = filter_params(
        vae_wo_ddp,
        nowd_keys={
            "cls_token",
            "start_token",
            "task_token",
            "cfg_uncond",
            "pos_embed",
            "pos_1LC",
            "pos_start",
            "start_pos",
            "lvl_embed",
            "gamma",
            "beta",
            "ada_gss",
            "moe_bias",
            "scale_mul",
        },
    )

    opt_name = str(cfg.get("opt", "adamw")).lower().strip()
    opt_clz = {
        "adam": partial(torch.optim.AdamW, betas=(0.9, 0.95)),
        "adamw": partial(torch.optim.AdamW, betas=(0.9, 0.95)),
    }[opt_name]
    torch_opt = opt_clz(
        params=para_groups,
        lr=float(cfg.get("lr", 3e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.005)),
    )

    vae_ddp, torch_opt = fabric.setup(vae_wo_ddp, torch_opt)
    ld_train = fabric.setup_dataloaders(ld_train)

    mixed_precision_lvl = 2 if (mix_precision and amp_dtype == "bf16") else (1 if mix_precision else 0)
    vae_opt = AmpOptimizer(
        mixed_precision=mixed_precision_lvl,
        optimizer=torch_opt,
        names=names,
        paras=paras,
        grad_clip=float(cfg.get("vclip", 10.0)),
        n_gradient_accumulation=int(cfg.get("ac", 1)),
        fabric=fabric,
    )

    trainer = VQVAETrainer(
        vae_wo_ddp=vae_wo_ddp,
        vae=vae_ddp,
        vae_opt=vae_opt,
        ema_ratio=float(cfg.get("vema", 0.99)),
        is_ema=float(cfg.get("vema", 0.99)) > 0.0,
        dct_loss_weight=float(cfg.get("dct_loss_weight", 0.05)),
    )

    start_ep, start_it = 0, 0
    last_ckpt = None
    latest_ckpt = os.path.join(work_dir, "latest.pth")
    if os.path.isfile(latest_ckpt):
        last_ckpt = latest_ckpt
    else:
        ckpts = sorted(
            [fn for fn in os.listdir(work_dir) if fn.startswith("vae-ckpt-") and fn.endswith(".pth")],
            key=lambda x: os.path.getmtime(os.path.join(work_dir, x)),
            reverse=True,
        ) if os.path.isdir(work_dir) else []
        if ckpts:
            last_ckpt = os.path.join(work_dir, ckpts[0])

    if last_ckpt is not None:
        if fabric.is_global_zero:
            print(f"[auto-resume] loading checkpoint from {last_ckpt}")
        state = torch.load(last_ckpt, map_location="cpu")
        start_ep = int(state.get("epoch", 0))
        start_it = int(state.get("iter", 0))
        tr_state = state.get("trainer", {})
        if tr_state:
            trainer.load_state_dict(tr_state, strict=False)

    iters_train = len(ld_train)
    return work_dir, trainer, start_ep, start_it, iters_train, ld_train, preprocess_fn, fabric


def train_one_ep(
    ep: int,
    start_it: int,
    cfg: DictConfig,
    dataloader,
    iters_train: int,
    trainer: VQVAETrainer,
    preprocess_fn: Callable,
    fabric: Fabric,
):
    me_lg = misc.MetricLogger(delimiter="  ", is_master=fabric.is_global_zero)
    me_lg.add_meter("vlr", misc.SmoothedValue(window_size=1, fmt="{value:.2g}"))
    header = f"[Ep]: [{ep:4d}/{int(cfg.get('epochs', 100))}]"

    g_it, max_it = ep * iters_train, int(cfg.get("epochs", 100)) * iters_train

    for it, obj in me_lg.log_every(start_it, iters_train, dataloader, 30 if iters_train > 8000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it:
            continue

        inp = preprocess_fn(obj, fabric.device)

        wp_it = float(cfg.get("vwp", 1)) * iters_train
        min_vlr, max_vlr, min_vwd, max_vwd = lr_wd_annealing(
            str(cfg.get("vsche", "cos")),
            trainer.vae_opt.optimizer,
            float(cfg.get("lr", 3e-5)),
            float(cfg.get("weight_decay", 0.005)),
            float(cfg.get("vwde", 0.0)),
            g_it,
            wp_it,
            max_it,
            wp0=float(cfg.get("vwp0", 0.005)),
            wpe=float(cfg.get("vwpe", 0.3)),
        )

        stepping = (g_it + 1) % int(cfg.get("ac", 1)) == 0
        grad_norm, scale_log2, log_dict = trainer.train_step(
            it=it,
            g_it=g_it,
            stepping=stepping,
            me_lg=me_lg,
            inp=inp,
        )

        me_lg.update(vlr=max_vlr)
        me_lg.update(grad_norm=grad_norm)

        if log_dict is not None and tdist.is_available() and tdist.is_initialized():
            for k, v in list(log_dict.items()):
                try:
                    t = torch.tensor(float(v), device=fabric.device)
                    tdist.all_reduce(t, op=tdist.ReduceOp.SUM)
                    log_dict[k] = (t / fabric.world_size).item()
                except Exception:
                    pass

        if (not bool(cfg.get("dry", False))) and fabric.is_global_zero:
            iter_metrics = {
                "opt/lr_min": min_vlr,
                "opt/lr_max": max_vlr,
                "opt/wd_min": min_vwd,
                "opt/wd_max": max_vwd,
                "opt/fp16_scale_log2": scale_log2,
                "opt/grad_norm": grad_norm,
                "opt/grad_clip": float(cfg.get("vclip", 10.0)),
            }
            if log_dict is not None:
                for k, v in log_dict.items():
                    iter_metrics[f"train_iter/{k}"] = v
            wandb.log(iter_metrics, step=g_it + 1)

    me_lg.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me_lg.meters.items()}


def run_vqvae_training(cfg: DictConfig):
    setup(cfg)

    work_dir, trainer, start_ep, start_it, iters_train, ld_train, preprocess_fn, fabric = build_training_components(cfg=cfg)
    runtime_cfg_dict = OmegaConf.to_container(build_runtime_vqvae_config(cfg), resolve=True)

    if fabric.is_global_zero:
        print("[SDAT] training starts")

    epochs = int(cfg.get("epochs", 100))
    save_freq = int(cfg.get("ckpt_interval", 20)) if int(cfg.get("ckpt_interval", 0)) > 0 else int(cfg.get("save_freq", 20))

    start_time = time.time()
    for ep in range(start_ep, epochs):
        if hasattr(ld_train, "sampler") and hasattr(ld_train.sampler, "set_epoch"):
            ld_train.sampler.set_epoch(ep)

        stats = train_one_ep(
            ep=ep,
            start_it=start_it if ep == start_ep else 0,
            cfg=cfg,
            dataloader=ld_train,
            iters_train=iters_train,
            trainer=trainer,
            preprocess_fn=preprocess_fn,
            fabric=fabric,
        )

        should_save = (ep + 1) % save_freq == 0 or (ep + 1) == epochs or ep == 0
        if fabric.is_global_zero:
            ckpt_payload = {
                "epoch": ep + 1,
                "iter": 0,
                "trainer": trainer.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "vqvae_runtime_cfg": runtime_cfg_dict,
            }

            latest_path = os.path.join(work_dir, "latest.pth")
            torch.save(ckpt_payload, latest_path)

            if should_save:
                ckpt_path = os.path.join(work_dir, f"vae-ckpt-{ep + 1}.pth")
                torch.save(ckpt_payload, ckpt_path)
                print(f"[saving ckpt] {ckpt_path}")

        fabric.barrier()

        if (not bool(cfg.get("dry", False))) and fabric.is_global_zero:
            epoch_metrics = {f"epoch/{k}": v for k, v in stats.items()}
            wandb.log(epoch_metrics, step=(ep + 1) * iters_train)

    if fabric.is_global_zero:
        total_time = (time.time() - start_time) / 3600.0
        print(f"[SDAT] finished, total {total_time:.2f}h")

    del stats
    del iters_train, ld_train
    time.sleep(1)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if wandb.run is not None:
        wandb.finish()
    fabric.barrier()


def setup(cfg: DictConfig):
    import warnings

    warnings.simplefilter("ignore")
    lightning.seed_everything(int(cfg.get("seed", 42)))


if __name__ == "__main__":
    raise RuntimeError("Use `python -m SDAT.train` to run VQ-VAE training.")

