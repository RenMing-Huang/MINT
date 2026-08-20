<div align="center">
  <h1><img src="docs/static/images/favicon.png" style="height:40px;vertical-align:middle;margin-right:10px;">Mimic Intent, Not Just Trajectories</h1>
  <p><strong>An intent-to-execution policy for precise and transferable robotic manipulation.</strong></p>

  <p>
    <a href="https://arxiv.org/abs/2602.08602"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.08602-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
    <a href="https://huggingface.co/huangrm/MINT-libero-130"><img alt="Hugging Face Policy" src="https://img.shields.io/badge/HuggingFace-Policy-ffca28?style=for-the-badge&logo=huggingface&logoColor=black"></a>
    <a href="https://huggingface.co/huangrm/MINT-tokenizer-libero-130"><img alt="Hugging Face Tokenizer" src="https://img.shields.io/badge/HuggingFace-Tokenizer-ffca28?style=for-the-badge&logo=huggingface&logoColor=black"></a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img alt="Status" src="https://img.shields.io/badge/Status-Released-16a34a?style=for-the-badge" />
    <img alt="License" src="https://img.shields.io/badge/License-MIT-16a34a?style=for-the-badge" />
  </p>
</div>

<div align="center">
  <img src="docs/static/images/teaser.png" width="92%" alt="MINT Teaser"/>
  <p style="text-align:left;"><em>SDAT maps each action chunk into multi-scale tokens: coarse tokens capture <strong>intent</strong>, and fine tokens capture <strong>execution details</strong>. The S1 token space forms behavior-level clusters.</em></p>
  <img src="docs/static/images/overview.jpg" width="92%" alt="MINT Overview"/>
  <p style="text-align:left;"><em>MINT predicts tokens from intent to execution with next-scale autoregression, then decodes them into actions. Intent-based ensemble improves long-horizon stability.</em></p>
</div>

---

## Overview

We introduce MINT (Mimic Intent, Not just Trajectories), a framework for end-to-end imitation learning in dexterous manipulation. MINT explicitly <span style="background-color:#e0f2fe;color:#075985;padding:0 4px;border-radius:4px;"><strong>disentangles behavior intent from execution details</strong></span> by learning a hierarchical, multi-scale token representation of actions. Coarse tokens capture global, low-frequency intent, while finer tokens encode high-frequency execution details. Our policy generates trajectories via <span style="background-color:#e0f2fe;color:#075985;padding:0 4px;border-radius:4px;"><strong>next-scale autoregression</strong></span>, performing progressive <span style="background-color:#e0f2fe;color:#075985;padding:0 4px;border-radius:4px;"><strong>intent-to-execution reasoning</strong></span>. This structure enables efficient learning, robust adaptation to environmental dynamics, and <span style="background-color:#e0f2fe;color:#075985;padding:0 4px;border-radius:4px;"><strong>one-shot skill transfer</strong></span> by reusing the intent token from a demonstration. Experiments on simulation and real robots demonstrate strong performance, high generalization, and effective skill transfer.

## Open-Source Roadmap

| Track | Scope | Status | Target |
|---|---|---|---|
| ✅ LeRobot Integration | MINT-4B training/evaluation pipeline | Released | Done |
| ✅ Public Weights | LIBERO-130 policy + tokenizer on Hugging Face | Released | Done |
| ✅ SDAT Training | Training scripts + configs | Released | Done |
| ✅ Lightweight MINT-30M | LeRobot-compatible MINT-Light policy | Released | Done |
| 🗓 Multi-dataset Checkpoints | CALVIN / Bridge policy-tokenizer pairs | Planned | 2026 H2 |
| 🗓 Support Bimanual Manipulation | RoboTwin and other bimanual manipulation benchmarks | Planned | 2026 H3 |

## Installation

### LeRobot compatibility

| Branch | LeRobot version | Status |
|---|---|---|
| `main` | `0.5.1` | Current release |
| `legacy/lerobot-0.4.3` | `0.4.3` | Archived previous codebase |

The current `main` branch supports LeRobot 0.5.1. Use the legacy branch only when an existing
environment must remain on LeRobot 0.4.3; the two integrations should not be mixed.

```bash
conda create -y -n mint python=3.12 cmake=3.11
conda activate mint

pip install "lerobot[pi]==0.5.1"
# Install LIBERO dependencies via LeRobot:
pip install "lerobot[libero]==0.5.1"

# Install all MINT policy runtime dependencies:
pip install -r requirements.txt

conda install -y ffmpeg -c conda-forge
```

```bash
# install policy
pip install -e ./policy/lerobot_policy_mint
pip install -e ./policy/lerobot_policy_mint_light
```

Note: If you encounter build errors on Linux, you may also need system packages such as cmake,
build-essential, python3-dev, pkg-config, and FFmpeg development libraries.

```bash
apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
```

## Model Zoo

The [`policy`](policy) directory contains two independent packages with matching
`pyproject.toml` plus `src/<package>/` layouts. Install the package you need, then choose
`--policy.type=mint` or `--policy.type=mint_light`; both use the standard LeRobot commands.

### MINT-4B (LeRobot implementation) 🤗
| Dataset | Policy | Tokenizer | Status | Notes |
|---|---|---|---|---|
| [LIBERO-130](Retoc71586/libero_130_lerobot_v3) | [huangrm/MINT-libero-130](https://huggingface.co/huangrm/MINT-libero-130) | [huangrm/MINT-tokenizer-libero-130](https://huggingface.co/huangrm/MINT-tokenizer-libero-130) | Available | LeRobot 0.5.1 release |
| CALVIN | Coming soon | Coming soon | Planned | Upcoming release |
| [Bridge](FedorX8/bridge_v2_lerobot) | Coming soon | Coming soon | Planned | Upcoming release |

### MINT-Light / MINT-30M ⚡

| Dataset | Policy | Tokenizer | Status | Notes |
|---|---|---|---|---|
| LIBERO | Train locally | [huangrm/MINT-tokenizer-libero-130](https://huggingface.co/huangrm/MINT-tokenizer-libero-130)  | Available | Light-weight version |


- MINT-4B (`mint`) uses the PaliGemma backbone and action expert for maximum capacity.
- MINT-Light (`mint_light`) keeps MINT's coarse-to-fine action-token prediction but replaces
the large VLM with a fixed DINOv3 ViT-L/16 visual encoder, a SigLIP2 text encoder.

## Training Example

First, download the required tokenizer:

```bash
hf download huangrm/MINT-tokenizer-libero-130 --local-dir <path/to/tokenizer>
```
Or, train your own tokenizer:
```bash
# install tokenizer training dependencies
pip install -r requirements.txt
python -m SDAT.train --config-name train
```

Start MINT-4B training:

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    $(which lerobot-train) \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --policy.type=mint \
    --output_dir=<path/to/output> \
    --job_name=mint_training \
    --policy.repo_id=mint \
    --policy.pretrained_path=huangrm/pi05_base \
    --policy.vqvae_name_or_path=<path/to/tokenizer-checkpoint> \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=float32 \
    --steps=100000 \
    --save_freq=20000 \
    --policy.device=cuda \
    --batch_size=16
```

For MINT-Light, use the same command and change the policy type. you can reuse the same tokenizer checkpoint from above, or train a new one with the SDAT training script.

```bash
lerobot-train \
    --dataset.repo_id=<dataset-repo-id> \
    --policy.type=mint_light \
    --policy.vqvae_name_or_path=<path/to/tokenizer-checkpoint> \
    --policy.device=cuda \
    --output_dir=<path/to/output>
```

## Evaluation

```bash
lerobot-eval \
    --policy.path=huangrm/MINT-libero-130 \
    --env.type=libero \
    --env.task=libero_90,libero_10,libero_goal,libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --seed=1000 \
    --policy.n_action_steps=4
```

Evaluation is identical for MINT-Light: point `--policy.path` at a MINT-Light
`pretrained_model` directory. LeRobot reads the saved policy type automatically.

## Citation

If you find this project useful, please cite:

```bibtex
@article{huang2026mimic,
  title={Mimic Intent, Not Just Trajectories},
  author={Huang, Renming and Zeng, Chendong and Tang, Wenjing and Cai, Jintian and Lu, Cewu and Cai, Panpan},
  journal={arXiv preprint arXiv:2602.08602},
  year={2026}
}
```

## Acknowledgement

This project is built on top of excellent open-source ecosystems.
We sincerely thank the teams behind [LeRobot](https://github.com/huggingface/lerobot)
and [OpenPI](https://github.com/Physical-Intelligence/openpi) for their impactful contributions.
