#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT="${CHECKPOINT:-${1:-}}"
: "${CHECKPOINT:?Set CHECKPOINT or pass a local model path or Hugging Face repo ID as the first argument}"
export CHECKPOINT
VQVAE="${VQVAE:-${2:-}}"
: "${VQVAE:?Set VQVAE or pass a local VQ-VAE path or Hugging Face repo ID as the second argument}"
export VQVAE

# transfer_new_layout: transfer the basket-placement intent from LIBERO-10
# task 7 in living room scene 1 to task 1 in living room scene 2.
export SUITE="libero_10"
export TASK_ID="1"
export DEMO_SUITE="libero_10"
export DEMO_TASK_ID="7"
export MODE="transfer"

export DEMO_PATH="${DEMO_PATH:-${TRANSFER_DIR}/demo_traj/transfer_new_layout.npz}"
if [[ ! -f "${DEMO_PATH}" ]]; then
    echo "Demonstration file not found: ${DEMO_PATH}" >&2
    exit 1
fi

export FORCE_LEVELS="${FORCE_LEVELS:-0}"
export NUM_EPISODES="${NUM_EPISODES:-20}"
export NUM_SAMPLES="${NUM_SAMPLES:-3}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-2}"
export MAX_STEPS="${MAX_STEPS:-170}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-${TRANSFER_DIR}/../outputs/transfer_new_layout}"

exec "${SCRIPT_DIR}/run_transfer.sh"
