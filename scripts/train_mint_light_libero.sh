#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Example: TOKENIZER_CKPT=/path/to/vae.pth GPU_IDS=0,1 bash scripts/train_mint_light_libero.sh
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_LIST[@]}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset/HuggingFaceVLA/libero}"
DATASET_REPO_ID="${DATASET_REPO_ID:-libero}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/mint_light_libero_${RUN_ID}}"
JOB_NAME="${JOB_NAME:-mint_light_libero}"

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-500}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
IMAGE_TRANSFORMS_ENABLE="${IMAGE_TRANSFORMS_ENABLE:-true}"
IMAGE_TRANSFORMS_MAX_NUM="${IMAGE_TRANSFORMS_MAX_NUM:-2}"
IMAGE_TRANSFORMS_RANDOM_ORDER="${IMAGE_TRANSFORMS_RANDOM_ORDER:-false}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
NUM_LAYERS="${NUM_LAYERS:-10}"
NUM_HEADS="${NUM_HEADS:-6}"
HEAD_DIM="${HEAD_DIM:-64}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-1024}"

if [[ -z "${TOKENIZER_CKPT}" ]]; then
    echo "TOKENIZER_CKPT is required." >&2
    exit 1
fi
if [[ ! -f "${TOKENIZER_CKPT}" ]]; then
    echo "Tokenizer checkpoint not found: ${TOKENIZER_CKPT}" >&2
    exit 1
fi
if [[ ! -f "$(dirname -- "${TOKENIZER_CKPT}")/config.yaml" ]]; then
    echo "Tokenizer config not found next to checkpoint: $(dirname -- "${TOKENIZER_CKPT}")/config.yaml" >&2
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "LeRobot dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2
    exit 1
fi
if (( NUM_PROCESSES < 1 || NUM_PROCESSES > ${#GPU_LIST[@]} )); then
    echo "NUM_PROCESSES=${NUM_PROCESSES} must be between 1 and the number of GPU_IDS (${#GPU_LIST[@]})." >&2
    exit 1
fi

TRAIN_BIN="${TRAIN_BIN:-$(command -v lerobot-train || true)}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate || true)}"
if [[ -z "${TRAIN_BIN}" ]]; then
    echo "lerobot-train is unavailable. Activate the MINT environment and install LeRobot 0.5.1." >&2
    exit 1
fi
if [[ -z "${ACCELERATE_BIN}" ]]; then
    echo "accelerate is unavailable. Activate the MINT environment and install the project requirements." >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output path already exists: ${OUTPUT_DIR}" >&2
    echo "Set RUN_ID or OUTPUT_DIR to a new path." >&2
    exit 1
fi

ACCELERATE_ARGS=(
    launch
    --num_machines=1
    --num_processes="${NUM_PROCESSES}"
    --main_process_port="${MAIN_PROCESS_PORT}"
    --mixed_precision="${MIXED_PRECISION}"
)
if (( NUM_PROCESSES > 1 )); then
    ACCELERATE_ARGS+=(--multi_gpu)
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "Starting MINT-Light training"
echo "  GPUs:       ${CUDA_VISIBLE_DEVICES} (${NUM_PROCESSES} processes)"
echo "  Dataset:    ${DATASET_ROOT}"
echo "  Tokenizer:  ${TOKENIZER_CKPT}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE} per process"
echo "  Image aug:  ${IMAGE_TRANSFORMS_ENABLE} (${IMAGE_TRANSFORMS_MAX_NUM} transforms per image)"
echo "  Vision:     DINOv3 ViT-L/16"
echo "  Transformer:${NUM_LAYERS} layers, ${NUM_HEADS} heads x ${HEAD_DIM}, hidden=${HIDDEN_DIM}"

exec "${ACCELERATE_BIN}" "${ACCELERATE_ARGS[@]}" "${TRAIN_BIN}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable="${IMAGE_TRANSFORMS_ENABLE}" \
    --dataset.image_transforms.max_num_transforms="${IMAGE_TRANSFORMS_MAX_NUM}" \
    --dataset.image_transforms.random_order="${IMAGE_TRANSFORMS_RANDOM_ORDER}" \
    --policy.type=mint_light \
    --policy.vqvae_name_or_path="${TOKENIZER_CKPT}" \
    --policy.hidden_dim="${HIDDEN_DIM}" \
    --policy.num_layers="${NUM_LAYERS}" \
    --policy.num_heads="${NUM_HEADS}" \
    --policy.head_dim="${HEAD_DIM}" \
    --policy.mlp_hidden_dim="${MLP_HIDDEN_DIM}" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --wandb.enable=false
