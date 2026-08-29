#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${CHECKPOINT:?Set CHECKPOINT to a local model path or Hugging Face repo ID}"
: "${TASK_ID:?Set TASK_ID to the target LIBERO task id}"
: "${DEMO_PATH:?Set DEMO_PATH to an extracted demonstration NPZ file}"

SUITE="${SUITE:-libero_90}"
DEMO_SUITE="${DEMO_SUITE:-${SUITE}}"
DEMO_TASK_ID="${DEMO_TASK_ID:-${TASK_ID}}"
FORCE_LEVELS="${FORCE_LEVELS:-0}"
NUM_EPISODES="${NUM_EPISODES:-20}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
SEED="${SEED:-1223}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/transfer}"
MODE="${MODE:-transfer}"

# httpx cannot use a SOCKS proxy without the optional socksio package. Keep the
# available HTTP(S) proxy as a fallback in that environment.
if [[ "${ALL_PROXY:-}" == socks* ]] && ! python -c "import socksio" >/dev/null 2>&1; then
    unset ALL_PROXY all_proxy
fi

export PYTHONPATH="${REPO_ROOT}/policy/lerobot_policy_mint_light/src:${REPO_ROOT}:${PYTHONPATH:-}"

COMMON_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --suite "${SUITE}"
    --task-id "${TASK_ID}"
    --demo-suite "${DEMO_SUITE}"
    --demo-task-id "${DEMO_TASK_ID}"
    --demo-path "${DEMO_PATH}"
    --mode "${MODE}"
    --force-levels "${FORCE_LEVELS}"
    --num-episodes "${NUM_EPISODES}"
    --num-samples "${NUM_SAMPLES}"
    --n-action-steps "${N_ACTION_STEPS}"
    --seed "${SEED}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ "${SAVE_VIDEO:-0}" == "1" ]]; then
    COMMON_ARGS+=(--save-video)
fi
if [[ -n "${VQVAE:-}" ]]; then
    COMMON_ARGS+=(--vqvae "${VQVAE}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_ARGS+=(--max-steps "${MAX_STEPS}")
fi

python -m transfer.verify_transfer "${COMMON_ARGS[@]}"
