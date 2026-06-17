#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -f "${TRAINING_ROOT}/.env" ]]; then
  set -a
  source "${TRAINING_ROOT}/.env"
  set +a
fi

SLIME_ROOT="${SLIME_ROOT:-${TRAINING_ROOT}/slime}"
DEEPDIVE_ROOT="${DEEPDIVE_ROOT:-$(cd "${TRAINING_ROOT}/.." && pwd)}"
export PYTHONPATH="${DEEPDIVE_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"
cd "${SLIME_ROOT}"

HF_MODEL_PATH="${HF_MODEL_PATH:?HF_MODEL_PATH is required}"
REF_MODEL_TORCH_DIST="${REF_MODEL_TORCH_DIST:?REF_MODEL_TORCH_DIST is required}"
PROMPT_DATA="${PROMPT_DATA:?PROMPT_DATA is required}"
CKPT_DIR="${CKPT_DIR:?CKPT_DIR is required}"

MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-64000}"
MAX_GEN_LEN="${MAX_GEN_LEN:-64000}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.8}"
export DEEPDIVE_TOOL_MAX_RETRY="${DEEPDIVE_TOOL_MAX_RETRY:-${TOOL_MAX_RETRY:-5}}"
export DEEPDIVE_TOOL_TIMEOUT="${DEEPDIVE_TOOL_TIMEOUT:-${TOOL_TIMEOUT:-300}}"
export DEEPDIVE_STOP_ONCE_ILLFORM="${DEEPDIVE_STOP_ONCE_ILLFORM:-1}"

MODEL_ARGS=(
  --max-position-embeddings "${MAX_CONTEXT_LEN}"
  --seq-length "${MAX_CONTEXT_LEN}"
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${HF_MODEL_PATH}"
)

CKPT_ARGS=(
  --hf-checkpoint "${HF_MODEL_PATH}"
  --ref-load "${REF_MODEL_TORCH_DIST}"
  --save "${CKPT_DIR}"
  --load "${CKPT_DIR}"
  --save-interval "${SAVE_INTERVAL:-20}"
  --ckpt-format torch_dist
  --no-load-rng
  --no-load-optim
)

ROLLOUT_ARGS=(
  --custom-generate-function-path training.rollout.deepdive_rollout.generate_with_tool
  --prompt-data "${PROMPT_DATA}"
  --input-key input_messages
  --label-key label
  --tool-key tools
  --metadata-key metadata
  --apply-chat-template
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
  --num-rollout "${NUM_ROLLOUT:-3000}"
  --rollout-max-context-len "${MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${MAX_GEN_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
  --rollout-top-p "${ROLLOUT_TOP_P:-1}"
)
if [[ "${USE_DEEPDIVE_REWARD_FUNC:-1}" == "1" ]]; then
  ROLLOUT_ARGS+=(--custom-rm-path training.rollout.deepdive_rollout.reward_func)
fi
if [[ -n "${CUSTOM_CONFIG_PATH:-}" ]]; then
  ROLLOUT_ARGS+=(--custom-config-path "${CUSTOM_CONFIG_PATH}")
fi

DISTRIBUTED_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --sequence-parallel
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef "${KL_LOSS_COEF:-0.00}"
  --kl-loss-type low_var_kl
  --kl-coef "${KL_COEF:-0.00}"
  --entropy-coef "${ENTROPY_COEF:-0.00}"
  --eps-clip "${EPS_CLIP:-0.2}"
  --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
  --calculate-per-token-loss
  --loss-mask-type "${LOSS_MASK_TYPE:-qwen3}"
)

OPTIMIZER_ARGS=(
  --lr "${LR:-2e-6}"
  --lr-warmup-iters "${LR_WARMUP_ITERS:-0}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --adam-beta1 0.9
  --adam-beta2 0.98
  --override-opt_param-scheduler
)

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-key "${WANDB_API_KEY}"
    --wandb-project "${WANDB_PROJECT:-deepdive-rl}"
    --wandb-group "${WANDB_GROUP:-deepdive-slime}"
    --disable-wandb-random-suffix
    --wandb-always-use-train-step
  )
fi

mkdir -p "${CKPT_DIR}"

python3 train.py \
  --actor-num-nodes "${ACTOR_NUM_NODES:-1}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE:-8}" \
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS:-8}" \
  --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}" \
  --sglang-router-request-timeout-secs 36000 \
  --sglang-router-balance-abs-threshold 0 \
  --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION}" \
  --offload \
  --colocate \
  --no-check-for-nan-in-loss-and-grad \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${DISTRIBUTED_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
