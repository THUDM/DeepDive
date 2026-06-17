#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -f "${TRAINING_ROOT}/.env" ]]; then
  set -a
  source "${TRAINING_ROOT}/.env"
  set +a
fi

cd "${TRAINING_ROOT}/reward_server"

python3 launch_server.py \
  --port "${RM_TRAIN_PORT:-8888}" \
  --model_name "${REWARD_MODEL_NAME:-gpt-4o-mini}" \
  --base_url "${REWARD_BASE_URL:-https://api.openai.com/v1}" \
  --api_key "${REWARD_API_KEY:?REWARD_API_KEY is required}"
