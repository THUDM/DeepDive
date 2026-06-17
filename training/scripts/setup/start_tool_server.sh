#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -f "${TRAINING_ROOT}/.env" ]]; then
  set -a
  source "${TRAINING_ROOT}/.env"
  set +a
fi

cd "${TRAINING_ROOT}/tool_server"

PROXY_ARGS=()
if [[ -n "${HTTP_PROXY:-}" ]]; then
  PROXY_ARGS+=(--http_proxy "${HTTP_PROXY}")
fi

SEARCH_PROVIDER="${SEARCH_PROVIDER:-serper}"
SEARCH_ARGS=(--search_provider "${SEARCH_PROVIDER}")
if [[ "${SEARCH_PROVIDER}" == "serper" ]]; then
  SEARCH_ARGS+=(--serper_api_key "${SERPER_API_KEY:?SERPER_API_KEY is required when SEARCH_PROVIDER=serper}")
else
  SEARCH_ARGS+=(--serp_api_key "${SERP_API_KEY:?SERP_API_KEY is required when SEARCH_PROVIDER=serpapi}")
fi

python3 launch_server.py \
  --jina_api_key "${JINA_API_KEY:?JINA_API_KEY is required}" \
  --port "${TOOL_SERVER_PORT:-7230}" \
  "${SEARCH_ARGS[@]}" \
  "${PROXY_ARGS[@]}"
