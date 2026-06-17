#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ ! -f "${TRAINING_ROOT}/.env" ]]; then
  cp "${TRAINING_ROOT}/.env.example" "${TRAINING_ROOT}/.env"
  echo "Created ${TRAINING_ROOT}/.env from .env.example"
else
  echo "${TRAINING_ROOT}/.env already exists"
fi

echo "Install dependencies with:"
echo "  pip install -r ${TRAINING_ROOT}/requirements.txt"
echo "  pip install -r ${TRAINING_ROOT}/slime/requirements.txt"
echo "  pip install -e ${TRAINING_ROOT}/slime"
echo "Then edit ${TRAINING_ROOT}/.env before starting training services."
