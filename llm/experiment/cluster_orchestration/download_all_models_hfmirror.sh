#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-mllm}"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary="${LOG_DIR}/download_summary_${timestamp}.json"

python -u "${ROOT_DIR}/tools/download_models.py" \
  --endpoint "${HF_ENDPOINT}" \
  --summary "${summary}" \
  "$@"
