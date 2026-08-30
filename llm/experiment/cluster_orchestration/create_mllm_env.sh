#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-mllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment ${ENV_NAME} already exists; installing/updating packages"
else
  echo "Creating conda environment ${ENV_NAME}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

python -m pip install \
  numpy \
  pandas \
  pillow \
  opencv-python \
  einops \
  "huggingface_hub[cli]" \
  qwen-vl-utils \
  safetensors \
  sentencepiece \
  transformers

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
  python -m pip install accelerate timm
else
  echo "Skipping torch/torchvision/accelerate/timm install because INSTALL_TORCH=${INSTALL_TORCH}"
fi

python - <<'PY'
import importlib
mods = ["transformers", "huggingface_hub", "cv2", "PIL", "numpy", "pandas"]
try:
    import torch  # noqa: F401
except Exception as exc:
    print("torch skipped or unavailable:", exc)
else:
    mods.extend(["torch", "accelerate"])
for name in mods:
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "ok"))
PY
