#!/usr/bin/env bash
#SBATCH --job-name=yesno-probe-gpu
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=$PROJECT_ROOT/analysis/review_package/Logs/%x-%j.out
#SBATCH --error=$PROJECT_ROOT/analysis/review_package/Logs/%x-%j.err

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: slurm_probe_gpu.sbatch <overall|task_conditioned> <run_name>" >&2
  exit 2
fi

SCOPE="$1"
RUN_NAME="$2"
CONDA_ENV="${CONDA_ENV:-mllm}"

PACKAGE_ROOT="$PROJECT_ROOT/analysis/review_package"
PROJECT_ROOT="$(cd "${PACKAGE_ROOT}/../.." && pwd)"
PROBE_ROOT="${PACKAGE_ROOT}/Probe"
RUN_ROOT="${PROJECT_ROOT}/outputs/semantic_yesno_main_experiment"
QUESTIONS_PATH="${PROJECT_ROOT}/questions.json"
OUTPUT_DIR="${PROBE_ROOT}/Data/Runs/${RUN_NAME}"

mkdir -p "${PACKAGE_ROOT}/Logs" "${OUTPUT_DIR}"

echo "[slurm-probe] scope=${SCOPE}"
echo "[slurm-probe] run_name=${RUN_NAME}"
echo "[slurm-probe] output_dir=${OUTPUT_DIR}"
echo "[slurm-probe] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

conda run -n "${CONDA_ENV}" python "${PROBE_ROOT}/Scripts/probe_training.py" \
  --probe-backend mlp \
  --run-name "${RUN_NAME}" \
  --analysis-scope "${SCOPE}" \
  --run-root "${RUN_ROOT}" \
  --questions-path "${QUESTIONS_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda
