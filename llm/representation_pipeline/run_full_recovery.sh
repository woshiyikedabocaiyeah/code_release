#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PACKAGE_ROOT}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-mllm}"
NOISE_ITERATIONS="${NOISE_ITERATIONS:-2000}"
PERMUTATIONS="${PERMUTATIONS:-10000}"
RUN_PROBE="${RUN_PROBE:-0}"
PROBE_BACKEND="${PROBE_BACKEND:-mlp}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"

RUN_ROOT="${PROJECT_ROOT}/outputs/semantic_yesno_main_experiment"
QUESTIONS_PATH="${PROJECT_ROOT}/questions.json"
PROBE_ROOT="${PACKAGE_ROOT}/Probe"

PY_CMD=(conda run -n "${CONDA_ENV}" python)

echo "[recovery] rebuilding main review package analyses"
"${PY_CMD[@]}" "${PACKAGE_ROOT}/Scripts/recover_review_package.py" \
  --noise-iterations "${NOISE_ITERATIONS}" \
  --permutations "${PERMUTATIONS}"

if [[ "${RUN_PROBE}" == "1" ]]; then
  echo "[recovery] rebuilding overall probe with ${PROBE_BACKEND} on ${PROBE_DEVICE}"
  "${PY_CMD[@]}" "${PROBE_ROOT}/Scripts/probe_training.py" \
    --probe-backend "${PROBE_BACKEND}" \
    --run-name overall_probe_yesno_20260503 \
    --analysis-scope overall \
    --run-root "${RUN_ROOT}" \
    --questions-path "${QUESTIONS_PATH}" \
    --output-dir "${PROBE_ROOT}/Data/Runs/overall_probe_yesno_20260503" \
    --device "${PROBE_DEVICE}"

  echo "[recovery] rebuilding task-conditioned probe with ${PROBE_BACKEND} on ${PROBE_DEVICE}"
  "${PY_CMD[@]}" "${PROBE_ROOT}/Scripts/probe_training.py" \
    --probe-backend "${PROBE_BACKEND}" \
    --run-name task_conditioned_probe_yesno_20260503 \
    --analysis-scope task_conditioned \
    --run-root "${RUN_ROOT}" \
    --questions-path "${QUESTIONS_PATH}" \
    --output-dir "${PROBE_ROOT}/Data/Runs/task_conditioned_probe_yesno_20260503" \
    --device "${PROBE_DEVICE}"
else
  echo "[recovery] keeping existing probe outputs; submit GPU MLP probe with Scripts/submit_probe_gpu_slurm.sh when probe retraining is needed"
fi

echo "[recovery] rebuilding probe derived tables and reports"
"${PY_CMD[@]}" "${PROBE_ROOT}/Scripts/build_probe_derivatives.py"
"${PY_CMD[@]}" "${PROBE_ROOT}/Scripts/build_probe_question_evidence.py"
"${PY_CMD[@]}" "${PROBE_ROOT}/Scripts/build_probe_report.py"
"${PY_CMD[@]}" "${PROBE_ROOT}/Scripts/build_tsne_figures.py" --scope task_conditioned
"${PY_CMD[@]}" "${PACKAGE_ROOT}/Scripts/repair_schema_compatibility.py"
"${PY_CMD[@]}" "${PACKAGE_ROOT}/Scripts/compare_review_package_schema.py"

echo "[recovery] complete: ${PACKAGE_ROOT}"
