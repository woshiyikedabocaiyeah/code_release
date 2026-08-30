#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${PACKAGE_ROOT}/Logs"

OVERALL_JOB_ID="$(
  sbatch --parsable "${SCRIPT_DIR}/slurm_probe_gpu.sbatch" \
    overall \
    overall_probe_yesno_20260503
)"

TASK_JOB_ID="$(
  sbatch --parsable "${SCRIPT_DIR}/slurm_probe_gpu.sbatch" \
    task_conditioned \
    task_conditioned_probe_yesno_20260503
)"

POST_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${OVERALL_JOB_ID}:${TASK_JOB_ID}" \
    "${SCRIPT_DIR}/slurm_probe_postprocess.sbatch"
)"

echo "overall_job_id=${OVERALL_JOB_ID}"
echo "task_conditioned_job_id=${TASK_JOB_ID}"
echo "postprocess_job_id=${POST_JOB_ID}"
echo "logs=${PACKAGE_ROOT}/Logs"
