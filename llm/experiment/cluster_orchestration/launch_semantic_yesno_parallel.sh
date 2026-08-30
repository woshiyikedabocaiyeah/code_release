#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS_FILE="${ROOT_DIR}/logs/semantic_yesno_parallel_jobs.tsv"
cd "${ROOT_DIR}"

TARGET_MODELS=(
  InternVL3.5
  GLM-4.1V-base
  GLM-4.1V-thinking
  Qwen
  Qwen-Thinking
  RynnBrain-8B
  RynnBrain-CoP
  RoboBrain2.5
  MiMo-Embodied
)

mkdir -p "${ROOT_DIR}/logs"
: > "${JOBS_FILE}"

for model in "${TARGET_MODELS[@]}"; do
  output="$("${ROOT_DIR}/scripts/submit_slurm.sh" "${ROOT_DIR}/slurm/run_semantic_yesno_model.sbatch" "${model}")"
  job_id="$(printf "%s\n" "${output}" | awk '{print $4}')"
  printf "%s\t%s\n" "${job_id}" "${model}" | tee -a "${JOBS_FILE}"
  printf "%s\n" "${output}"
done

printf "Parallel semantic Yes/No jobs written to %s\n" "${JOBS_FILE}"
