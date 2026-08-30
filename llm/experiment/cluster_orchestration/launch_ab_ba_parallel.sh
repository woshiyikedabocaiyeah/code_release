#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
JOBS_FILE="${LOG_DIR}/ab_ba_parallel_jobs.tsv"

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

mkdir -p "${LOG_DIR}"
python "${ROOT_DIR}/scripts/prepare_ba_questions.py" \
  --input "${ROOT_DIR}/questions.json" \
  --output "${ROOT_DIR}/questions_ba.json"

printf "job_id\torder\tmodel\toutput_root\n" > "${JOBS_FILE}"

for model in "${TARGET_MODELS[@]}"; do
  ab_out="${ROOT_DIR}/outputs/ab_order_run_by_model/${model}"
  output="$("${ROOT_DIR}/scripts/submit_slurm.sh" "${ROOT_DIR}/slurm/run_ab_model.sbatch" "${model}" "${ab_out}")"
  job_id="$(printf "%s\n" "${output}" | awk '{print $4}')"
  printf "%s\tab\t%s\t%s\n" "${job_id}" "${model}" "${ab_out}" | tee -a "${JOBS_FILE}"
  printf "%s\n" "${output}"

  ba_out="${ROOT_DIR}/outputs/ba_order_run_by_model/${model}"
  output="$("${ROOT_DIR}/scripts/submit_slurm.sh" "${ROOT_DIR}/slurm/run_ba_model.sbatch" "${model}" "${ba_out}")"
  job_id="$(printf "%s\n" "${output}" | awk '{print $4}')"
  printf "%s\tba\t%s\t%s\n" "${job_id}" "${model}" "${ba_out}" | tee -a "${JOBS_FILE}"
  printf "%s\n" "${output}"
done

printf "AB/BA jobs written to %s\n" "${JOBS_FILE}"
