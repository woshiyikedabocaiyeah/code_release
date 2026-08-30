#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 slurm/job.sbatch [job-args...]" >&2
  exit 2
fi

JOB_SCRIPT="$1"
shift

SBATCH_ARGS=()
if [[ -n "${SBATCH_PARTITION:-}" ]]; then
  SBATCH_ARGS+=(--partition "${SBATCH_PARTITION}")
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account "${SBATCH_ACCOUNT}")
fi
if [[ -n "${SBATCH_GPUS:-}" ]]; then
  SBATCH_ARGS+=(--gpus "${SBATCH_GPUS}")
fi
if [[ -n "${SBATCH_GRES:-}" ]]; then
  SBATCH_ARGS+=(--gres "${SBATCH_GRES}")
fi

exec sbatch "${SBATCH_ARGS[@]}" "${JOB_SCRIPT}" "$@"
