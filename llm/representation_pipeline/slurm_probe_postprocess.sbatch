#!/usr/bin/env bash
#SBATCH --job-name=yesno-probe-post
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=$PROJECT_ROOT/analysis/review_package/Logs/%x-%j.out
#SBATCH --error=$PROJECT_ROOT/analysis/review_package/Logs/%x-%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-mllm}"

PACKAGE_ROOT="$PROJECT_ROOT/analysis/review_package"
PROBE_ROOT="${PACKAGE_ROOT}/Probe"

mkdir -p "${PACKAGE_ROOT}/Logs"

echo "[slurm-post] rebuilding probe derived tables, figures, report, and schema comparison"

conda run -n "${CONDA_ENV}" python "${PROBE_ROOT}/Scripts/build_probe_derivatives.py"
conda run -n "${CONDA_ENV}" python "${PROBE_ROOT}/Scripts/build_probe_question_evidence.py"
conda run -n "${CONDA_ENV}" python "${PROBE_ROOT}/Scripts/build_probe_report.py"
conda run -n "${CONDA_ENV}" python "${PROBE_ROOT}/Scripts/build_tsne_figures.py" --scope task_conditioned
conda run -n "${CONDA_ENV}" python "${PACKAGE_ROOT}/Scripts/repair_schema_compatibility.py"
conda run -n "${CONDA_ENV}" python "${PACKAGE_ROOT}/Scripts/compare_review_package_schema.py"

echo "[slurm-post] complete"
