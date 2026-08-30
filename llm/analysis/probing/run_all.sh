#!/usr/bin/env bash
# Layer-wise linear probing, with its controls.  Usage: bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== 01  probes under video-grouped cross-validation (~4 min) ============"
"$PY" 01_rerun_probes_grouped_cv.py --jobs 8

echo; echo "== 02  nearest-neighbour geometry, with the raw-pixel baseline ========="
"$PY" 02_plausibility_geometry.py

if [ "${SKIP_PERM:-0}" = "1" ] && [ -f derived/permutation_control.csv ]; then
  echo; echo "== 03  label-permutation control ............... skipped (SKIP_PERM=1)"
else
  echo; echo "== 03  label-permutation control (~25 min) ============================="
  "$PY" 03_permutation_control.py --perms 200 --jobs 8
fi

echo; echo "== 08  architecture control ============================================="
"$PY" 08_architecture_control.py

echo; echo "== 09  probe capacity diagnostics ======================================="
"$PY" 09_probe_capacity_diagnostics.py

echo; echo "== 05  verify the manuscript prose against the data ====================="
"$PY" 05_check_text_vs_data.py
