#!/usr/bin/env bash
# RSA: model representations against human behavioural geometry.  Usage: bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== 01  permutation tests for all 9 human RDMs (~70 s) =================="
"$PY" 01_regenerate_permutation_nulls.py

echo; echo "== 00  extract the reported statistics ================================="
"$PY" 00_extract_rsa_stats.py

echo; echo "== 03  verify the manuscript prose against the data ===================="
"$PY" 03_check_text_vs_data.py
