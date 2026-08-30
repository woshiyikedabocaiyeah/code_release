#!/usr/bin/env bash
# PCA on the 396 representation matrices, with the eta-squared decomposition of
# component scores by physical concept / plausibility / scene template (Fig. 6b).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== 00  eta-squared per component ========================================"
"$PY" 00_extract_pca_stats.py
