#!/usr/bin/env bash
set -euo pipefail

# Rebuild the Apple Silicon environment used for this project.
# Assumes Homebrew and pyenv are installed.

brew list gcc >/dev/null 2>&1 || brew install gcc
pyenv install -s 3.9.25

PYENV_VERSION=3.9.25 python -m venv .venv39
.venv39/bin/python -m pip install 'pip<26' 'setuptools<70' wheel
.venv39/bin/python -m pip install -r requirements-hddm.txt

SDK_PATH="$(xcrun --show-sdk-path)"
SDKROOT="$SDK_PATH" \
FFLAGS='-fallow-argument-mismatch' \
FCFLAGS='-fallow-argument-mismatch' \
LDFLAGS="-L${SDK_PATH}/usr/lib -isysroot ${SDK_PATH}" \
PIP_USE_PEP517=0 \
.venv39/bin/python -m pip install --no-build-isolation 'pymc==2.3.8'

PIP_USE_PEP517=0 \
.venv39/bin/python -m pip install --no-build-isolation 'kabuki==0.6.5'

.venv39/bin/python - <<'PY'
from pathlib import Path
import kabuki

path = Path(kabuki.__file__).with_name("utils.py")
text = path.read_text()
old = "def flatten(l):\n    return reduce(lambda x, y: list(x) + list(y), l)\n"
new = "def flatten(l):\n    if len(l) == 0:\n        return []\n    return reduce(lambda x, y: list(x) + list(y), l)\n"
if old in text:
    path.write_text(text.replace(old, new))
PY

PIP_USE_PEP517=0 \
.venv39/bin/python -m pip install --no-build-isolation --no-deps 'ssm-simulators==0.3.2'

SDKROOT="$SDK_PATH" \
FFLAGS='-fallow-argument-mismatch' \
FCFLAGS='-fallow-argument-mismatch' \
LDFLAGS="-L${SDK_PATH}/usr/lib -isysroot ${SDK_PATH}" \
PIP_USE_PEP517=0 \
.venv39/bin/python -m pip install --no-build-isolation --no-deps ./vendor/HDDM-1.0.1

.venv39/bin/python - <<'PY'
import hddm
print("HDDM", hddm.__version__, "is ready.")
PY
