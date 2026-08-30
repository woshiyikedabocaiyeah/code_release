"""Filesystem layout of the data deposit.

Every analysis script resolves its inputs through this module. Point it at the
deposit before running anything::

    export NMI_DATA_ROOT=/path/to/the/data/deposit

``NMI_DATA_ROOT`` is the directory that contains ``stimuli/``, ``behaviour/``,
``representations/``, ``pca/``, ``probing/`` and ``alignment_with_human/``.

Results are written to ``NMI_OUT_ROOT`` when that is set, and otherwise to a
``derived/`` directory beside the script being run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ``model_naming`` lives beside this file; make it importable for any script
# that imports paths.
sys.path.insert(0, str(Path(__file__).resolve().parent))

_REQUIRED = ("stimuli", "behaviour", "representations", "alignment_with_human")


def _data_root() -> Path:
    raw = os.environ.get("NMI_DATA_ROOT")
    if not raw:
        raise SystemExit(
            "NMI_DATA_ROOT is not set.\n"
            "Point it at the data deposit, e.g.\n"
            "    export NMI_DATA_ROOT=/path/to/deposit"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"NMI_DATA_ROOT does not exist: {root}")
    missing = [d for d in _REQUIRED if not (root / d).is_dir()]
    if missing:
        raise SystemExit(
            f"NMI_DATA_ROOT={root} does not look like the deposit; "
            f"missing: {', '.join(missing)}"
        )
    return root


DATA = _data_root()

# --- stimuli ----------------------------------------------------------------
STIMULI = DATA / "stimuli"
VIDEOS = STIMULI / "videos"
VIDEO_METADATA = STIMULI / "video_metadata.csv"
QUESTIONS = STIMULI / "questions_144.json"

# --- model behaviour --------------------------------------------------------
BEHAVIOUR = DATA / "behaviour"
TRIAL_MANIFEST = BEHAVIOUR / "trial_manifest_6336.csv"
RAW_MODEL_OUTPUTS = BEHAVIOUR / "raw_model_outputs"
TASK_PERFORMANCE = BEHAVIOUR / "task_performance"
SAMPLING_STABILITY = BEHAVIOUR / "sampling_stability"

# --- model representations --------------------------------------------------
REPRESENTATIONS = DATA / "representations"
MAT = REPRESENTATIONS / "matrices_ready"
META = REPRESENTATIONS / "row_metadata"
RDM_MEAN_POOL = REPRESENTATIONS / "rdm_mean_pool"

# --- analyses ---------------------------------------------------------------
PCA_DIR = DATA / "pca"
PCA_RESULTS = PCA_DIR / "results"

PROBE = DATA / "probing"

ALIGNMENT = DATA / "alignment_with_human"
RSA_DIR = ALIGNMENT / "rsa"
HUMAN_ALIGNED = RSA_DIR / "human_aligned"
RT_CRITICAL = ALIGNMENT / "rt_critical"
NOISE_CEILING = ALIGNMENT / "noise_ceiling"
PERMUTATION_TEST = ALIGNMENT / "permutation_test"

RAW_ACTIVATIONS = DATA / "raw_activations"


def rt_critical_dir(derived: Path) -> Path:
    """Where the RT(critical) layer lives.

    The deposit ships this layer already built. ``05_build_rt_critical.py``
    rebuilds it from participant-level human data, which the deposit does not
    contain; when that script has been run, its output is used instead.
    """
    local = derived / "rt_critical"
    return local if (local / "rsa_pairwise_rt_critical.csv").is_file() else RT_CRITICAL


def derived_dir(script_file: str) -> Path:
    """Output directory for the analysis the given script belongs to."""
    here = Path(script_file).resolve().parent
    root = os.environ.get("NMI_OUT_ROOT")
    out = Path(root).expanduser().resolve() / here.name if root else here / "derived"
    out.mkdir(parents=True, exist_ok=True)
    return out


def section_md(default_name: str) -> Path | None:
    """Optional manuscript section used by the prose-versus-data checks."""
    raw = os.environ.get("NMI_SECTION_MD")
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_dir():
        p = p / default_name
    return p if p.is_file() else None
