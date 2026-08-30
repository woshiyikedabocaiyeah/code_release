"""Publication-style helpers for HDDM manuscript figures."""

from __future__ import annotations

import os
from pathlib import Path


TASK_LABELS = {
    "Semantic": "Concept Verification",
    "Intuitive": "Plausibility Assessment",
    "Voe": "Plausibility Assessment",
    "Action": "Affordance Recognition",
}

TASK_LABELS_WRAPPED = {
    "Semantic": "Concept\nVerification",
    "Intuitive": "Plausibility\nAssessment",
    "Voe": "Plausibility\nAssessment",
    "Action": "Affordance\nRecognition",
}

TASK_FILE_LABELS = {
    "Semantic": "concept_verification",
    "Intuitive": "plausibility_assessment",
    "Voe": "plausibility_assessment",
    "Action": "affordance_recognition",
}

# Okabe-Ito palette: colorblind-safe and still close to the existing mapping.
TASK_COLORS = {
    "Semantic": "#0072B2",
    "Intuitive": "#009E73",
    "Voe": "#009E73",
    "Action": "#D55E00",
}

VISUAL_Z = r"$\mathrm{Visual}_{z}$"
PHYSICAL_Z = r"$\mathrm{Physical}_{z}$"
RT_ONSET = r"RT$_{onset}$"
RT_CRITICAL = r"RT$_{critical}$"


def is_academic_mode() -> bool:
    return os.environ.get("HDDM_ACADEMIC_FIGURES") == "1"


def apply_academic_style() -> None:
    """Apply a Nature-like style from the scientific-visualization skill."""
    if not is_academic_mode():
        return

    import matplotlib as mpl
    import matplotlib.style as mplstyle

    skill_dir = Path(
        os.environ.get(
            "SCIENTIFIC_VISUALIZATION_SKILL",
            "skills/scientific-visualization",
        )
    )
    style_path = skill_dir / "assets" / "nature.mplstyle"
    if style_path.exists():
        mplstyle.use(str(style_path))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def install_academic_savefig_patch() -> None:
    """Raise raster export DPI in academic mode without changing every script."""
    if not is_academic_mode():
        return

    from matplotlib.figure import Figure

    if getattr(Figure.savefig, "_hddm_academic_patch", False):
        return

    original_savefig = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        suffix = Path(str(fname)).suffix.lower()
        if suffix in {".png", ".tif", ".tiff"}:
            dpi = kwargs.get("dpi", None)
            if dpi is None or float(dpi) < 600:
                kwargs["dpi"] = 600
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("pad_inches", 0.04)
        kwargs.setdefault("facecolor", "white")
        return original_savefig(self, fname, *args, **kwargs)

    savefig._hddm_academic_patch = True
    Figure.savefig = savefig


def update_module_style(module, wrapped_labels: bool = False) -> None:
    """Update common plotting constants on imported plot modules."""
    if not is_academic_mode():
        return

    if hasattr(module, "COLORS"):
        for key, color in TASK_COLORS.items():
            if key in module.COLORS:
                module.COLORS[key] = color

    if hasattr(module, "DISPLAY_LABELS"):
        labels = TASK_LABELS_WRAPPED if wrapped_labels else TASK_LABELS
        for key, label in labels.items():
            if key in module.DISPLAY_LABELS:
                module.DISPLAY_LABELS[key] = label

    if hasattr(module, "FILE_LABELS"):
        for key, label in TASK_FILE_LABELS.items():
            if key in module.FILE_LABELS:
                module.FILE_LABELS[key] = label
