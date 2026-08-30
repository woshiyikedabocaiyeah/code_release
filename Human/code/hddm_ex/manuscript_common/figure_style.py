"""Shared figure style for the Nature Machine Intelligence submission.

Implements sections 1-4 of FIGURE_STYLE_GUIDE.md.  Scripts must not set their
own rcParams or colours:

    import figure_style as fs
    fs.apply(mpl, dpi=500)
    ...
    fs.save_at_width(fig, OUT / "figureN_name", mm=180)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np

# ---------------------------------------------------------------- type scale
# Five steps only.  Use fontsize=fs.ANNOT, never fontsize=7.
LETTER = 12.0     # panel letters a b c d, bold
PRIMARY = 9.0     # axis titles, panel titles
SECONDARY = 8.0   # tick labels, legends
ANNOT = 7.0       # numbers drawn inside the axes
MICRO = 6.0       # dense thumbnail grids only

# ------------------------------------------------------------------- palette
# Okabe-Ito: green-red pairings are near-indistinguishable under the most
# common form of colour-vision deficiency; this set is designed to avoid that.
BLUE = "#0072B2"
GREEN = "#009E73"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"

# Light companions for violin interiors / histogram fills.
FILL = {BLUE: "#9ecae8", GREEN: "#8fd8c2", VERMILLION: "#f2b48c",
        ORANGE: "#f2d492", PURPLE: "#e6bdd2"}

# Neutrals.  Never use pure black for text -- use INK.
INK = "#1a1a1a"
GREY = "#7a7a7a"
GREY_LIGHT = "#b0b0b0"
CEIL = "#d6d6d6"

# --------------------------------------------------------- semantic bindings
# One variable keeps one colour across every figure in the manuscript.
TASK_ORDER = ["Concept Verification", "Plausibility Assessment",
              "Affordance Recognition"]

TASK_COL = {"Concept Verification": BLUE,
            "Plausibility Assessment": GREEN,
            "Affordance Recognition": VERMILLION}

TASK_FILL = {k: FILL[v] for k, v in TASK_COL.items()}

# Internal condition codes -> manuscript names.  Figures must show the
# manuscript name; the codes exist only in the data files.
CODE_TO_TASK = {"categorization": "Concept Verification",
                "Voe": "Plausibility Assessment",
                "sensorimotor": "Affordance Recognition",
                "Category": "Concept Verification",
                "VoE": "Plausibility Assessment",
                "SM": "Affordance Recognition",
                "Semantic": "Concept Verification",
                "Intuitive": "Plausibility Assessment",
                "Action": "Affordance Recognition"}

MEASURE_COL = {"rt": BLUE, "rt_critical": GREEN, "corr": VERMILLION}

# --------------------------------------------------------------- text labels
# Mathtext for the outcome and feature names, so subscripts render as
# subscripts rather than as literal underscores.
LBL_RT_ONSET = r"$\mathrm{RT_{onset}}$"
LBL_RT_CRIT = r"$\mathrm{RT_{critical}}$"
LBL_VISUAL = r"$\mathrm{Visual_{Z}}$"
LBL_PHYSICAL = r"$\mathrm{Physical_{Z}}$"
LBL_ACC = "ACC (%)"

# White outline so annotations stay readable over any fill.
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]
# For annotations that cannot avoid a scatter point: a stroke follows the
# glyph outline, so widening it covers a point touching the glyph edge
# without the rectangular footprint of an opaque label background.
WIDE_HALO = [pe.withStroke(linewidth=3.2, foreground="white")]


def apply(mpl, dpi: int = 500) -> None:
    """Install the shared rcParams.  Call once, before creating any figure."""
    mpl.rcParams.update({
        # Sans-serif, embedded.  Type 3 (the default) cannot be embedded or
        # searched in many typesetting pipelines and gets bounced by the
        # journal, so fonttype 42 is mandatory.
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Mathtext must use the same family as body text, otherwise a panel
        # mixes Helvetica tick labels with DejaVu subscripted axis labels.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "mathtext.default": "regular",
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "axes.titlesize": PRIMARY,
        "axes.labelsize": PRIMARY,
        "xtick.labelsize": SECONDARY,
        "ytick.labelsize": SECONDARY,
        "legend.fontsize": SECONDARY,
        "font.size": SECONDARY,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
    })


def letter(ax, s: str, dx: float = -0.16, dy: float = 0.06):
    """Panel letter, placed outside the axes in axes-fraction coordinates so
    it does not move when the data range changes."""
    return ax.text(dx, 1 + dy, s, transform=ax.transAxes, fontsize=LETTER,
                   fontweight="bold", va="bottom", ha="right", color=INK)


# Trim padding.  mathtext underestimates the bbox of a trailing glyph such
# as the ")" in "time, t (s)": at 0.02 in the closing paren reached the
# very last pixel column of the file, so any trim tolerance would clip it.
_PAD_IN = 0.06


def _measure_mm(path) -> tuple[float, float]:
    """Physical size of a written raster, in mm."""
    from PIL import Image
    im = Image.open(path)
    dpi = im.info.get("dpi", (100.0, 100.0))[0]
    return im.width / float(dpi) * 25.4, im.height / float(dpi) * 25.4


def _iterate(fig, stem: Path, target_mm: float, axis: str,
             tol: float = 0.25, max_iter: int = 14) -> tuple[float, float]:
    """Rescale the canvas until the *written file* measures target_mm.

    bbox_inches="tight" crops to the drawn content and adds pad_inches on top,
    so the saved size is not figsize.  Worse, text is fixed in points and does
    not scale with the canvas, so the relationship is non-linear and a single
    correction does not converge.  Hence: measure the file, correct, repeat.

    Known limitation: if the widest element is fixed-point text (a legend line
    too long to wrap), the canvas shrinks while the text does not, eventually
    squeezing out the plotting area.  Always look at the finished figure.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    w_in, h_in = fig.get_size_inches()
    for _ in range(max_iter):
        fig.savefig(png, bbox_inches="tight", pad_inches=_PAD_IN)
        got = _measure_mm(png)
        cur = got[0] if axis == "w" else got[1]
        # The journal limit is a ceiling, not a nominal value: accept only
        # sizes at or below it.  A symmetric tolerance let the composite settle
        # at 180.09 mm against a 180 mm limit.
        if target_mm - tol <= cur <= target_mm:
            break
        # Damp the step: text does not shrink with the canvas, so the naive
        # correction overshoots.  Aim just inside the ceiling.
        scale = 1.0 + ((target_mm - tol * 0.5) / cur - 1.0) * 0.85
        w_in, h_in = w_in * scale, h_in * scale
        fig.set_size_inches(w_in, h_in)
    fig.savefig(png, bbox_inches="tight", pad_inches=_PAD_IN)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=_PAD_IN)
    return _measure_mm(png)


def save_at_width(fig, stem, mm: float = 180.0) -> tuple[float, float]:
    """Main figures: scale to an exact width.  Writes .pdf and .png."""
    return _iterate(fig, Path(stem), mm, "w")


def save_within(fig, stem, max_w: float = 180.0,
                max_h: float = 247.0) -> tuple[float, float]:
    """Supplementary figures: fit inside the 180 x 247 mm page box, taking
    whichever of width or height binds first."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", pad_inches=_PAD_IN)
    w, h = _measure_mm(png)
    if w / max_w >= h / max_h:
        return _iterate(fig, stem, max_w, "w")
    return _iterate(fig, stem, max_h, "h")


def check_text_overlaps(fig) -> list[tuple[str, str]]:
    """Return pairs of visible text objects whose bounding boxes intersect."""
    import matplotlib as _mpl
    fig.canvas.draw()
    items = [(t, t.get_window_extent()) for t in fig.findobj(_mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    return [(a.get_text(), b.get_text())
            for i, (a, ba) in enumerate(items)
            for b, bb in items[i + 1:] if ba.overlaps(bb)]

def check_ink_under_text(fig, allowed_colors=None, tol=30.0, pad=(2, 1),
                         match=lambda t: "$b$" in t.get_text()):
    """Report annotations that have foreign ink showing inside their own box.

    check_text_overlaps compares text against text only, so a data point
    peeking out from under a haloed glyph passes it: an orange scatter point
    beneath the "b" of a slope label rendered as "b," in the composite.  The
    test here is done on rendered pixels rather than geometry, so a point that
    is cleanly occluded by an opaque label background is correctly not
    reported, while a point whose halo-clipped crescent is still visible is.

    Anti-aliased glyph pixels are blends between white and the glyph colour, so
    they lie on the white->colour segment in RGB; distance to that segment
    separates them from any other ink.  All three task colours are allowed by
    default because a neighbouring line of the same stack is a different colour
    and falls inside the padded box.

    Returns [(axes_ylabel, text, n_foreign_pixels), ...] for flagged texts.
    """
    if allowed_colors is None:
        allowed_colors = list(TASK_COL.values())
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    w_px, h_px = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)
    cols = [np.asarray(mcolors.to_rgb(c)) * 255.0 for c in allowed_colors]
    px, py = pad
    flagged = []
    for ax in fig.axes:
        for t in ax.texts:
            if not match(t):
                continue
            bb = t.get_window_extent(rend)
            x0, x1 = int(bb.x0) - px, int(np.ceil(bb.x1)) + px
            y0, y1 = int(bb.y0) - py, int(np.ceil(bb.y1)) + py
            sub = buf[max(h_px - y1, 0):max(h_px - y0, 0), max(x0, 0):max(x1, 0)]
            if sub.size == 0:
                continue
            dists = []
            for col in cols:
                v = col - 255.0
                w = sub - 255.0
                tt = np.clip((w * v).sum(2) / (v * v).sum(), 0, 1)[..., None]
                dists.append(np.sqrt(((sub - (255.0 + tt * v)) ** 2).sum(2)))
            n = int((np.min(np.stack(dists), 0) > tol).sum())
            if n:
                flagged.append((ax.get_ylabel()[:24], t.get_text(), n))
    return flagged
