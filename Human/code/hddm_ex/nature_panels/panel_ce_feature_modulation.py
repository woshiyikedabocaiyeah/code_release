#!/usr/bin/env python3
"""Panels c and e: behavioural modulation by Visual_Z (c) and Physical_Z (e).

Each panel row is three sub-axes: RT_onset, RT_critical, accuracy.  The
statistics are NOT refitted here -- the LMM fits and per-group slopes come
straight from make_hddm_feature_modulation_figures, exactly as the previous
Nature-quality version used them.
"""

from __future__ import annotations

import sys
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT / "nature"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR / "manuscript_common"))
sys.path.insert(0, str(CODE_DIR))
import figure_style as fs  # noqa: E402
import make_hddm_feature_modulation_figures as M  # noqa: E402

OLD2NEW = {
    "Semantic": "Concept Verification",
    "Intuitive": "Plausibility Assessment",
    "Action": "Affordance Recognition",
}

FEATURES = {
    "Visual_Z": dict(covariate="Physical_Z", xlabel=fs.LBL_VISUAL, letter="c"),
    "Physical_Z": dict(covariate="Visual_Z", xlabel=fs.LBL_PHYSICAL, letter="e"),
}

MEASURES = [
    ("RT_onset", fs.LBL_RT_ONSET + " (s)", True, None),
    ("RT_critical", fs.LBL_RT_CRIT + " (s)", True, None),
    # ACC y-limits are derived from the data (see dyn_ylim), not fixed at
    # (72, 100): the Affordance Recognition fit reaches ~66.7%, and the old
    # fixed window clipped that line off the bottom of the axes.
    ("ACC", "Accuracy (%)", False, None),
]

# Slopes are reported on the same scale as the plotted curves.  The LMM
# `estimate` for ACC is on the proportion scale while the curves are drawn as
# percentages, so it must be multiplied by the same prediction_scale (100) --
# otherwise the panel shows "b = 0.086" beside a line spanning 26 percentage
# points.  These scaled values are the AME figures reported in the text.
SLOPE_SCALE = {"RT_onset": 1.0, "RT_critical": 1.0, "ACC": 100.0}

# Ink budget for an in-axes slope block, counted as scatter points plus fitted
# curve samples inside the corner box.  A few stray scatter points are fine
# behind a haloed label; anything more means a fitted band runs through the
# corner, and the block goes outside the axes instead.
MAX_CORNER_INK = 8

#: Populated by draw_measure as a record of what the axis floor excluded:
#: (feature, outcome) -> (n_below_axis, n_points).  Inspect it after a build to
#: see what a panel hid.  The caption does NOT read this -- it recomputes the
#: counts from dyn_ylim so caption generation never depends on rendering.
CLIPPED_NOTES: dict[tuple[str, str], tuple[int, int]] = {}


@lru_cache(maxsize=None)
def raw_behavior():
    return M.load_raw_behavior()


@lru_cache(maxsize=None)
def fit(feature: str, outcome: str):
    """Cached LMM fit; identical call signature to the previous version."""
    cov = FEATURES[feature]["covariate"]
    if outcome == "ACC":
        curves, slopes = M.fit_lmm_feature_effect(
            raw_behavior(), "ACC", feature, cov, prediction_scale=100.0)
    else:
        curves, slopes = M.fit_lmm_feature_effect(
            raw_behavior(), outcome, feature, cov, filter_rt=True)
    return {OLD2NEW[k]: v for k, v in curves.items()}, \
           {OLD2NEW[k]: v for k, v in slopes.items()}


@lru_cache(maxsize=None)
def scatter(feature: str, outcome: str):
    if outcome == "ACC":
        df = M.build_empirical_feature_points(raw_behavior(), feature, "ACC",
                                              value_scale=100.0)
    else:
        df = M.build_empirical_feature_points(raw_behavior(), feature, outcome,
                                              filter_rt=True)
    df = df.copy()
    df["group"] = df["group"].map(OLD2NEW)
    return df


def dyn_ylim(sc, curves, pad_fraction=0.12, floor_q=None):
    """Data-driven y-limits.

    The fitted mean/CI bands are always fully included -- clipping a fit line
    at the axis edge misrepresents the model.  `floor_q` optionally excludes a
    lower tail of the scatter from the limit calculation, for accuracy panels
    where a handful of near-floor subject means would otherwise compress the
    whole plot; the excluded count is reported so it can be annotated.
    """
    pts = sc["value"].to_numpy(float)
    band = np.concatenate([np.asarray(curves[t][k], float)
                           for t in fs.TASK_ORDER
                           for k in ("mean", "low", "high")])
    kept = pts
    n_clipped = 0
    if floor_q is not None:
        cut = float(np.nanquantile(pts, floor_q))
        kept = pts[pts >= cut]
        n_clipped = int((pts < cut).sum())
    allv = np.concatenate([kept, band])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    pad = (hi - lo) * pad_fraction if hi > lo else 0.5
    return (lo - pad * 0.45, hi + pad), n_clipped


def draw_measure(ax, feature, outcome, ylabel, ylim, x_grid, show_ylabel=True):
    sc = scatter(feature, outcome)
    curves, slopes = fit(feature, outcome)

    for task in fs.TASK_ORDER:
        col = fs.TASK_COL[task]
        g = sc[sc["group"] == task]
        ax.scatter(g["x"], g["value"], s=3.0, color=col, alpha=0.55,
                   edgecolor="none", zorder=2, rasterized=True)
        cx = curves[task].get("x", x_grid)
        ax.fill_between(cx, curves[task]["low"], curves[task]["high"],
                        color=col, alpha=0.16, lw=0, zorder=3)
        ax.plot(cx, curves[task]["mean"], color=col, lw=1.5, zorder=4)

    ax.set_xlabel(FEATURES[feature]["xlabel"])
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(ylabel)
    ax.set_xlim(float(x_grid.min()) - 0.05, float(x_grid.max()) + 0.05)
    # Explicit ticks: an automatic tick at the extreme left (x = -2) lands in
    # the corner and collides with the lowest y tick label.
    ax.set_xticks([-1.0, 0.0, 1.0])
    if ylim:
        ax.set_ylim(*ylim)
        n_clipped = 0
    else:
        # Accuracy: exclude the bottom 4% of subject means from the limit
        # calculation (a few near-floor performers), never the fitted bands.
        lims, n_clipped = dyn_ylim(sc, curves,
                                   floor_q=0.04 if outcome == "ACC" else None)
        ax.set_ylim(*lims)
    # Both the slope block and the clipped-point note need empty space, and in
    # the composite the cells are narrow enough that a bottom-left note reaches
    # a bottom-right label.  Rank the corners once by data ink: the slope block
    # takes the emptiest, the note the next emptiest.
    # The clipped-point count is NOT drawn on the panel.  It is metadata about
    # the axis limits, and every on-panel position for it fails in the narrow
    # composite cells: the corners are taken by data ink or the slope block, and
    # below the axes it lands on the x label.  It is reported in the caption
    # instead: generate_figure_1_caption.py recomputes it from dyn_ylim, so
    # removing the annotation does not drop the disclosure.
    if n_clipped:
        CLIPPED_NOTES[(feature, outcome)] = (n_clipped, len(sc))
    return slopes, sc, curves


def pick_corner(ax, sc, curves, x_grid, box=(0.34, 0.34)):
    """Choose the corner with the least data ink for the slope-label block.

    check_text_overlaps only compares text against text, so a label can pass
    that check while sitting directly on a fitted line (as -5.38 pp did on the
    Physical_Z accuracy panel).  Occupancy is counted in axes-fraction space
    over the scatter and the fitted curves.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    def to_frac(xv, yv):
        return ((np.asarray(xv, float) - x0) / (x1 - x0),
                (np.asarray(yv, float) - y0) / (y1 - y0))

    sx, sy = to_frac(sc["x"], sc["value"])
    cxs, cys = [], []
    for task in fs.TASK_ORDER:
        cx = np.asarray(curves[task].get("x", x_grid), float)
        for key in ("mean", "low", "high"):
            gx, gy = to_frac(cx, curves[task][key])
            cxs.append(gx); cys.append(gy)
    cxs = np.concatenate(cxs); cys = np.concatenate(cys)
    # Scatter and fitted-curve ink are kept apart: a label sitting on a fitted
    # band is a mild aesthetic cost, but a scatter point peeking out from under
    # a haloed glyph reads as a typo (an orange point under "b" rendered as
    # "b,"), so the two are weighted differently downstream.
    bands = []
    for task in fs.TASK_ORDER:
        cx = np.asarray(curves[task].get("x", x_grid), float)
        bx, blo = to_frac(cx, curves[task]["low"])
        _, bhi = to_frac(cx, curves[task]["high"])
        bands.append((bx, blo, bhi))
    ink = {"scatter": (sx, sy), "curve": (cxs, cys), "bands": bands}
    fx = np.concatenate([sx, cxs]); fy = np.concatenate([sy, cys])

    # The corner box is sized in millimetres of rendered axes, not as a fixed
    # axes fraction: the same fraction covers very different amounts of data
    # when the cell is 55 mm wide (standalone) versus 34 mm (composite), so a
    # fraction-based box picked corners that were empty at one width and full
    # of fitted lines at the other.
    bb = ax.get_window_extent()
    dpi = ax.figure.dpi
    w_mm = bb.width / dpi * 25.4
    h_mm = bb.height / dpi * 25.4
    bw = min(0.46, max(0.30, 19.0 / w_mm)) if w_mm > 0 else box[0]
    bh = min(0.46, max(0.30, 11.0 / h_mm)) if h_mm > 0 else box[1]
    boxes = {
        "upper_left": (0.0, bw, 1.0 - bh, 1.0),
        "upper_right": (1.0 - bw, 1.0, 1.0 - bh, 1.0),
        "lower_left": (0.0, bw, 0.0, bh),
        "lower_right": (1.0 - bw, 1.0, 0.0, bh),
    }
    counts = {}
    for name, (bx0, bx1, by0, by1) in boxes.items():
        inside = (fx >= bx0) & (fx <= bx1) & (fy >= by0) & (fy <= by1)
        counts[name] = int(inside.sum())
    order = list(boxes)
    best = min(counts, key=lambda k: (counts[k], order.index(k)))
    # If even the emptiest corner carries data ink, no in-axes position is
    # clean (this happens in the composite's narrow accuracy cells, where the
    # fitted lines cross near the top right).  The caller then places the
    # slope block outside the axes rather than on top of a fitted band.
    if counts[best] > MAX_CORNER_INK:
        return "outside", counts, ink
    return best, counts, ink


def label_slopes(ax, slopes, corner, scale=1.0, unit="", ink=None):
    """Slope values as direct annotations, on the same scale as the curves."""
    # Anchors are axes fractions, so a fixed value only clears the spines at
    # one axes height.  Tuned against the 1.50 in standalone axes, y=0.95/0.26
    # put the 3-line stack across the top and bottom spines in the composite,
    # where the cell is 0.71 in tall and the same 6 pt line spans twice the
    # fraction.  Derive the inset from the measured height instead.
    step = -0.105
    # Measure a real line rather than deriving from the nominal point size:
    # a 6 pt line with ascenders and a halo renders ~2x its point height, so
    # the point-size estimate left the stack touching both spines.
    h_in = ax.get_window_extent().height / ax.figure.dpi
    probe = ax.text(0, 0, "$b$ = -0.00 pp", transform=ax.transAxes,
                    fontsize=fs.MICRO, va="center")
    ax.figure.canvas.draw()
    half = probe.get_window_extent().height / ax.figure.dpi / max(h_in, 1e-6) / 2.0
    probe.remove()
    pad = 1.10 * half
    y_top = 1.0 - half - pad
    y_bot = 2.0 * abs(step) + half + pad
    anchors = {
        "upper_left": (0.03, y_top, "left"),
        "upper_right": (0.97, y_top, "right"),
        "lower_left": (0.03, y_bot, "left"),
        "lower_right": (0.97, y_bot, "right"),
        # Just outside the right spine, stacked downward from the top.
        "outside": (1.03, y_top, "left"),
    }
    x, y0, ha = anchors[corner]
    box = None
    wide_halo = False
    lines = []
    for i, task in enumerate(["Affordance Recognition", "Plausibility Assessment",
                              "Concept Verification"]):
        b = slopes[task]["estimate"] * scale
        fmt = f"{b:+.2f}" if scale != 1.0 else f"{b:.3f}"
        lines.append((task, f"$b$ = {fmt}{unit}", y0 + i * step))

    # A corner that passes pick_corner can still hold one or two stray scatter
    # points, and MAX_CORNER_INK deliberately tolerates them.  The halo then
    # erases only part of such a point and leaves a crescent touching a glyph:
    # in the composite an orange point under the "b" of the Physical_Z
    # RT_critical cell read as "b,".  Nudging the block by a few fixed steps was
    # not enough (it failed whenever every offset still touched a fitted band),
    # so search the axes for the position that is free of scatter, breaking ties
    # towards little curve ink and towards the corner pick_corner chose.
    if ink is not None and corner != "outside":
        probes = [ax.text(x, yy, txt, transform=ax.transAxes, fontsize=fs.MICRO,
                          ha=ha, va="center") for _, txt, yy in lines]
        ax.figure.canvas.draw()
        rend = ax.figure.canvas.get_renderer()
        inv = ax.transAxes.inverted()
        boxes = []
        for p in probes:
            bb = p.get_window_extent(rend)
            (fx0, fy0) = inv.transform((bb.x0, bb.y0))
            (fx1, fy1) = inv.transform((bb.x1, bb.y1))
            boxes.append((fx0, fy0, fx1, fy1))
        for p in probes:
            p.remove()
        bw = max(b[2] - b[0] for b in boxes)
        bh = max(b[3] - b[1] for b in boxes)
        stack_h = (max(b[3] for b in boxes) - min(b[1] for b in boxes))
        # Halo half-width converted to axes fractions on each axis.
        abb = ax.get_window_extent()
        pad_x = 2.0 / 72.0 * ax.figure.dpi / max(abb.width, 1e-6)
        pad_y = 2.0 / 72.0 * ax.figure.dpi / max(abb.height, 1e-6)

        sx, sy = ink["scatter"]
        cx_, cy_ = ink["curve"]
        bands = ink.get("bands", [])

        def band_overlap(ax0, ay0, ax1, ay1):
            """Does the box overlap the interior of any confidence band?

            Counting only the band's upper and lower edges is not enough: a box
            placed between them reports zero curve ink yet a white background
            would punch a hole in the shaded band.
            """
            n = 0
            for bx, blo, bhi in bands:
                m = (bx >= ax0) & (bx <= ax1)
                if not m.any():
                    continue
                n += int(((bhi[m] >= ay0) & (blo[m] <= ay1)).sum())
            return n

        def score(x_left, y_low):
            """(scatter, curve) ink under the block anchored at its lower-left."""
            ns = nc = 0
            for (bx0, by0, bx1, by1) in boxes:
                ax0 = x_left + (bx0 - min(b[0] for b in boxes))
                ay0 = y_low + (by0 - min(b[1] for b in boxes))
                ax1 = ax0 + (bx1 - bx0)
                ay1 = ay0 + (by1 - by0)
                ns += int(((sx >= ax0 - pad_x) & (sx <= ax1 + pad_x) &
                           (sy >= ay0 - pad_y) & (sy <= ay1 + pad_y)).sum())
                nc += int(((cx_ >= ax0 - pad_x) & (cx_ <= ax1 + pad_x) &
                           (cy_ >= ay0 - pad_y) & (cy_ <= ay1 + pad_y)).sum())
                nc += band_overlap(ax0 - pad_x, ay0 - pad_y, ax1 + pad_x, ay1 + pad_y)
            return ns, nc

        x_left0 = min(b[0] for b in boxes)
        y_low0 = min(b[1] for b in boxes)
        # Keep the block inside the spines with the same visual inset as the
        # hand-tuned anchors (0.03 of the axes width, one half-line vertically).
        xs = np.linspace(0.03, max(0.03, 1.0 - bw - 0.03), 21)
        # The lower bound used to be pad_y + bh/2, which excluded the bottom of
        # the axes: in the Visual_Z accuracy cell the only curve-free position
        # is flush with the lower spine (y_frac 0.00), and clipping it away made
        # every candidate overlap the three regression lines, so the block
        # stayed in a corner where the halo cut a break in the orange line.
        ys = np.linspace(pad_y, max(pad_y, 1.0 - stack_h - pad_y), 25)
        cands = []
        for xv in xs:
            for yv in ys:
                ns, nc = score(xv, yv)
                d = abs(xv - x_left0) + abs(yv - y_low0)
                # Curve ink is ranked first: scatter under the block can be
                # occluded by an opaque background without loss (the points are
                # a cloud), whereas a gap in a fitted line reads as missing
                # data and cannot be repaired by any background choice.
                cands.append((nc, ns, d, xv, yv))
        nc, ns, _, xv, yv = min(cands)
        ns0, nc0 = score(x_left0, y_low0)
        if (nc, ns) < (nc0, ns0):
            dx = xv - x_left0
            dy = yv - y_low0
            lines = [(t_, s_, yy + dy) for t_, s_, yy in lines]
            x = x + dx
        else:
            ns, nc = ns0, nc0
        # In the composite's 34 mm cells the two RT columns have no scatter-free
        # position at all (measured minima: 1 and 2 points under the 3-line
        # block), and a halo only erases part of a point, leaving a crescent
        # against the glyph.  Where the block sits clear of the fitted curves,
        # an opaque label background removes those points cleanly -- the same
        # occlusion any legend box makes.
        #
        # A background is never used when curve ink is unavoidable: in the
        # Visual_Z accuracy cell the three regression lines cross the whole
        # frame, so the block overlaps ~90 curve samples wherever it is placed,
        # and both an opaque box and the white halo cut a visible break in the
        # orange line that misreads as missing data.  There the curves are
        # only remaining lever is to shrink the block until a curve-free
        # position exists; see the search over box widths below.
        # The background is decided by what the chosen position actually sits
        # on, not by which branch chose it: after curve ink was ranked first,
        # the Physical_Z accuracy block moved to a curve-free spot that still
        # holds one scatter point, and without a background its halo again left
        # a crescent reading as "b,".
        if ns > 0 and nc == 0:
            box = dict(facecolor="white", edgecolor="none", pad=0.8)
        elif ns > 0 and nc > 0:
            # Both unavoidable (the Physical_Z accuracy cell: the three bands
            # cover the bottom of the frame, so every curve-free candidate is
            # ruled out, and one scatter point stays under the block wherever
            # it goes).  An opaque box is rejected here because it would break
            # a fitted line, which reads as missing data, whereas the offending
            # point sits at the glyph's outer edge: widening the halo covers it
            # without touching the lines, since a stroke follows the glyph
            # outline instead of filling a rectangle.
            wide_halo = True

    for task, txt, yy in lines:
        ax.text(x, yy, txt,
                transform=ax.transAxes, fontsize=fs.MICRO,
                color=fs.TASK_COL[task], ha=ha, va="center", zorder=7,
                clip_on=False, bbox=box,
                path_effects=None if box else
                (fs.WIDE_HALO if wide_halo else fs.HALO))


def build(feature, fig=None, axes=None, force_in_axes=False):
    """force_in_axes: never place the slope block outside the right spine.

    The "outside" fallback is correct for the standalone panel, where the axes
    is 1.87 in wide and a dense scatter can genuinely fill every corner.  In
    the composite the same cell is ~1.07 in wide, and a block at x=1.03 with
    clip_on=False makes constrained_layout reserve 0.37-0.49 in of width for
    it -- shrinking the axes by ~27% relative to the row above and pushing the
    text into the neighbouring cell.  A haloed label over a few scatter points
    is the better trade there: measured, one label spans 35% of the axes width
    and three stacked spans 31% of its height, so they do fit inside.
    """
    standalone = fig is None
    if standalone:
        # constrained layout, not subplots_adjust: save_at_width re-renders
        # with bbox_inches="tight" while iterating on the target width, which
        # discards manual subplot spacing and lets tick labels collide.
        fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.05), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.055, wspace=0.045)
    df = raw_behavior()
    x_grid = np.linspace(float(df[feature].min()), float(df[feature].max()), 220)

    pending = []
    for k, (outcome, ylabel, _, ylim) in enumerate(MEASURES):
        slopes, sc, curves = draw_measure(
            axes[k], feature, outcome, ylabel, ylim, x_grid)
        pending.append((axes[k], outcome, slopes, sc, curves, x_grid))

    # Corner choice needs the final axes size, which constrained layout only
    # fixes at draw time.  Placing the labels before that made the composite
    # reuse the standalone layout's occupancy and drop two slope labels onto
    # the fitted bands in the narrower cells.
    fig.canvas.draw()
    for ax, outcome, slopes, sc, curves, xg in pending:
        corner, counts, ink = pick_corner(ax, sc, curves, xg)
        if force_in_axes and corner == "outside":
            corner = min(counts, key=counts.get)
        label_slopes(ax, slopes, corner, ink=ink, scale=SLOPE_SCALE[outcome],
                     unit=" pp" if outcome == "ACC" else "")
        print(f"    {outcome:12s} label corner: {corner:12s} occupancy={counts}")
    return fig, axes


def main():
    fs.apply(mpl, dpi=500)
    for feature, cfg in FEATURES.items():
        fig, axes = build(feature)
        fs.letter(axes[0], cfg["letter"])
        stem = OUT_DIR / f"panel_{cfg['letter']}_{feature.lower()}_modulation"
        w, h = fs.save_at_width(fig, stem, mm=180)
        ov = fs.check_text_overlaps(fig)
        ink = fs.check_ink_under_text(fig)
        print(f"panel {cfg['letter']} ({feature}): {w:.2f} x {h:.2f} mm "
              f"| overlaps: {len(ov)} {ov[:3]} | ink under labels: {len(ink)} {ink[:3]}")
        plt.close(fig)


if __name__ == "__main__":
    main()
