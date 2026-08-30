#!/usr/bin/env python3
"""Panels d and f: posterior distributions of the drift-rate coefficients for
Visual_Z (d) and Physical_Z (f), RT_critical regression model.

Posterior draws come from M.combined_trace with the same TRACE_KEYS the
previous figures used, so the densities are identical to the published
version.  Presentation changes: the redundant histogram-under-KDE double
encoding is dropped and the legend is replaced by direct labels.  Whether a
coefficient excludes zero is the main thing this panel is read for; that is
read off the x-axis tick at 0.00 rather than from a drawn reference line.
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
from scipy.stats import gaussian_kde

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
    "visual_z": dict(coef=r"$v_{\mathrm{Visual_{Z}}}$", letter="d"),
    "physical_z": dict(coef=r"$v_{\mathrm{Physical_{Z}}}$", letter="f"),
}

RT_KEY = "rt_critical"


@lru_cache(maxsize=None)
def draws(feature: str, code: str) -> np.ndarray:
    return M.combined_trace(RT_KEY, M.TRACE_KEYS[feature][code])


def build(feature, fig=None, ax=None, show_key=True):
    """show_key=False drops the in-panel colour key.

    The composite already keys the three tasks once, in panel b; repeating the
    key in d and f would spend three lines of each cell restating it.  The
    standalone files keep it, since they must be readable on their own.
    """
    standalone = fig is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.46, 2.0))
    by_task = {OLD2NEW[c]: draws(feature, c) for c in M.GROUPS}
    allv = np.concatenate(list(by_task.values()))
    lo, hi = float(allv.min()), float(allv.max())
    pad = max((hi - lo) * 0.16, 0.02)
    xs = np.linspace(lo - pad, hi + pad, 600)

    peak = 0.0
    for task in fs.TASK_ORDER:
        col = fs.TASK_COL[task]
        d = by_task[task]
        ys = gaussian_kde(d)(xs)
        peak = max(peak, float(ys.max()))
        ax.fill_between(xs, 0, ys, color=fs.TASK_FILL[task], alpha=0.50,
                        lw=0, zorder=2)
        ax.plot(xs, ys, color=col, lw=1.4, zorder=3)
        ax.axvline(float(d.mean()), color=col, lw=0.8, ls=(0, (3, 2)),
                   alpha=0.9, zorder=4)

    # No zero reference line: the x-axis tick at 0.00 already locates zero, and
    # the extra vertical competed visually with the three dashed posterior-mean
    # lines it sat among.
    ax.set_xlabel(f"{FEATURES[feature]['coef']} coefficient")
    # A rotated y label cannot be shorter than the text it holds, so in the
    # composite "Posterior density" renders 117 px against a 71 px axes and
    # necessarily reaches down into the x tick row.  The caption states these
    # are posterior densities, so the composite carries the short form.
    ax.set_ylabel("Posterior density" if standalone else "Density")
    ax.set_xlim(xs.min(), xs.max())
    head = 1.34 if show_key else 1.08
    ax.set_ylim(0, peak * head)
    # Drop ticks the locator kept above the new limit: in the composite they
    # are still drawn, land outside the axes, and collide with the row above.
    top = peak * head
    keep = [v for v in ax.get_yticks() if 0 <= v <= top]
    ax.set_yticks(keep)

    # The three task names total ~60 characters, which cannot be placed side
    # by side at single-column width -- an attempt to stagger them
    # horizontally made all three collide.  They are stacked in the empty
    # upper-centre region instead, each preceded by a short colour swatch that
    # carries the association to its curve.
    # Fixed order, not sorted by posterior mean: sorting made panels d and f
    # list the three tasks in different sequences, which reads as a difference
    # between the panels rather than a property of the estimates.  The colour
    # swatch, not the row position, carries the identity.
    rows = list(fs.TASK_ORDER) if show_key else []
    x0 = xs.min() + (xs.max() - xs.min()) * 0.30
    for i, task in enumerate(rows):
        y = peak * (1.25 - 0.105 * i)
        ax.plot([x0, x0 + (xs.max() - xs.min()) * 0.045], [y, y],
                color=fs.TASK_COL[task], lw=1.6, solid_capstyle="round",
                zorder=7, clip_on=False)
        ax.annotate(task, xy=(x0 + (xs.max() - xs.min()) * 0.058, y),
                    ha="left", va="center", fontsize=fs.MICRO,
                    color=fs.TASK_COL[task], zorder=7, path_effects=fs.HALO)
    return fig, ax, by_task


def main():
    fs.apply(mpl, dpi=500)
    stats = []
    for feature, cfg in FEATURES.items():
        fig, ax, by_task = build(feature)
        fs.letter(ax, cfg["letter"])
        stem = OUT_DIR / f"panel_{cfg['letter']}_{feature}_posterior_coefs"
        w, h = fs.save_at_width(fig, stem, mm=88)
        ov = fs.check_text_overlaps(fig)
        print(f"panel {cfg['letter']} ({feature}): {w:.2f} x {h:.2f} mm | overlaps: {len(ov)} {ov[:3]}")
        for task, d in by_task.items():
            p_gt0 = float(np.mean(d > 0))
            stats.append((feature, task, d.mean(), *np.percentile(d, [2.5, 97.5]), p_gt0))
        plt.close(fig)
    print()
    for row in stats:
        print(f"{row[0]:11s} {row[1]:24s} mean={row[2]:+.4f}  95% CI [{row[3]:+.4f}, {row[4]:+.4f}]  P(b>0)={row[5]:.3f}")


if __name__ == "__main__":
    main()
