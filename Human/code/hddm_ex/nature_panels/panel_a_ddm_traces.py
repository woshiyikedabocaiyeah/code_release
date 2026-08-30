#!/usr/bin/env python3
"""Panel a: DDM trajectory schematic per task condition (RT_critical).

Restyled to FIGURE_STYLE_GUIDE.md.  All quantities -- drift intercepts, mean
RTs, RT densities -- are read from the same fitted results the original
figures used; nothing is refitted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# --- path repair after reorganisation into _organized/code/ ---------------
# BASE_DIR: original project directory, still the source of data and the
# destination of figures. CODE_DIR: where this script and its sibling
# modules now live.
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "hddm_results_4chains_2000samples"
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT / "nature"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR / "manuscript_common"))
import figure_style as fs  # noqa: E402

# Internal code -> manuscript name, in the left-to-right order of the figure.
# Column order follows figure_style.TASK_ORDER so that the three tasks appear
# in the same left-to-right sequence in every panel of the composite figure.
_TRACE_CODES = ["Semantic", "Intuitive", "Action"]
CODES = sorted(_TRACE_CODES,
               key=lambda c: fs.TASK_ORDER.index(fs.CODE_TO_TASK[c]))
RT_KIND = "rt_critical"

X_START, X_END = 1.25, 6.75
X_LEFT, X_RIGHT = 0.72, 7.15
Y_TOP, Y_MID, Y_BOT = 1.0, 0.0, -1.0


def drift_intercept(rt_kind: str, code: str) -> float:
    """Posterior mean of the regression drift intercept, as reported."""
    summary = json.loads((RESULTS_DIR / rt_kind / "rt_summary.json").read_text())
    hyp = summary["models"]["regression"]["hypotheses"]["drift_intercepts"]
    return float(hyp[code.lower()]["mean"])


def map_rt_to_x(rt, x_start, x_end, rt_max):
    return x_start + (np.clip(rt, 0, rt_max) / rt_max) * (x_end - x_start)


def noisy_trajectory(x0, x1, y1, n_points, rng, noise_scale):
    xs = np.linspace(x0, x1, n_points)
    base = np.linspace(0.0, y1, n_points)
    noise = np.cumsum(rng.normal(0.0, noise_scale, size=n_points))
    noise = noise - np.linspace(0.0, noise[-1], n_points)
    ys = base + noise * np.sin(np.linspace(0.0, np.pi, n_points))
    ys[0], ys[-1] = 0.0, y1
    return xs, ys


def kde_band(ax, x_values, y_anchor, color, direction):
    if len(x_values) < 5:
        return
    xs = np.linspace(x_values.min(), x_values.max(), 400)
    ys = gaussian_kde(x_values)(xs)
    ys = ys / ys.max() * 0.13
    sign = 1.0 if direction == "up" else -1.0
    ax.fill_between(xs, y_anchor, y_anchor + sign * ys, color=color,
                    alpha=0.30, linewidth=0, zorder=4)
    ax.plot(xs, y_anchor + sign * ys, color=color, lw=1.0, zorder=4)


def draw_condition(ax, df, code, rng, wrap_title=False):
    task = fs.CODE_TO_TASK[code]
    col = fs.TASK_COL[task]
    d = df[df["group"] == code]
    correct = d.loc[d["response"] == 1, "rt"].to_numpy()
    error = d.loc[d["response"] == 0, "rt"].to_numpy()

    rt_max = max(float(np.quantile(d["rt"], 0.90)),
                 max(correct.mean(), error.mean()) * 1.35)
    cx = map_rt_to_x(correct, X_START, X_END, rt_max)
    ex = map_rt_to_x(error, X_START, X_END, rt_max)
    mcx, mex = float(cx.mean()), float(ex.mean())

    ax.hlines([Y_TOP, Y_MID, Y_BOT], X_LEFT, X_RIGHT, colors=fs.GREY_LIGHT,
              lw=0.9, zorder=1)

    # Pale individual trajectories, sampled across the observed RT quantiles.
    for rt in np.quantile(correct, np.linspace(0.10, 0.92, 14)):
        xs, ys = noisy_trajectory(X_START, map_rt_to_x(rt, X_START, X_END, rt_max),
                                  Y_TOP, 90, rng, 0.012)
        ax.plot(xs, ys, color=col, alpha=0.30, lw=0.7, zorder=2)
    n_err = min(12, len(error))
    for rt in np.quantile(error, np.linspace(0.10, 0.90, n_err)):
        xs, ys = noisy_trajectory(X_START, map_rt_to_x(rt, X_START, X_END, rt_max),
                                  Y_BOT, 90, rng, 0.012)
        ax.plot(xs, ys, color=col, alpha=0.20, lw=0.7, zorder=2)

    kde_band(ax, cx, Y_TOP, col, "up")
    kde_band(ax, ex, Y_BOT, col, "down")

    # Mean-RT markers.
    for mx in (mcx, mex):
        ax.plot([mx, mx], [Y_BOT, Y_TOP], color=fs.GREY, ls=(0, (4, 3)),
                lw=0.7, zorder=3)

    # Representative mean paths.
    ax.plot(np.linspace(X_START, mcx, 120), np.linspace(0, Y_TOP, 120),
            color=col, lw=1.8, zorder=5, solid_capstyle="butt")
    ax.plot(np.linspace(X_START, mex, 120), np.linspace(0, Y_BOT, 120),
            color=col, lw=1.8, alpha=0.55, zorder=5, solid_capstyle="butt")

    # Start point and non-decision time.
    ax.scatter([X_START], [0], s=11, color=fs.INK, zorder=6, lw=0)
    ax.annotate("", xy=(X_LEFT + 0.04, -0.30), xytext=(X_START, -0.30),
                arrowprops=dict(arrowstyle="<->", color=fs.INK, lw=0.7,
                                shrinkA=0, shrinkB=0), zorder=6)
    # Just the symbol, over the arrow that marks the interval.  The spelt-out
    # "non-decision time (t)" spans ~5.8 data units in a 6.4-unit axes, so at
    # composite width there is no position for it that does not cover either the
    # trajectory bundle or the "error" boundary label; below the axes it collided
    # with the next row and cost panel a 20% of its height.  The caption carries
    # the definition ("non-decision time t is marked but not contrasted"),
    # which is where a term the reader must be told belongs.
    ax.annotate("$t$", xy=((X_LEFT + X_START) / 2, -0.40),
                ha="center", va="top", fontsize=fs.ANNOT, color=fs.INK,
                zorder=7, path_effects=fs.HALO)

    # Drift rate, annotated directly rather than in a legend.
    v = drift_intercept(RT_KIND, code)
    # y = 0.30, not 0.50: the mean-RT label sits at Y_TOP - 0.30 = 0.70, and in
    # the composite the axes is ~0.90 in tall, so a 0.20 gap put the two texts
    # in contact.  0.40 of axis units clears it at both panel sizes.
    # Left-aligned right of both dashed mean markers, for the same reason as the
    # RT label: right-aligned at X_END the text extended leftwards across the
    # dashes.  Sits a row below the RT label (y = 0.30 vs 0.70) so the two
    # annotations share the x range without touching.
    ax.annotate(f"$v$ = {v:.3f}", xy=(max(mcx, mex), 0.30), xytext=(3, 0),
                textcoords="offset points", ha="left",
                va="center", fontsize=fs.ANNOT, color=col, zorder=7,
                path_effects=fs.HALO)

    # Mean-RT value, anchored right of BOTH dashed mean markers rather than
    # just the correct-response one.  Offsetting from mcx alone let the text run
    # across the mex dash, which reads as a strikethrough at composite size.
    ax.annotate(f"{correct.mean():.2f} s", xy=(max(mcx, mex), Y_TOP - 0.30),
                xytext=(3, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=fs.ANNOT, color=fs.GREY, zorder=7,
                path_effects=fs.HALO)

    ax.set_xlim(X_LEFT - 0.55, X_RIGHT)
    ax.set_ylim(-1.38, 1.22)
    # Each title names its task and the RT the traces terminate on.  Set on one
    # line the widest of the three measures 1.64 in at SECONDARY, which the
    # standalone cell pitch (1.94 in) accommodates.  In the composite a shares
    # row 1 with b, leaving ~1.0 in per cell at 180 mm page width, so the title
    # is wrapped there instead: task name over two lines, the RT qualifier on a
    # third.  Wrapping rather than shrinking keeps the type at SECONDARY, well
    # clear of the 5 pt floor.
    if wrap_title:
        parts = task.split()
        head = " ".join(parts[:-1]) + "\n" + parts[-1] if len(parts) > 1 else task
        title = f"{head}\n({fs.LBL_RT_CRIT})"
    else:
        title = f"{task} ({fs.LBL_RT_CRIT})"
    ax.set_title(title, fontsize=fs.SECONDARY, pad=3, linespacing=1.25)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.annotate("time", xy=(X_RIGHT, Y_MID), xytext=(0, -7),
                textcoords="offset points", ha="right", va="top",
                fontsize=fs.ANNOT, color=fs.GREY, style="italic")
    return mcx


def build(fig=None, axes=None, wrap_title=None):
    df = pd.read_csv(RESULTS_DIR / RT_KIND / f"{RT_KIND}_cleaned.csv")
    standalone = fig is None
    if standalone:
        fig, axes = plt.subplots(1, 3, figsize=(7.09, 1.95))
    # Wrapped titles are a composite-only measure: the standalone cell is wide
    # enough for one line.
    if wrap_title is None:
        wrap_title = not standalone
    for k, code in enumerate(CODES):
        rng = np.random.default_rng(2201 + CODES.index(code))
        draw_condition(axes[k], df, code, rng, wrap_title=wrap_title)
    # Threshold labels only on the leftmost panel -- the rows are aligned, so
    # repeating them three times adds nothing.
    for lbl, y in (("correct", Y_TOP), ("error", Y_BOT)):
        axes[0].annotate(lbl, xy=(X_LEFT - 0.10, y), ha="right", va="center",
                         fontsize=fs.ANNOT, color=fs.INK)
    return fig, axes


def main():
    fs.apply(mpl, dpi=500)
    fig, axes = build()
    fs.letter(axes[0], "a", dx=-0.06, dy=0.02)
    fig.subplots_adjust(wspace=0.10)
    w, h = fs.save_at_width(fig, OUT_DIR / "panel_a_ddm_traces", mm=180)
    print(f"panel a: {w:.2f} x {h:.2f} mm")
    ov = fs.check_text_overlaps(fig)
    print(f"text overlaps: {len(ov)}", ov[:4])


if __name__ == "__main__":
    main()
