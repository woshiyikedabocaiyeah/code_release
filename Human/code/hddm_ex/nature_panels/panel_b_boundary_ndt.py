#!/usr/bin/env python3
"""Panel b: task posteriors for boundary separation (a) and baseline drift
rate (v), RT_critical group model.

Drawn as violins with an inner box-and-whisker summary, one violin per task,
so the two parameters are compared on the same visual grammar: the boundary
posteriors overlap substantially while the drift posteriors separate.

Non-decision time is not shown here.  The fitted group model estimates a
SINGLE t shared across the three conditions (the trace has one `t` node, not
`t(group)`), so it carries no task contrast and belongs in the text rather
than in a by-task panel.
"""

from __future__ import annotations

import itertools
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np

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

CODES = ["Semantic", "Intuitive", "Action"]
RT_KIND = "rt_critical"

# Draw a comparison bracket only when the posterior puts at least this much
# mass on one sign of the difference.  Below it the two posteriors overlap
# enough that a bracket would assert a difference the model does not support.
P_BRACKET = 0.95

# For the manuscript caption.  The two sub-panels come from different fits and
# the reader has to be told; the panel itself has no room for the statement.
CAPTION_NOTE = (
    "Boundary separation is taken from the condition model, which estimates one "
    "value per task.  Baseline drift is taken from the regression model as "
    "intercept plus group shift, reconstructed draw-by-draw; that model "
    "estimates a single shared boundary, so it cannot supply a per-task "
    "boundary contrast.  Bracket stars denote the POSTERIOR PROBABILITY that "
    "the difference has the sign shown, not a frequentist p-value: "
    "*** P > 0.999, ** P > 0.99, * P > 0.95.  Pairs below P = %.2f are left "
    "unmarked."
) % P_BRACKET


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    return pickle.loads(Path(path).read_bytes())


def cat_trace(key: str, model: str = "group") -> np.ndarray:
    return np.concatenate([
        np.asarray(_load(str(RESULTS_DIR / RT_KIND / model / f"chain_{c}_trace.db"))[key][0], float)
        for c in range(1, 5)
    ])


def boundary_draws() -> dict:
    """Per-task boundary separation, condition ("group") model.

    The regression model estimates a single shared `a`, so a per-task boundary
    contrast can only come from the condition model.
    """
    return {fs.CODE_TO_TASK[c]: cat_trace(f"a({c})", model="group")
            for c in CODES}


def drift_draws() -> dict:
    """Per-task baseline drift, regression model, on the draw level.

    The regression model is parameterised with Semantic as the reference, so
    the other two conditions are intercept + group shift.  Reconstructing them
    draw-by-draw (rather than adding posterior means) keeps the correlation
    between intercept and shift, which the credible intervals depend on.
    This is the same quantity annotated on panel a, so the two panels agree.
    """
    ref = "v_C(group, Treatment('Semantic'))"
    ic = cat_trace("v_Intercept", model="regression")
    out = {fs.CODE_TO_TASK["Semantic"]: ic}
    for code in ("Action", "Intuitive"):
        out[fs.CODE_TO_TASK[code]] = ic + cat_trace(f"{ref}[T.{code}]",
                                                    model="regression")
    return {t: out[t] for t in (fs.CODE_TO_TASK[c] for c in CODES)}


#: Star thresholds, keyed to the POSTERIOR PROBABILITY that the difference has
#: the sign shown -- not to a frequentist p-value.  The caption has to say so,
#: because the glyph is borrowed from significance testing and a reader will
#: otherwise assume a null-hypothesis test was run.
STAR_LEVELS = ((0.999, "***"), (0.99, "**"), (0.95, "*"))


def _fmt_p(p: float) -> str:
    """Star notation for the posterior probability of the stated direction."""
    for threshold, stars in STAR_LEVELS:
        if p > threshold:
            return stars
    return "n.s."


def violin(ax, x, values, color, fill):
    """One posterior as a violin with an inner box-and-whisker summary.

    The violin shows the full posterior shape; the black box spans the
    interquartile range, the whisker the 95% credible interval, and the white
    dot the posterior mean.  Same encoding for every task, so the panel is read
    by comparing positions rather than colours.
    """
    parts = ax.violinplot([values], positions=[x], widths=0.62,
                          showextrema=False, showmeans=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor(fill)
        body.set_edgecolor(color)
        body.set_linewidth(0.9)
        body.set_alpha(0.85)
        body.set_zorder(2)
    q1, q3 = np.percentile(values, [25, 75])
    lo, hi = np.percentile(values, [2.5, 97.5])
    ax.plot([x, x], [lo, hi], color=fs.INK, lw=0.9, solid_capstyle="butt",
            zorder=3)
    ax.add_patch(mpatches.Rectangle((x - 0.055, q1), 0.11, q3 - q1,
                                    facecolor=fs.INK, edgecolor="none",
                                    zorder=4))
    ax.plot([x], [values.mean()], marker="o", ms=2.6, color="white",
            mec=fs.INK, mew=0.5, zorder=5)
    # The KDE tail reaches further than any posterior quantile, so the drawn
    # extent -- not a percentile of the draws -- is what the y limits must
    # clear.  Setting limits from quantiles clipped the violins.
    ys = np.concatenate([b.get_paths()[0].vertices[:, 1]
                         for b in parts["bodies"]])
    return float(ys.min()), float(ys.max())


def bracket(ax, x1, x2, y, label, h):
    """Pairwise comparison bracket.

    Labelled with stars keyed to STAR_LEVELS -- thresholds on the POSTERIOR
    PROBABILITY that the difference has the sign shown, not on a p-value.
    Because the glyph is borrowed from significance testing, CAPTION_NOTE
    states the mapping explicitly; the panel has no room for it.
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=fs.INK, lw=0.7,
            solid_joinstyle="miter", zorder=6)
    ax.annotate(label, xy=((x1 + x2) / 2, y + h), xytext=(0, 0.8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=fs.MICRO, color=fs.INK, zorder=6)


def build(fig=None, axes=None, show_key=None):
    """show_key defaults to standalone-only.

    In the composite the three tasks are keyed once for the whole figure, below
    the last row; keying them inside b as well would eat the row's height and
    squeeze the violins.
    """
    standalone = fig is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(4.5, 1.95))
    ax_a, ax_v = axes

    tasks = [fs.CODE_TO_TASK[c] for c in CODES]
    # No per-violin tick labels.  The three display names are 19-23 characters
    # each and two sub-axes carry three apiece: horizontal, wrapped and rotated
    # all collide once the row is compressed in the composite.  Task identity
    # is carried by colour instead, keyed once for the whole panel -- the same
    # resolution the coefficient panels reached for the same cause.

    # The two parameters come from different fits, and the panel says so: the
    # regression model estimates a single shared boundary, so a per-task
    # boundary contrast exists only in the condition model.
    sources = (
        (ax_a, "Boundary separation, $a$", "condition model", boundary_draws),
        (ax_v, "Baseline drift rate, $v$", "regression model", drift_draws),
    )
    for ax, xlabel, model_note, get_draws in sources:
        draws = get_draws()
        extents = []
        for k, t in enumerate(tasks):
            extents.append(violin(ax, k, draws[t], fs.TASK_COL[t],
                                  fs.TASK_FILL[t]))
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(tasks) - 0.45)
        if ax is ax_a:
            # Rotated, "Posterior value" is 105 px against a 94 px axes in the
            # composite and overhangs into the row below's tick labels.  Each
            # sub-axes titles its own quantity (a, v), so the composite drops
            # the "value" without losing what the axis measures.
            ax.set_ylabel("Posterior value" if standalone else "Posterior")
        else:
            # Same quantity type as the left axes; a second identical label
            # would spend width the violins need.
            ax.set_ylabel("")
        # Parameter name only.  Which fit produced each sub-panel belongs in the
        # caption: as an in-panel annotation it collided with the tick labels
        # wherever it was placed, and appended to the title it made the two
        # titles touch each other.  CAPTION_NOTE below carries the text.
        ax.set_title(xlabel, fontsize=fs.SECONDARY, pad=2.5)


        # Pairwise posterior probabilities.  Brackets are drawn only where the
        # direction is essentially certain; a bracket on an overlapping pair
        # would imply a difference the posterior does not support.
        lo = min(e[0] for e in extents)
        hi = max(e[1] for e in extents)
        rng = hi - lo
        drawn = 0
        for i, j in itertools.combinations(range(len(tasks)), 2):
            d = draws[tasks[i]] - draws[tasks[j]]
            p = max(float(np.mean(d > 0)), float(np.mean(d < 0)))
            if p < P_BRACKET:
                continue
            # Stars sit closer to their bracket than the old "P > 0.999" text
            # did, so the three tiers need more vertical separation than the
            # 0.145 step used when the label itself supplied the gap.
            y = hi + rng * (0.10 + 0.20 * drawn)
            bracket(ax, i, j, y, _fmt_p(p), rng * 0.035)
            drawn += 1
        head = rng * (0.10 + 0.20 * drawn) + rng * 0.10
        ax.set_ylim(lo - rng * 0.14, hi + head)
        # Cap the tick count and prune the topmost tick.  Adding headroom
        # otherwise makes the locator emit a new highest tick that reaches the
        # title, so more padding produced more collisions, not fewer.
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="upper"))
        print(f"    {xlabel.split(',')[0]:24s} {drawn} bracket(s)")

    if show_key is None:
        show_key = standalone

    # One colour key for both sub-axes, anchored under the left one and spanning
    # the pair.  Drawn as a legend rather than as tick labels so the long names
    # can run horizontally across the full panel width instead of being
    # squeezed into three violin slots.
    # Two columns, not three: at composite width the three names set on one
    # line measure wider than the two sub-axes together and ran off the page.
    if not show_key:
        return fig, axes

    handles = [mpatches.Patch(facecolor=fs.TASK_FILL[t], edgecolor=fs.TASK_COL[t],
                              linewidth=0.9, label=t) for t in tasks]
    # Anchored to the figure, not to one sub-axes: anchored under the left
    # axes the second column reached beneath the right axes and collided with
    # its tick labels.
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=fs.MICRO,
               handlelength=0.9, handleheight=0.9,
               handletextpad=0.4, columnspacing=1.4,
               borderaxespad=0.2)

    return fig, axes


def main():
    fs.apply(mpl, dpi=500)
    fig, axes = build()
    fs.letter(axes[0], "b")
    w, h = fs.save_at_width(fig, OUT_DIR / "panel_b_boundary_ndt", mm=114)
    print(f"panel b: {w:.2f} x {h:.2f} mm")
    ov = fs.check_text_overlaps(fig)
    print(f"text overlaps: {len(ov)}", ov[:4])


if __name__ == "__main__":
    main()
