"""
Nature-quality posterior-coefficient distribution figure.

Reuses the original HDDM trace-loading machinery from
``make_hddm_feature_modulation_figures`` so the posterior draws (and therefore
the plotted densities and means) are byte-identical to the original figure.
Only labels, fonts, and output styling are changed:

  * group names   Semantic  -> Concept Verification
                  Intuitive -> Plausibility Assessment
                  Action    -> Affordance Recognition
  * the ``Z`` in ``Visual_Z`` is rendered as a subscript (mathtext)
  * Arial font, embedded/editable vector text, 400 dpi PNG + PDF + SVG
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_hddm_feature_modulation_figures as M

# --- display config (kept identical to the behavior figures) ---
OLD2NEW = {
    "Semantic": "Concept Verification",
    "Intuitive": "Plausibility Assessment",
    "Action": "Affordance Recognition",
}
NEW_COLORS = {OLD2NEW[g]: M.COLORS[g] for g in M.GROUPS}
# legend / draw order mirrors the original module (GROUPS = Semantic, Intuitive, Action)
LEGEND_ORDER = [OLD2NEW[g] for g in M.GROUPS]

# mathtext coefficient labels: Z rendered as subscript
FEATURE_CONFIG = {
    "visual_z": {
        "coef_label": r"$v_{\mathrm{Visual_{Z}}}$",
        "stem": "paper_figure_visual_z_posterior_coefficients_nature",
    },
    "physical_z": {
        "coef_label": r"$v_{\mathrm{Physical_{Z}}}$",
        "stem": "paper_figure_physical_z_posterior_coefficients_nature",
    },
}

OUT_DIR = M.OUT_DIR

mpl.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,   # embed TrueType so text stays editable / selectable
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 1.1,
})


def build_posterior_figure(rt_key: str = "rt_critical", feature: str = "visual_z"):
    cfg = FEATURE_CONFIG[feature]
    coef_label = cfg["coef_label"]
    # --- identical posterior draws: reuse original trace loader ---
    draws_by_group = {
        g: M.combined_trace(rt_key, M.TRACE_KEYS[feature][g]) for g in M.GROUPS
    }
    all_draws = np.concatenate([draws_by_group[g] for g in M.GROUPS])
    x_min, x_max = float(all_draws.min()), float(all_draws.max())
    pad = max((x_max - x_min) * 0.18, 0.02)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="white")
    ax.set_facecolor("white")
    max_density = 0.0

    for g in M.GROUPS:
        new = OLD2NEW[g]
        color = NEW_COLORS[new]
        draws = draws_by_group[g]
        ax.hist(draws, bins=46, density=True, color=color, alpha=0.15,
                edgecolor="white", linewidth=0.25, zorder=1)
        kde = gaussian_kde(draws)
        density = kde(x_grid)
        max_density = max(max_density, float(density.max()))
        ax.fill_between(x_grid, 0, density, color=color, alpha=0.10,
                        linewidth=0, zorder=2)
        ax.plot(x_grid, density, color=color, linewidth=2.8, zorder=3)
        ax.axvline(float(draws.mean()), color=color, linewidth=1.3,
                   alpha=0.95, zorder=4)

    ax.axhline(0, color="#111111", linewidth=1.0)
    ax.set_title(rf"Model results: posterior distributions of {coef_label} coefficients",
                 fontsize=22, pad=16)
    ax.set_xlabel(r"$\beta$ coefficient", fontsize=19, labelpad=9)
    ax.set_ylabel("Posterior density", fontsize=19, labelpad=9)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(0, max_density * 1.13)
    ax.tick_params(axis="both", labelsize=16, width=1.2, length=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    # legend placed BELOW the axes, laid out horizontally (single row) so it
    # spans the width of the panel and no longer overlaps any distribution
    handles = [Line2D([0], [0], color=NEW_COLORS[g], lw=3.5) for g in LEGEND_ORDER]
    ax.legend(handles, LEGEND_ORDER, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=len(LEGEND_ORDER), fontsize=16, handlelength=1.7,
              columnspacing=1.8, borderaxespad=0.0)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = cfg["stem"]
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    svg = OUT_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    # report posterior means (for verification against the original)
    for g in M.GROUPS:
        d = draws_by_group[g]
        print(f"{OLD2NEW[g]}: mean={d.mean():.4f}  sd={d.std():.4f}  n={d.size}")
    print(str(png)); print(str(pdf)); print(str(svg))
    return png, pdf, svg


def main():
    build_posterior_figure(feature="visual_z")
    build_posterior_figure(feature="physical_z")


if __name__ == "__main__":
    main()
