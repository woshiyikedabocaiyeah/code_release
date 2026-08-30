#!/usr/bin/env python3
"""Regenerate the Visual_Z 3-panel behavioural figure at Nature-submission quality.

Statistics are IDENTICAL to make_hddm_feature_modulation_figures.make_visual_z_behavior_hddm_figure:
the statistical functions are imported and reused, so the reported slopes (b) are unchanged.
Only the presentation changes:
  * Task names:  Semantic -> Concept Verification
                 Intuitive -> Plausibility Assessment
                 Action    -> Affordance Recognition
  * Subscripts:  RT_onset -> RT_onset, RT_critical -> RT_critical, Visual_Z -> Visual_Z (mathtext)
  * y-axis labels drop the word "Predicted"
  * ACC axis capped at 100%
  * Arial font, vector (PDF/SVG) + high-dpi PNG output
"""
from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import make_hddm_feature_modulation_figures as M

# ---------------------------------------------------------------- display config
OLD2NEW = {
    "Semantic": "Concept Verification",
    "Intuitive": "Plausibility Assessment",
    "Action": "Affordance Recognition",
}
NEW_COLORS = {
    "Concept Verification": "#4FA3C7",     # blue  (was Semantic)
    "Plausibility Assessment": "#2C9C84",  # green (was Intuitive)
    "Affordance Recognition": "#E63E32",   # red   (was Action)
}
# legend / draw order mirrors the original module (GROUPS = Semantic, Intuitive, Action)
LEGEND_ORDER = ["Concept Verification", "Plausibility Assessment", "Affordance Recognition"]
# per-panel slope-label order mirrors original (Action, Intuitive, Semantic)
LABEL_ORDER = ["Affordance Recognition", "Plausibility Assessment", "Concept Verification"]

# mathtext labels with subscripts
LBL_VISUAL = r"$\mathrm{Visual_{Z}}$"
LBL_PHYSICAL = r"$\mathrm{Physical_{Z}}$"
LBL_RT_ONSET = r"$\mathrm{RT_{onset}}$"
LBL_RT_CRIT = r"$\mathrm{RT_{critical}}$"

OUT_DIR = M.OUT_DIR

mpl.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,   # embed TrueType so text stays editable / selectable
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 1.1,
})


def remap(d: dict) -> dict:
    """Remap a dict keyed by old group names to new display names."""
    return {OLD2NEW[k]: v for k, v in d.items()}


def relabel_group_col(df):
    df = df.copy()
    df["group"] = df["group"].map(OLD2NEW)
    return df


def draw_panel(ax, scatter, curves, x_grid, title, ylabel, ylim,
               slope_values, slope_loc, xlabel=LBL_VISUAL):
    ax.set_facecolor("white")
    for group in LEGEND_ORDER:
        color = NEW_COLORS[group]
        g = scatter[scatter["group"] == group]
        ax.scatter(g["x"], g["value"], s=20, color=color, alpha=0.9,
                   edgecolor="none", zorder=2)
        cx = curves[group].get("x", x_grid)
        ax.fill_between(cx, curves[group]["low"], curves[group]["high"],
                        color=color, alpha=0.14, linewidth=0, zorder=1)
        ax.plot(cx, curves[group]["mean"], color=color, linewidth=2.5, zorder=3)

    ax.set_title(title, fontsize=15, pad=11)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)
    ax.set_xlim(float(x_grid.min()) - 0.05, float(x_grid.max()) + 0.05)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=12, width=1.1, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    if slope_loc == "upper_left":
        x_text, y_text, ha, y_step = 0.04, 0.94, "left", -0.082
    else:  # lower_right
        x_text, y_text, ha, y_step = 0.97, 0.24, "right", -0.082
    for i, group in enumerate(LABEL_ORDER):
        ax.text(x_text, y_text + i * y_step,
                f"{group}: b = {slope_values[group]:.3f}",
                transform=ax.transAxes, fontsize=10.5, color=NEW_COLORS[group],
                ha=ha, va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2))


FEATURE_CONFIG = {
    "Visual_Z": {
        "covariate": "Physical_Z",
        "xlabel": LBL_VISUAL,
        "title_feat": LBL_VISUAL,
        "stem": "paper_figure_visual_z_modulation_nature",
    },
    "Physical_Z": {
        "covariate": "Visual_Z",
        "xlabel": LBL_PHYSICAL,
        "title_feat": LBL_PHYSICAL,
        "stem": "paper_figure_physical_z_modulation_nature",
    },
}


def build_figure(feature: str):
    raw_df = M.load_raw_behavior()
    cfg = FEATURE_CONFIG[feature]
    covariate = cfg["covariate"]
    xlabel = cfg["xlabel"]
    title_feat = cfg["title_feat"]

    feature_lo = float(raw_df[feature].min())
    feature_hi = float(raw_df[feature].max())
    x_grid = np.linspace(feature_lo, feature_hi, 220)

    # --- observed scatter (group means per unique Visual_Z) ---
    sc_onset = relabel_group_col(M.build_empirical_feature_points(raw_df, feature, "RT_onset", filter_rt=True))
    sc_crit = relabel_group_col(M.build_empirical_feature_points(raw_df, feature, "RT_critical", filter_rt=True))
    sc_acc = relabel_group_col(M.build_empirical_feature_points(raw_df, feature, "ACC", value_scale=100.0))

    # --- LMM prediction lines + slopes (identical statistics) ---
    cv_onset, sl_onset = M.fit_lmm_feature_effect(raw_df, "RT_onset", feature, covariate, filter_rt=True)
    cv_crit, sl_crit = M.fit_lmm_feature_effect(raw_df, "RT_critical", feature, covariate, filter_rt=True)
    cv_acc, sl_acc = M.fit_lmm_feature_effect(raw_df, "ACC", feature, covariate, prediction_scale=100.0)

    cv_onset, cv_crit, cv_acc = remap(cv_onset), remap(cv_crit), remap(cv_acc)

    def slopes(sl):
        return {OLD2NEW[g]: sl[g]["estimate"] for g in OLD2NEW}

    # y-limits: reuse original dynamic logic for RT panels, cap ACC at 100
    yl_onset = M.dynamic_ylim(
        relabel_group_col(sc_onset).rename(columns={"value": "value"}),
        {OLD2NEW.get(k, k): v for k, v in [] } or cv_onset, "value") if False else None
    # compute directly to avoid group-name mismatch in dynamic_ylim
    def dyn_ylim(scatter, curves, pad_fraction=0.12):
        vals = [scatter["value"].to_numpy(float)]
        for grp in LEGEND_ORDER:
            vals += [curves[grp]["mean"], curves[grp]["low"], curves[grp]["high"]]
        allv = np.concatenate(vals)
        ymin, ymax = float(np.nanmin(allv)), float(np.nanmax(allv))
        pad = (ymax - ymin) * pad_fraction if ymax > ymin else 0.5
        return ymin - pad * 0.45, ymax + pad

    yl_onset = dyn_ylim(sc_onset, cv_onset)
    yl_crit = dyn_ylim(sc_crit, cv_crit)
    # ACC: keep original focused window (bottom 72) but cap top at 100
    # (original used 102.5; an accuracy axis above 100% is not meaningful)
    yl_acc = (72.0, 100.0)

    # --- figure ---
    fig = plt.figure(figsize=(16.0, 5.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    draw_panel(axes[0], sc_onset, cv_onset, x_grid,
               rf"{title_feat} effect on {LBL_RT_ONSET}", rf"{LBL_RT_ONSET} (s)",
               yl_onset, slopes(sl_onset), "upper_left", xlabel=xlabel)
    draw_panel(axes[1], sc_crit, cv_crit, x_grid,
               rf"{title_feat} effect on {LBL_RT_CRIT}", rf"{LBL_RT_CRIT} (s)",
               yl_crit, slopes(sl_crit), "upper_left", xlabel=xlabel)
    draw_panel(axes[2], sc_acc, cv_acc, x_grid,
               rf"{title_feat} effect on ACC", "ACC (%)",
               yl_acc, slopes(sl_acc), "lower_right", xlabel=xlabel)

    handles = [Line2D([0], [0], color=NEW_COLORS[g], lw=3, marker="o", markersize=7)
               for g in LEGEND_ORDER]
    fig.legend(handles, LEGEND_ORDER, loc="lower center",
               bbox_to_anchor=(0.5, -0.055), ncol=3, frameon=False,
               fontsize=13, handlelength=1.7, columnspacing=1.4)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.17)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = cfg["stem"]
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    svg = OUT_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    print(f"=== {feature} ===")
    for name, sl in [("RT_onset", sl_onset), ("RT_critical", sl_crit), ("ACC", sl_acc)]:
        print(name, {OLD2NEW[g]: round(sl[g]["estimate"], 3) for g in OLD2NEW})
    print(str(png)); print(str(pdf)); print(str(svg))
    return png, pdf, svg


def main():
    for feature in ("Visual_Z", "Physical_Z"):
        build_figure(feature)


if __name__ == "__main__":
    main()
