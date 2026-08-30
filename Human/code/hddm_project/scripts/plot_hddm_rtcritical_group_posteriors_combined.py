#!/usr/bin/env python
"""Combine RT_critical HDDM diagrams with group posterior violin panels."""

import argparse
from pathlib import Path

import matplotlib


# --- figure output redirected to _organized/figures/ -----------------------
# Models, tables, and prepared data still go to --output-dir;
# only figures are redirected.
_ORGANIZED_FIG_ROOT = Path(__file__).resolve().parents[3] / "figures" / "hddm_project"


def _organized_fig_dir(output_dir) -> Path:
    d = _ORGANIZED_FIG_ROOT / Path(output_dir).name
    d.mkdir(parents=True, exist_ok=True)
    return d

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import plot_group_posteriors as group_posteriors


HDDM_PANEL_PATHS = [
    ("Concept Verification", ("rt_critical_concept_verification_sample.png", "rt_critical_semantic_sample.png")),
    ("Plausibility Assessment", ("rt_critical_plausibility_assessment_sample.png", "rt_critical_intuitive_sample.png")),
    ("Affordance Recognition", ("rt_critical_affordance_recognition_sample.png", "rt_critical_action_sample.png")),
]


def resolve_image_path(sample_dir, filenames):
    for filename in filenames:
        path = sample_dir / filename
        if path.exists():
            return path
    names = ", ".join(filenames)
    raise FileNotFoundError(f"Could not find any RT_critical sample image in {sample_dir}: {names}")


def add_hddm_image(ax, image_path):
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_panel_label(ax, label):
    ax.text(
        -0.065,
        1.16,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )


def add_v_significance(ax, v_traces):
    semantic_vs_intuitive = v_traces["Semantic"] - v_traces["Intuitive"]
    semantic_vs_action = v_traces["Semantic"] - v_traces["Action"]
    intuitive_vs_action = v_traces["Intuitive"] - v_traces["Action"]
    sig_brackets = [
        (1, 2, group_posteriors.posterior_star(semantic_vs_intuitive)),
        (2, 3, group_posteriors.posterior_star(intuitive_vs_action)),
        (1, 3, group_posteriors.posterior_star(semantic_vs_action)),
    ]

    vmax = max(float(np.max(trace)) for trace in v_traces.values())
    ymin, ymax = ax.get_ylim()
    axis_range = ymax - ymin
    bracket_gap = axis_range * 0.043
    bracket_height = axis_range * 0.016
    text_pad = axis_range * 0.001
    base_y = vmax + axis_range * 0.002

    for x1, x2, label in sig_brackets:
        if label == "n.s.":
            continue
        local_y = base_y
        local_text_pad = text_pad
        if (x1, x2) == (1, 2):
            local_y = base_y - axis_range * 0.022
            local_text_pad = -axis_range * 0.014
        if (x1, x2) == (2, 3):
            local_y = base_y - axis_range * 0.035
            local_text_pad = -axis_range * 0.01
        group_posteriors.add_sig_bracket(ax, x1, x2, local_y, bracket_height, label, local_text_pad)
        base_y += bracket_gap

    ax.set_ylim(ymin, max(ymax, base_y + bracket_height * 2.4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = _organized_fig_dir(output_dir)
    sample_dir = figure_dir / "rt_condition_sample_style"

    a_traces, v_traces = group_posteriors.build_traces(output_dir, args.chains)

    fig = plt.figure(figsize=(16.2, 9.2))
    grid = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.08, 1.0],
        hspace=0.18,
        wspace=0.18,
    )

    hddm_axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
    ]
    for ax, (_, filenames) in zip(hddm_axes, HDDM_PANEL_PATHS):
        add_hddm_image(ax, resolve_image_path(sample_dir, filenames))
    add_panel_label(hddm_axes[0], "A")

    ax_b = fig.add_subplot(grid[1, 0:3])
    ax_c = fig.add_subplot(grid[1, 3:6])
    group_posteriors.add_violin_panel(
        ax_b,
        a_traces,
        r"Boundary separation ($a$)",
        "posterior value",
    )
    group_posteriors.add_violin_panel(
        ax_c,
        v_traces,
        r"Baseline drift rate ($v$)",
        "posterior value",
    )
    add_v_significance(ax_c, v_traces)
    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C")

    fig.text(
        0.69,
        0.032,
        "* p < .05, ** p < .01, *** p < .001 (mapped from posterior contrast tail probability)",
        ha="center",
        va="center",
        fontsize=13,
        color="#333333",
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1.0])
    c_pos = ax_c.get_position()
    ax_c.set_position([c_pos.x0 + 0.065, c_pos.y0, c_pos.width - 0.04, c_pos.height])

    png_path = figure_dir / "hddm_rtcritical_group_posteriors_combined.png"
    pdf_path = figure_dir / "hddm_rtcritical_group_posteriors_combined.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
