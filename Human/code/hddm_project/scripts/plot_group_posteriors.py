#!/usr/bin/env python
"""Plot pooled posterior distributions for group-level a and baseline v."""

import argparse
from pathlib import Path

import hddm
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUPS = ["Semantic", "Intuitive", "Action"]
DISPLAY_LABELS = {
    "Semantic": "Concept\nVerification",
    "Intuitive": "Plausibility\nAssessment",
    "Action": "Affordance\nRecognition",
}
COLORS = {
    "Semantic": "#E76F51",
    "Intuitive": "#E9C46A",
    "Action": "#2A9D8F",
}


def load_models(output_dir, model_name, chains):
    models = []
    model_dir = output_dir / "models" / model_name
    for chain in range(1, chains + 1):
        path = model_dir / f"{model_name}_chain{chain}.pkl"
        models.append(hddm.load(str(path)))
    return models


def pooled_trace(models, node_name):
    traces = []
    for model in models:
        trace = np.asarray(model.nodes_db.node[node_name].trace(), dtype=float)
        traces.append(trace)
    return np.concatenate(traces)


def summarize_trace(parameter, group, trace, source_model):
    q025, q25, q50, q75, q975 = np.quantile(trace, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {
        "parameter": parameter,
        "group": group,
        "source_model": source_model,
        "n_draws": int(len(trace)),
        "mean": float(np.mean(trace)),
        "sd": float(np.std(trace, ddof=1)),
        "hdi_2.5": float(q025),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "hdi_97.5": float(q975),
    }


def build_traces(output_dir, chains):
    group_models = load_models(output_dir, "group", chains)
    regression_models = load_models(output_dir, "regression", chains)

    a_traces = {
        "Semantic": pooled_trace(group_models, "a(Semantic)"),
        "Intuitive": pooled_trace(group_models, "a(Voe)"),
        "Action": pooled_trace(group_models, "a(Action)"),
    }

    intercept = pooled_trace(regression_models, "v_Intercept")
    semantic_offset = pooled_trace(regression_models, "v_C(group)[T.Semantic]")
    voe_offset = pooled_trace(regression_models, "v_C(group)[T.Voe]")
    v_traces = {
        "Semantic": intercept + semantic_offset,
        "Intuitive": intercept + voe_offset,
        "Action": intercept,
    }
    return a_traces, v_traces


def style_axis(ax):
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#333333", labelsize=13)


def add_violin_panel(ax, traces, title, ylabel):
    positions = np.arange(1, len(GROUPS) + 1)
    series = [traces[group] for group in GROUPS]
    violin = ax.violinplot(
        series,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, group in zip(violin["bodies"], GROUPS):
        color = COLORS[group]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
        body.set_linewidth(1.0)

    for idx, group in enumerate(GROUPS, start=1):
        trace = traces[group]
        q025, q25, q50, q75, q975 = np.quantile(trace, [0.025, 0.25, 0.5, 0.75, 0.975])
        ax.vlines(idx, q025, q975, color="#1F1F1F", linewidth=1.5, zorder=3)
        ax.vlines(idx, q25, q75, color="#1F1F1F", linewidth=4.5, zorder=4)
        ax.scatter(idx, q50, s=34, color="white", edgecolor="#1F1F1F", linewidth=0.8, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_LABELS[group] for group in GROUPS], fontsize=13)
    ax.set_title(title, fontsize=17, pad=14)
    ax.set_ylabel(ylabel, fontsize=14)
    style_axis(ax)


def posterior_star(diff):
    p_error = min(float(np.mean(diff > 0)), float(np.mean(diff < 0)))
    if p_error < 0.001:
        return "***"
    if p_error < 0.01:
        return "**"
    if p_error < 0.05:
        return "*"
    return "n.s."


def add_sig_bracket(ax, x1, x2, y, h, label, text_pad):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#222222", linewidth=1.2, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + h + text_pad,
        label,
        ha="center",
        va="bottom",
        fontsize=17,
        color="#111111",
    )


def save_summary(output_dir, a_traces, v_traces):
    rows = []
    for group, trace in a_traces.items():
        rows.append(summarize_trace("a", group, trace, "group"))
    for group, trace in v_traces.items():
        rows.append(summarize_trace("baseline_v", group, trace, "regression"))
    frame = pd.DataFrame(rows)
    path = output_dir / "tables" / "group_posterior_plot_summary.csv"
    frame.to_csv(path, index=False)
    return path


def plot_posteriors(output_dir, a_traces, v_traces):
    figure_dir = _organized_fig_dir(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "group_posteriors_a_baseline_v.png"
    pdf_path = figure_dir / "group_posteriors_a_baseline_v.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    add_violin_panel(
        axes[0],
        a_traces,
        "Substantial overlap in boundary separation (a)",
        "posterior value",
    )
    add_violin_panel(
        axes[1],
        v_traces,
        "Significant separation in baseline drift rate (v)",
        "posterior value",
    )

    semantic_vs_intuitive = v_traces["Semantic"] - v_traces["Intuitive"]
    semantic_vs_action = v_traces["Semantic"] - v_traces["Action"]
    intuitive_vs_action = v_traces["Intuitive"] - v_traces["Action"]
    star_si = posterior_star(semantic_vs_intuitive)
    star_sa = posterior_star(semantic_vs_action)
    star_ia = posterior_star(intuitive_vs_action)

    vmax = max(float(np.max(trace)) for trace in v_traces.values())
    ymin, ymax = axes[1].get_ylim()
    axis_range = ymax - ymin
    bracket_gap = axis_range * 0.043
    bracket_height = axis_range * 0.016
    text_pad = axis_range * 0.001
    base_y = vmax + axis_range * 0.002
    sig_brackets = [
        (1, 2, star_si),
        (2, 3, star_ia),
        (1, 3, star_sa),
    ]
    drawn = 0
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
        add_sig_bracket(axes[1], x1, x2, local_y, bracket_height, label, local_text_pad)
        base_y += bracket_gap
        drawn += 1
    axes[1].set_ylim(ymin, max(ymax, base_y + bracket_height * 2.4))

    fig.suptitle(
        "HDDM posterior distributions across the three task conditions\n"
        "Boundary separation a overlaps across groups, whereas baseline drift rate v separates reliably",
        fontsize=13,
        y=0.98,
    )
    axes[0].text(
        0.02,
        0.98,
        "A",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#111111",
    )
    axes[1].text(
        0.02,
        0.98,
        "B",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.72,
        0.03,
        "* p < .05, ** p < .01, *** p < .001 (mapped from posterior contrast tail probability)",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    a_traces, v_traces = build_traces(output_dir, args.chains)
    png_path, pdf_path = plot_posteriors(output_dir, a_traces, v_traces)
    summary_path = save_summary(output_dir, a_traces, v_traces)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
