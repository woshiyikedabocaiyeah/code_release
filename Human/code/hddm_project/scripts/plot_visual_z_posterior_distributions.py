#!/usr/bin/env python
"""Plot posterior distributions of Visual_Z coefficients across conditions."""

import argparse
import csv
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
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde


PLOT_ORDER = ["Semantic", "Voe", "Action"]
DISPLAY_LABELS = {
    "Semantic": "Concept Verification",
    "Voe": "Plausibility Assessment",
    "Action": "Affordance Recognition",
}
COLORS = {
    "Semantic": "#4EA6C6",
    "Voe": "#1F987B",
    "Action": "#E3362A",
}


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(colors="#111111", width=1.0, labelsize=10)


def load_models(output_dir, model_name, chains):
    models = []
    model_dir = output_dir / "models" / model_name
    for chain in range(1, chains + 1):
        models.append(hddm.load(str(model_dir / f"{model_name}_chain{chain}.pkl")))
    return models


def pooled_trace(models, node_name):
    return np.concatenate(
        [np.asarray(model.nodes_db.node[node_name].trace(), dtype=float) for model in models]
    )


def load_visual_slopes(output_dir, chains):
    models = load_models(output_dir, "regression", chains)
    return summarize_visual_slopes(models)


def summarize_visual_slopes(models):
    node_names = {
        "Action": "v_C(group)[Action]:visual_z",
        "Semantic": "v_C(group)[Semantic]:visual_z",
        "Voe": "v_C(group)[Voe]:visual_z",
    }
    slopes = {}
    for group, node_name in node_names.items():
        trace = pooled_trace(models, node_name)
        slopes[group] = {
            "node": node_name,
            "trace": trace,
            "mean": float(np.mean(trace)),
            "hdi_2.5": float(np.quantile(trace, 0.025)),
            "hdi_97.5": float(np.quantile(trace, 0.975)),
        }
    return slopes


def save_summary(output_dir, slopes):
    path = output_dir / "tables" / "visual_z_posterior_distribution_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "display_label", "mean", "hdi_2.5", "hdi_97.5", "node", "n_draws"])
        for group in PLOT_ORDER:
            row = slopes[group]
            writer.writerow(
                [
                    group,
                    DISPLAY_LABELS[group],
                    row["mean"],
                    row["hdi_2.5"],
                    row["hdi_97.5"],
                    row["node"],
                    len(row["trace"]),
                ]
            )
    return path


def plot_posterior_distributions(ax, slopes):
    all_traces = np.concatenate([slopes[group]["trace"] for group in PLOT_ORDER])
    xmin = float(np.min(all_traces))
    xmax = float(np.max(all_traces))
    span = xmax - xmin
    x_grid = np.linspace(xmin - span * 0.08, xmax + span * 0.08, 500)
    bin_width = 0.005
    bins = np.arange(
        np.floor(x_grid.min() / bin_width) * bin_width,
        np.ceil(x_grid.max() / bin_width) * bin_width + bin_width,
        bin_width,
    )
    max_density = 0.0

    density_cache = {}
    for group in PLOT_ORDER:
        trace = slopes[group]["trace"]
        kde = gaussian_kde(trace)
        density = kde(x_grid)
        density_cache[group] = {"kde": kde, "density": density}
        max_density = max(max_density, float(np.max(density)))

    for group in PLOT_ORDER:
        row = slopes[group]
        color = COLORS[group]
        trace = row["trace"]
        kde = density_cache[group]["kde"]
        density = density_cache[group]["density"]

        ax.hist(
            trace,
            bins=bins,
            density=True,
            color=color,
            alpha=0.16,
            edgecolor="white",
            linewidth=0.7,
            zorder=1,
        )
        ax.plot(x_grid, density, color=color, linewidth=2.0, zorder=3)
        ax.vlines(
            row["mean"],
            0,
            float(kde(row["mean"])),
            color=color,
            linewidth=1.2,
            alpha=0.95,
            zorder=4,
        )

    ax.set_title(
        r"Model results: posterior distributions of $\mathrm{Visual}_{z}$ coefficients",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Beta coefficient")
    ax.set_ylabel("Posterior density")
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(0, max_density * 1.15)
    style_axis(ax)

    legend_handles = [
        Line2D([0], [0], color=COLORS[group], lw=2.6, label=DISPLAY_LABELS[group])
        for group in PLOT_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.98),
        fontsize=8.4,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = root / args.output_dir
    slopes = load_visual_slopes(output_dir, args.chains)

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    plot_posterior_distributions(ax, slopes)
    fig.tight_layout()

    fig_dir = _organized_fig_dir(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "visual_z_posterior_distributions.png"
    pdf_path = fig_dir / "visual_z_posterior_distributions.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = save_summary(output_dir, slopes)
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
