#!/usr/bin/env python
"""Create sample-style RT-based task-wise HDDM figures."""

import argparse
import warnings
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
from matplotlib import colors as mcolors
from scipy.stats import gaussian_kde


GROUP_MAP = {
    "sensorimotor": "Action",
    "categorization": "Semantic",
    "Voe": "Voe",
}
PLOT_ORDER = ["Semantic", "Voe", "Action"]
DISPLAY_LABELS = {
    "Semantic": "Concept Verification",
    "Voe": "Plausibility Assessment",
    "Action": "Affordance Recognition",
}
FILE_LABELS = {
    "Semantic": "concept_verification",
    "Voe": "plausibility_assessment",
    "Action": "affordance_recognition",
}
COLORS = {
    "Semantic": "#4EA6C6",
    "Voe": "#1F987B",
    "Action": "#E3362A",
}
MODEL_ORDER = ["sensorimotor", "categorization", "Voe"]
VISUAL_LABEL = r"$\mathrm{Visual}_{z}$"
PHYSICAL_LABEL = r"$\mathrm{Physical}_{z}$"
RT_LABELS = {
    "RT_onset": r"RT$_{onset}$",
    "RT_critical": r"RT$_{critical}$",
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


def lighten_color(color, mix=0.32):
    base = np.array(mcolors.to_rgb(color), dtype=float)
    white = np.ones(3, dtype=float)
    return tuple(base * (1.0 - mix) + white * mix)


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


def load_regression_slopes(output_dir, chains, feature):
    models = load_models(output_dir, "regression", chains)
    node_names = {
        "Action": f"v_C(group)[Action]:{feature}",
        "Semantic": f"v_C(group)[Semantic]:{feature}",
        "Voe": f"v_C(group)[Voe]:{feature}",
    }
    out = {}
    for group, node_name in node_names.items():
        trace = pooled_trace(models, node_name)
        out[group] = {
            "node": node_name,
            "trace": trace,
            "mean": float(np.mean(trace)),
            "hdi_2.5": float(np.quantile(trace, 0.025)),
            "hdi_97.5": float(np.quantile(trace, 0.975)),
        }
    return out


def load_response_rt(raw_csv, rt_column, min_rt=0.2):
    data = pd.read_csv(raw_csv)
    out = {}
    for group in PLOT_ORDER:
        original = "categorization" if group == "Semantic" else ("Voe" if group == "Voe" else "sensorimotor")
        subset = data[data["condition"] == original].copy()
        subset = subset[np.isfinite(subset[rt_column])]
        subset = subset[subset[rt_column] > min_rt]
        out[group] = {
            1: np.asarray(subset.loc[subset["ACC"] == 1, rt_column], dtype=float),
            0: np.asarray(subset.loc[subset["ACC"] == 0, rt_column], dtype=float),
        }
    return out


def plot_single_task_ddm_panel(ax, group, slope_row, feature_label, rt_label, response_rt):
    rng = np.random.default_rng(13)
    x0 = 0.06
    upper = 0.78
    lower = -0.78
    center = 0.0
    t = np.linspace(0.0, 1.0, 150)
    color = COLORS[group]
    error_color = lighten_color(color, mix=0.32)

    correct_rt = np.asarray(response_rt[1], dtype=float)
    error_rt = np.asarray(response_rt[0], dtype=float)
    all_rt = np.concatenate([correct_rt, error_rt])
    rt_lo = float(np.quantile(all_rt, 0.05))
    rt_hi = float(np.quantile(all_rt, 0.95))
    if rt_hi <= rt_lo:
        rt_hi = rt_lo + 1.0

    def rt_to_crossing_x(rt_values):
        clipped = np.clip(np.asarray(rt_values, dtype=float), rt_lo, rt_hi)
        scaled = (clipped - rt_lo) / (rt_hi - rt_lo)
        return x0 + 0.34 + 0.42 * scaled

    def make_trajectory(x_end, y_end_local):
        tau = t
        x = x0 + (x_end - x0) * tau
        base = y_end_local * np.power(tau, 1.02)
        noise = np.cumsum(rng.normal(0.0, 1.0, size=tau.size))
        bridge = np.linspace(noise[0], noise[-1], tau.size)
        noise = noise - bridge
        scale = np.max(np.abs(noise))
        if scale > 0:
            noise = noise / scale
        y = base + noise * 0.080 * (1.0 - tau) * (0.55 + 0.45 * tau)
        y[0] = 0.0
        y[-1] = y_end_local
        return x, y

    display_rt_label = RT_LABELS.get(rt_label, rt_label)
    ax.set_title(f"{DISPLAY_LABELS[group]}: {feature_label} with {display_rt_label}", fontsize=12.5, pad=10)
    ax.set_xlim(-0.03, 1.08)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.hlines([upper, center, lower], 0.0, 0.92, colors="#777777", linewidth=1.0, zorder=0)
    ax.scatter([x0], [0.0], s=20, color="#111111", zorder=6)
    ax.text(-0.01, upper + 0.02, "correct", ha="right", va="bottom", fontsize=9.2, color="#222222")
    ax.text(-0.01, lower - 0.02, "error", ha="right", va="top", fontsize=9.2, color="#222222")

    ax.annotate(
        "",
        xy=(x0, -0.16),
        xytext=(0.0, -0.16),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=0.9),
    )
    ax.text(
        0.0,
        -0.34,
        "non-decision\n time (t)",
        ha="left",
        va="top",
        fontsize=8.3,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.3),
    )
    ax.annotate(
        "",
        xy=(0.38, -1.05),
        xytext=(0.22, -1.05),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9),
    )
    ax.text(
        0.30,
        -1.01,
        "time",
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.22),
    )

    correct_x = rt_to_crossing_x(correct_rt)
    error_x = rt_to_crossing_x(error_rt)

    def draw_bundle(x_ends, y_end_local, line_color, density_color, is_upper):
        sampled = rng.choice(x_ends, size=min(18, len(x_ends)), replace=False)
        for x_end in sampled:
            x, y = make_trajectory(float(x_end), y_end_local)
            ax.plot(x, y, color=line_color, linewidth=1.0, alpha=0.28, zorder=2)

        mean_x = float(np.mean(x_ends))
        ax.plot(
            [x0, mean_x],
            [0.0, y_end_local],
            color="#2F2F2F",
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=3,
        )
        x_mean, y_mean = make_trajectory(mean_x, y_end_local)
        ax.plot(x_mean, y_mean, color=line_color, linewidth=2.35, zorder=4)

        bins = np.linspace(x0 + 0.30, 0.88, 17)
        hist, edges = np.histogram(x_ends, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        heights = hist.astype(float)
        if heights.max() > 0:
            heights = heights / heights.max() * 0.15
        x_grid = np.linspace(edges[0], edges[-1], 240)
        density = gaussian_kde(x_ends)(x_grid)
        density = density / density.max() * 0.15
        if is_upper:
            ax.bar(
                centers,
                heights,
                width=np.diff(edges) * 0.92,
                bottom=upper,
                color=density_color,
                alpha=0.18,
                edgecolor="none",
                align="center",
                zorder=1,
            )
            ax.plot(x_grid, upper + density, color=line_color, linewidth=1.2, zorder=5)
        else:
            ax.bar(
                centers,
                heights,
                width=np.diff(edges) * 0.92,
                bottom=lower - heights,
                color=density_color,
                alpha=0.18,
                edgecolor="none",
                align="center",
                zorder=1,
            )
            ax.plot(x_grid, lower - density, color=line_color, linewidth=1.2, zorder=5)
        return mean_x

    draw_bundle(correct_x, upper, color, color, True)
    draw_bundle(error_x, lower, error_color, error_color, False)

    rt_correct_med = float(np.median(correct_rt))
    rt_error_med = float(np.median(error_rt))
    x_correct_med = float(rt_to_crossing_x([rt_correct_med])[0])
    x_error_med = float(rt_to_crossing_x([rt_error_med])[0])
    for x_med in [x_correct_med, x_error_med]:
        ax.vlines(
            x_med,
            lower - 0.02,
            center + 0.02,
            colors="#444444",
            linewidth=0.95,
            linestyles=(0, (4, 3)),
            zorder=1,
        )
    ax.text(x_correct_med - 0.012, lower - 0.11, f"{display_rt_label} correct", ha="center", va="top", fontsize=8.1)
    ax.text(x_error_med + 0.014, lower - 0.20, f"{display_rt_label} error", ha="center", va="top", fontsize=8.1)

    summary = (
        f"{feature_label} -> $v$ mean = {slope_row['mean']:.3f}\n"
        f"95% HDI [{slope_row['hdi_2.5']:.3f}, {slope_row['hdi_97.5']:.3f}]"
    )
    ax.text(
        0.52,
        0.89,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#222222",
        bbox=dict(facecolor="white", edgecolor="#D5D9DF", alpha=0.95, pad=0.28),
    )


def save_taskwise_figure(output_dir, set_name, rt_label, group, fig):
    folder = _organized_fig_dir(output_dir) / f"{set_name}_{rt_label.lower()}_sample_style"
    folder.mkdir(parents=True, exist_ok=True)
    label = FILE_LABELS[group]
    png = folder / f"{set_name}_{rt_label.lower()}_{label}_sample.png"
    pdf = folder / f"{set_name}_{rt_label.lower()}_{label}_sample.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def build_set(raw_csv, output_dir, chains, rt_column, focal_predictor, feature_node):
    slopes = load_regression_slopes(output_dir, chains, feature_node)
    response_rt = load_response_rt(raw_csv, rt_column)
    saved = []
    set_name = "visual_z" if focal_predictor == "Visual_Z" else "physical_z"

    for group in PLOT_ORDER:
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.2))
        feature_label = VISUAL_LABEL if focal_predictor == "Visual_Z" else PHYSICAL_LABEL
        plot_single_task_ddm_panel(ax, group, slopes[group], feature_label, rt_column, response_rt[group])
        fig.tight_layout()
        saved.append(save_taskwise_figure(output_dir, set_name, rt_column, group, fig))
        plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="all_dat_merged.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    root = Path.cwd()
    raw_csv = root / args.input
    output_dir = root / args.output_dir

    generated = []
    for rt_column in ["RT_onset", "RT_critical"]:
        generated.extend(build_set(raw_csv, output_dir, args.chains, rt_column, "Visual_Z", "visual_z"))
        generated.extend(build_set(raw_csv, output_dir, args.chains, rt_column, "Physical_Z", "physical_z"))

    for png, pdf in generated:
        print(png)
        print(pdf)


if __name__ == "__main__":
    main()
