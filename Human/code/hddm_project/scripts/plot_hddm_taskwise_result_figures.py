#!/usr/bin/env python
"""Create task-wise HDDM result figures with posterior distributions."""

import argparse
import csv
import warnings
from collections import defaultdict
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
import statsmodels.formula.api as smf
from scipy.interpolate import PchipInterpolator
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


def load_visual_behavior(input_csv):
    grouped = defaultdict(lambda: {"x": [], "acc": []})
    all_x = []
    with open(input_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group = GROUP_MAP[row["condition"]]
            x = float(row["Visual_Z"])
            acc = float(row["ACC"])
            grouped[group]["x"].append(x)
            grouped[group]["acc"].append(acc)
            all_x.append(x)

    cuts = np.quantile(np.array(all_x, dtype=float), np.linspace(0, 1, 6))
    out = {}
    for group in PLOT_ORDER:
        x = np.array(grouped[group]["x"], dtype=float)
        y = np.array(grouped[group]["acc"], dtype=float)
        centers = []
        means = []
        for idx in range(len(cuts) - 1):
            lo = cuts[idx]
            hi = cuts[idx + 1]
            if idx == len(cuts) - 2:
                mask = (x >= lo) & (x <= hi)
            else:
                mask = (x >= lo) & (x < hi)
            centers.append(float(np.mean(x[mask])))
            means.append(float(np.mean(y[mask])))
        out[group] = {
            "x": np.array(centers, dtype=float),
            "acc": np.array(means, dtype=float) * 100.0,
        }
    return out


def fit_physical_lmm(input_csv):
    data = pd.read_csv(input_csv)
    data["condition"] = pd.Categorical(data["condition"], categories=MODEL_ORDER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "ACC ~ C(condition) * Visual_Z + C(condition) * Physical_Z",
            data=data,
            groups=data["Subject"],
        )
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)

    fe_params = result.fe_params.copy()
    cov_fe = result.cov_params().loc[fe_params.index, fe_params.index]
    x_grid = np.linspace(float(data["Physical_Z"].min()), float(data["Physical_Z"].max()), 240)

    def design_row(group, physical_z):
        row = {name: 0.0 for name in fe_params.index}
        row["Intercept"] = 1.0
        row["Physical_Z"] = physical_z
        if group == "Semantic":
            row["C(condition)[T.categorization]"] = 1.0
            row["C(condition)[T.categorization]:Physical_Z"] = physical_z
        elif group == "Voe":
            row["C(condition)[T.Voe]"] = 1.0
            row["C(condition)[T.Voe]:Physical_Z"] = physical_z
        return row

    lines = {}
    summaries = {}
    slope_vectors = {
        "Action": {"Physical_Z": 1.0},
        "Semantic": {"Physical_Z": 1.0, "C(condition)[T.categorization]:Physical_Z": 1.0},
        "Voe": {"Physical_Z": 1.0, "C(condition)[T.Voe]:Physical_Z": 1.0},
    }

    for group in PLOT_ORDER:
        mean_vals, lo_vals, hi_vals = [], [], []
        for physical_z in x_grid:
            row = design_row(group, physical_z)
            vec = np.array([row[name] for name in fe_params.index], dtype=float)
            mean = float(vec @ fe_params.values)
            se = float(np.sqrt(vec @ cov_fe.values @ vec))
            mean_vals.append(mean)
            lo_vals.append(mean - 1.96 * se)
            hi_vals.append(mean + 1.96 * se)
        lines[group] = {
            "x": x_grid,
            "mean": np.array(mean_vals) * 100.0,
            "lo": np.array(lo_vals) * 100.0,
            "hi": np.array(hi_vals) * 100.0,
        }

        slope_row = {name: 0.0 for name in fe_params.index}
        for name, weight in slope_vectors[group].items():
            slope_row[name] = weight
        vec = np.array([slope_row[name] for name in fe_params.index], dtype=float)
        est = float(vec @ fe_params.values)
        se = float(np.sqrt(vec @ cov_fe.values @ vec))
        summaries[group] = {
            "estimate": est,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
        }
    return lines, summaries


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
            "q25": float(np.quantile(trace, 0.25)),
            "q75": float(np.quantile(trace, 0.75)),
            "hdi_2.5": float(np.quantile(trace, 0.025)),
            "hdi_97.5": float(np.quantile(trace, 0.975)),
        }
    return out


def plot_visual_behavior(ax, group, behavior):
    row = behavior[group]
    color = COLORS[group]
    curve = PchipInterpolator(row["x"], row["acc"])
    x_dense = np.linspace(float(np.min(row["x"])), float(np.max(row["x"])), 300)
    y_dense = curve(x_dense)
    ax.scatter(row["x"], row["acc"], s=62, color=color, edgecolor="black", linewidth=0.7, zorder=3)
    ax.plot(x_dense, y_dense, color=color, linewidth=2.8, zorder=2)
    ax.set_title("Behavioral pattern", fontsize=12, pad=10)
    ax.set_xlabel(VISUAL_LABEL)
    ax.set_ylabel("ACC (%)")
    ax.set_ylim(72, 97)
    style_axis(ax)


def plot_physical_behavior(ax, group, lines, summary):
    row = lines[group]
    color = COLORS[group]
    ax.fill_between(row["x"], row["lo"], row["hi"], color=color, alpha=0.14, linewidth=0)
    ax.plot(row["x"], row["mean"], color=color, linewidth=2.8)
    label = f"b = {summary['estimate']:.3f}"
    x_anchor = float(row["x"][-24])
    y_anchor = float(row["mean"][-24])
    offset_map = {"Action": (0.18, 1.0), "Voe": (0.18, 0.4), "Semantic": (0.18, -1.4)}
    dx, dy = offset_map[group]
    ax.annotate(
        label,
        xy=(x_anchor, y_anchor),
        xytext=(x_anchor + dx, y_anchor + dy),
        fontsize=9.2,
        color=color,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.18),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=0, shrinkB=0),
    )
    ax.set_title("Behavioral pattern", fontsize=12, pad=10)
    ax.set_xlabel(PHYSICAL_LABEL)
    ax.set_ylabel("Predicted ACC (%)")
    ax.set_xlim(float(row["x"].min()) - 0.1, float(row["x"].max()) + 0.26)
    all_vals = np.concatenate([lines[g]["mean"] for g in PLOT_ORDER])
    ymin = np.floor((all_vals.min() - 4.0) / 5.0) * 5.0
    ymax = np.ceil((all_vals.max() + 3.0) / 5.0) * 5.0
    ax.set_ylim(ymin, ymax)
    style_axis(ax)


def plot_posterior_density(ax, group, slope_row, feature_label):
    color = COLORS[group]
    trace = slope_row["trace"]
    xmin = float(np.min(trace))
    xmax = float(np.max(trace))
    span = xmax - xmin if xmax > xmin else 1.0
    x_grid = np.linspace(xmin - span * 0.15, xmax + span * 0.15, 500)
    density = gaussian_kde(trace)(x_grid)

    hdi_mask = (x_grid >= slope_row["hdi_2.5"]) & (x_grid <= slope_row["hdi_97.5"])
    iqr_mask = (x_grid >= slope_row["q25"]) & (x_grid <= slope_row["q75"])

    ax.fill_between(x_grid, 0, density, color=color, alpha=0.12, zorder=1)
    ax.fill_between(x_grid[hdi_mask], 0, density[hdi_mask], color=color, alpha=0.22, zorder=2)
    ax.fill_between(x_grid[iqr_mask], 0, density[iqr_mask], color=color, alpha=0.34, zorder=3)
    ax.plot(x_grid, density, color=color, linewidth=2.2, zorder=4)

    ax.axvline(0, color="#222222", linestyle="--", linewidth=1.1, zorder=0)
    ax.axvline(slope_row["mean"], color="#222222", linewidth=1.2, zorder=5)
    ymax = float(np.max(density))
    ax.scatter([slope_row["mean"]], [ymax * 0.06], s=42, color="white", edgecolor="#222222", linewidth=0.9, zorder=6)

    textbox = (
        f"mean = {slope_row['mean']:.3f}\n"
        f"95% HDI [{slope_row['hdi_2.5']:.3f}, {slope_row['hdi_97.5']:.3f}]"
    )
    # Keep the summary box off the density peak: place it on the blank side of the panel.
    if slope_row["mean"] < 0:
        zero_axes = (0.0 - x_grid[0]) / (x_grid[-1] - x_grid[0])
        text_x = min(0.88, zero_axes - 0.04)
        text_ha = "right"
    else:
        text_x = 0.04
        text_ha = "left"
    ax.text(
        text_x,
        0.95,
        textbox,
        transform=ax.transAxes,
        ha=text_ha,
        va="top",
        fontsize=9.4,
        color="#222222",
        bbox=dict(facecolor="white", edgecolor="#D5D9DF", alpha=0.96, pad=0.28),
    )

    ax.set_title("HDDM posterior distribution", fontsize=12, pad=10)
    ax.set_xlabel(f"{feature_label} effect on drift rate $v$")
    ax.set_ylabel("Density")
    style_axis(ax)


def save_figure(output_dir, set_name, group, fig):
    folder = _organized_fig_dir(output_dir) / f"{set_name}_taskwise_results"
    folder.mkdir(parents=True, exist_ok=True)
    label = FILE_LABELS[group]
    png = folder / f"{set_name}_{label}_result.png"
    pdf = folder / f"{set_name}_{label}_result.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def build_visual_set(input_csv, output_dir, chains):
    behavior = load_visual_behavior(input_csv)
    slopes = load_regression_slopes(output_dir, chains, "visual_z")
    saved = []
    for group in PLOT_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
        plot_visual_behavior(axes[0], group, behavior)
        plot_posterior_density(axes[1], group, slopes[group], VISUAL_LABEL)
        fig.suptitle(f"Visual complexity set: {DISPLAY_LABELS[group]} task", fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        saved.append(save_figure(output_dir, "visual_z", group, fig))
        plt.close(fig)
    return saved


def build_physical_set(input_csv, output_dir, chains):
    lines, lmm_summaries = fit_physical_lmm(input_csv)
    slopes = load_regression_slopes(output_dir, chains, "physical_z")
    saved = []
    for group in PLOT_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
        plot_physical_behavior(axes[0], group, lines, lmm_summaries[group])
        plot_posterior_density(axes[1], group, slopes[group], PHYSICAL_LABEL)
        fig.suptitle(f"Physical uncertainty set: {DISPLAY_LABELS[group]} task", fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        saved.append(save_figure(output_dir, "physical_z", group, fig))
        plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="all_dat_merged.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    root = Path.cwd()
    input_csv = root / args.input
    output_dir = root / args.output_dir

    visual_files = build_visual_set(input_csv, output_dir, args.chains)
    physical_files = build_physical_set(input_csv, output_dir, args.chains)

    print("Visual_Z result figures:")
    for png, pdf in visual_files:
        print(png)
        print(pdf)
    print("Physical_Z result figures:")
    for png, pdf in physical_files:
        print(png)
        print(pdf)


if __name__ == "__main__":
    main()
