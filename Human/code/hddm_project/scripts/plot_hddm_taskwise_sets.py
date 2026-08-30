#!/usr/bin/env python
"""Create task-wise HDDM figures: one task per figure, three figures per set."""

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
from matplotlib import colors as mcolors
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


def lighten_color(color, mix=0.45):
    base = np.array(mcolors.to_rgb(color), dtype=float)
    white = np.ones(3, dtype=float)
    return tuple(base * (1.0 - mix) + white * mix)


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


def load_response_rt(prepared_csv):
    data = pd.read_csv(prepared_csv)
    out = {}
    for group in PLOT_ORDER:
        subset = data[data["group"] == group]
        out[group] = {
            1: np.asarray(subset.loc[subset["response"] == 1, "rt"], dtype=float),
            0: np.asarray(subset.loc[subset["response"] == 0, "rt"], dtype=float),
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
    slope_vectors = {
        "Action": {"Physical_Z": 1.0},
        "Semantic": {"Physical_Z": 1.0, "C(condition)[T.categorization]:Physical_Z": 1.0},
        "Voe": {"Physical_Z": 1.0, "C(condition)[T.Voe]:Physical_Z": 1.0},
    }
    summaries = {}

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


def plot_visual_behavior_panel(ax, group, behavior):
    row = behavior[group]
    color = COLORS[group]
    curve = PchipInterpolator(row["x"], row["acc"])
    x_dense = np.linspace(float(np.min(row["x"])), float(np.max(row["x"])), 300)
    y_dense = curve(x_dense)

    ax.scatter(
        row["x"],
        row["acc"],
        s=60,
        color=color,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    ax.plot(x_dense, y_dense, color=color, linewidth=2.8, zorder=2)
    ax.set_title(f"{DISPLAY_LABELS[group]}: ACC across {VISUAL_LABEL}", fontsize=12, pad=10)
    ax.set_xlabel(VISUAL_LABEL)
    ax.set_ylabel("ACC (%)")
    ax.set_ylim(72, 97)
    style_axis(ax)


def plot_physical_behavior_panel(ax, group, lines, slope_summary):
    row = lines[group]
    color = COLORS[group]
    ax.fill_between(row["x"], row["lo"], row["hi"], color=color, alpha=0.14, linewidth=0)
    ax.plot(row["x"], row["mean"], color=color, linewidth=2.8)

    label = f"b = {slope_summary['estimate']:.3f}"
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
    ax.set_title(f"{DISPLAY_LABELS[group]}: ACC across {PHYSICAL_LABEL}", fontsize=12, pad=10)
    ax.set_xlabel(PHYSICAL_LABEL)
    ax.set_ylabel("Predicted ACC (%)")
    ax.set_xlim(float(row["x"].min()) - 0.1, float(row["x"].max()) + 0.26)
    all_vals = np.concatenate([lines[g]["mean"] for g in PLOT_ORDER])
    ymin = np.floor((all_vals.min() - 4.0) / 5.0) * 5.0
    ymax = np.ceil((all_vals.max() + 3.0) / 5.0) * 5.0
    ax.set_ylim(ymin, ymax)
    style_axis(ax)


def plot_single_task_ddm_panel(ax, group, slope_row, feature_label, response_rt):
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
        y = base + noise * 0.085 * (1.0 - tau) * (0.55 + 0.45 * tau)
        y[0] = 0.0
        y[-1] = y_end_local
        return x, y

    ax.set_title(f"{DISPLAY_LABELS[group]}: HDDM {feature_label} -> drift rate $v$", fontsize=12, pad=10)
    ax.set_xlim(-0.03, 1.10)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.hlines([upper, center, lower], 0.0, 0.93, colors="#777777", linewidth=1.0, zorder=0)
    ax.scatter([x0], [0.0], s=20, color="#111111", zorder=6)
    ax.text(-0.01, upper + 0.02, "correct", ha="right", va="bottom", fontsize=9.2, color="#222222")
    ax.text(-0.01, lower - 0.02, "error", ha="right", va="top", fontsize=9.2, color="#222222")

    ax.annotate(
        "",
        xy=(1.00, upper),
        xytext=(1.00, lower),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0, linestyle=(0, (4, 3))),
    )
    ax.text(1.03, 0.0, "threshold (a)", rotation=90, ha="left", va="center", fontsize=9.0)

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
        draw_n = min(18, len(x_ends))
        sampled = rng.choice(x_ends, size=draw_n, replace=False)
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

    mean_correct_x = draw_bundle(correct_x, upper, color, color, True)
    mean_error_x = draw_bundle(error_x, lower, error_color, error_color, False)

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

    ax.text(x_correct_med - 0.012, lower - 0.11, "RT_correct", ha="center", va="top", fontsize=8.1)
    ax.text(x_error_med + 0.014, lower - 0.20, "RT_error", ha="center", va="top", fontsize=8.1)

    summary = (
        f"mean = {slope_row['mean']:.3f}\n"
        f"95% HDI [{slope_row['hdi_2.5']:.3f}, {slope_row['hdi_97.5']:.3f}]"
    )
    ax.text(
        0.50,
        0.95,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#222222",
        bbox=dict(facecolor="white", edgecolor="#D5D9DF", alpha=0.95, pad=0.28),
    )


def save_taskwise_figure(output_dir, set_name, group, fig):
    folder = _organized_fig_dir(output_dir) / f"{set_name}_taskwise"
    folder.mkdir(parents=True, exist_ok=True)
    label = FILE_LABELS[group]
    png = folder / f"{set_name}_{label}.png"
    pdf = folder / f"{set_name}_{label}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def build_visual_set(input_csv, output_dir, chains):
    behavior = load_visual_behavior(input_csv)
    slopes = load_regression_slopes(output_dir, chains, "visual_z")
    response_rt = load_response_rt(output_dir / "prepared_hddm_data.csv")
    saved = []
    for group in PLOT_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
        plot_visual_behavior_panel(axes[0], group, behavior)
        plot_single_task_ddm_panel(axes[1], group, slopes[group], VISUAL_LABEL, response_rt[group])
        fig.suptitle(f"Visual complexity set: {DISPLAY_LABELS[group]} task", fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        saved.append(save_taskwise_figure(output_dir, "visual_z", group, fig))
        plt.close(fig)
    return saved


def build_physical_set(input_csv, output_dir, chains):
    lines, lmm_summaries = fit_physical_lmm(input_csv)
    slopes = load_regression_slopes(output_dir, chains, "physical_z")
    response_rt = load_response_rt(output_dir / "prepared_hddm_data.csv")
    saved = []
    for group in PLOT_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
        plot_physical_behavior_panel(axes[0], group, lines, lmm_summaries[group])
        plot_single_task_ddm_panel(axes[1], group, slopes[group], PHYSICAL_LABEL, response_rt[group])
        fig.suptitle(f"Physical uncertainty set: {DISPLAY_LABELS[group]} task", fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        saved.append(save_taskwise_figure(output_dir, "physical_z", group, fig))
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

    print("Visual_Z taskwise figures:")
    for png, pdf in visual_files:
        print(png)
        print(pdf)
    print("Physical_Z taskwise figures:")
    for png, pdf in physical_files:
        print(png)
        print(pdf)


if __name__ == "__main__":
    main()
