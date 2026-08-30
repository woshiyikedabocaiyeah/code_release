#!/usr/bin/env python
"""Plot the reversal of physical uncertainty across behavior and HDDM slopes."""

import argparse
import csv
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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
COLORS = {
    "Semantic": "#4EA6C6",
    "Voe": "#1F987B",
    "Action": "#E3362A",
}
PHYSICAL_LABEL = r"$\mathrm{Physical}_{z}$"
MODEL_ORDER = ["sensorimotor", "categorization", "Voe"]


def load_empirical_physical_acc(input_csv):
    data = pd.read_csv(input_csv)
    data["group"] = data["condition"].map(GROUP_MAP)

    out = {}
    for group in PLOT_ORDER:
        subset = (
            data[data["group"] == group]
            .groupby("Physical_Z", as_index=False)
            .agg(acc=("ACC", "mean"), n=("ACC", "size"))
            .sort_values("Physical_Z")
        )
        out[group] = {
            "x": subset["Physical_Z"].to_numpy(dtype=float),
            "acc": subset["acc"].to_numpy(dtype=float),
            "n": subset["n"].to_numpy(dtype=int),
        }
    return out


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(colors="#111111", width=1.0, labelsize=10)


def pooled_trace(models, node_name):
    return np.concatenate(
        [np.asarray(model.nodes_db.node[node_name].trace(), dtype=float) for model in models]
    )


def load_models(output_dir, model_name, chains):
    models = []
    model_dir = output_dir / "models" / model_name
    for chain in range(1, chains + 1):
        models.append(hddm.load(str(model_dir / f"{model_name}_chain{chain}.pkl")))
    return models


def fit_lmm_simple_slopes(input_csv):
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

    summaries = {}
    fixed_lines = {}
    slope_vectors = {
        "Action": {"Physical_Z": 1.0},
        "Semantic": {"Physical_Z": 1.0, "C(condition)[T.categorization]:Physical_Z": 1.0},
        "Voe": {"Physical_Z": 1.0, "C(condition)[T.Voe]:Physical_Z": 1.0},
    }

    for group in PLOT_ORDER:
        mean_vals = []
        lo_vals = []
        hi_vals = []
        for physical_z in x_grid:
            row = design_row(group, physical_z)
            vec = np.array([row[name] for name in fe_params.index], dtype=float)
            mean = float(vec @ fe_params.values)
            se = float(np.sqrt(vec @ cov_fe.values @ vec))
            mean_vals.append(mean)
            lo_vals.append(mean - 1.96 * se)
            hi_vals.append(mean + 1.96 * se)

        fixed_lines[group] = {
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

    return fixed_lines, summaries


def load_hddm_physical_slopes(output_dir, chains):
    models = load_models(output_dir, "regression", chains)
    node_names = {
        "Action": "v_C(group)[Action]:physical_z",
        "Semantic": "v_C(group)[Semantic]:physical_z",
        "Voe": "v_C(group)[Voe]:physical_z",
    }
    summaries = {}
    for group, node_name in node_names.items():
        trace = pooled_trace(models, node_name)
        summaries[group] = {
            "node": node_name,
            "trace": trace,
            "mean": float(np.mean(trace)),
            "q25": float(np.quantile(trace, 0.25)),
            "q75": float(np.quantile(trace, 0.75)),
            "hdi_2.5": float(np.quantile(trace, 0.025)),
            "hdi_97.5": float(np.quantile(trace, 0.975)),
        }
    return summaries


def plot_lmm_panel(ax, empirical, fixed_lines, slope_summaries):
    label_offsets = {
        "Action": (0.30, 1.2),
        "Voe": (0.30, 0.4),
        "Semantic": (0.30, -1.6),
    }
    for group in PLOT_ORDER:
        row = fixed_lines[group]
        color = COLORS[group]
        ax.fill_between(row["x"], row["lo"], row["hi"], color=color, alpha=0.14, linewidth=0)
        ax.plot(row["x"], row["mean"], color=color, linewidth=2.6, zorder=3)

        x_points = empirical[group]["x"]
        y_points = empirical[group]["acc"] * 100.0
        ax.scatter(
            x_points,
            y_points,
            s=9,
            alpha=0.92,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            zorder=4,
        )

        slope = slope_summaries[group]
        label = f"{DISPLAY_LABELS[group]}: b = {slope['estimate']:.3f}"
        x_anchor = float(row["x"][-20])
        y_anchor = float(row["mean"][-20])
        dx, dy = label_offsets[group]
        ax.annotate(
            label,
            xy=(x_anchor, y_anchor),
            xytext=(x_anchor + dx, y_anchor + dy),
            fontsize=9.1,
            color=color,
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.2),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=0, shrinkB=0),
        )

    ax.set_title(f"{PHYSICAL_LABEL} effect on ACC", fontsize=12, pad=10)
    ax.set_xlabel(PHYSICAL_LABEL)
    ax.set_ylabel("Predicted ACC (%)")
    all_vals = np.concatenate([fixed_lines[group]["mean"] for group in PLOT_ORDER])
    ymax = np.ceil((all_vals.max() + 2.0) / 5.0) * 5.0
    ax.set_ylim(72.0, ymax)
    ax.set_xlim(float(fixed_lines["Semantic"]["x"].min()) - 0.15, float(fixed_lines["Semantic"]["x"].max()) + 0.45)
    style_axis(ax)


def plot_hddm_panel(ax, slope_summaries):
    rng = np.random.default_rng(11)
    x0 = 0.06
    upper = 0.78
    lower = -0.78
    center = 0.0
    t = np.linspace(0.0, 1.0, 150)
    abs_max = max(np.max(np.abs(slope_summaries[group]["trace"])) for group in PLOT_ORDER)

    def slope_to_crossing_x(trace):
        magnitude = np.clip(np.abs(trace) / abs_max, 0.0, 1.0)
        return x0 + 0.44 + 0.35 * (1.0 - np.power(magnitude, 0.75))

    def make_trajectory(x_end, y_end):
        tau = t
        x = x0 + (x_end - x0) * tau
        base = y_end * np.power(tau, 1.02)
        noise = np.cumsum(rng.normal(0.0, 1.0, size=tau.size))
        bridge = np.linspace(noise[0], noise[-1], tau.size)
        noise = noise - bridge
        scale = np.max(np.abs(noise))
        if scale > 0:
            noise = noise / scale
        y = base + noise * 0.085 * (1.0 - tau) * (0.55 + 0.45 * tau)
        y[0] = 0.0
        y[-1] = y_end
        return x, y

    ax.set_title(f"HDDM {PHYSICAL_LABEL} modulation of drift rate $v$", fontsize=12, pad=10)
    ax.set_xlim(-0.03, 1.10)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.hlines([upper, center, lower], 0.0, 0.93, colors="#777777", linewidth=1.0, zorder=0)
    ax.scatter([x0], [0.0], s=20, color="#111111", zorder=6)

    ax.annotate(
        "",
        xy=(1.00, upper),
        xytext=(1.00, lower),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0, linestyle=(0, (4, 3))),
    )
    ax.text(1.03, 0.0, "threshold (a)", rotation=90, ha="left", va="center", fontsize=9.2)
    ax.text(0.89, upper + 0.03, "upper threshold", ha="left", va="bottom", fontsize=8.8, style="italic")
    ax.text(0.89, lower - 0.03, "lower threshold", ha="left", va="top", fontsize=8.8, style="italic")

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
        fontsize=8.6,
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
        fontsize=8.8,
        style="italic",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.22),
    )

    for group in PLOT_ORDER:
        row = slope_summaries[group]
        color = COLORS[group]
        y_end = upper if row["mean"] >= 0 else lower
        x_ends = slope_to_crossing_x(row["trace"])
        sampled_x_ends = rng.choice(x_ends, size=min(16, len(x_ends)), replace=False)

        for x_end in sampled_x_ends:
            x, y = make_trajectory(float(x_end), y_end)
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.30, zorder=2)

        mean_x_end = float(np.mean(x_ends))
        ax.plot(
            [x0, mean_x_end],
            [0.0, y_end],
            color="#2F2F2F",
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=3,
        )
        x_mean, y_mean = make_trajectory(mean_x_end, y_end)
        ax.plot(x_mean, y_mean, color=color, linewidth=2.4, zorder=4)

        bins = np.linspace(x0 + 0.36, 0.92, 18)
        hist, edges = np.histogram(x_ends, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        heights = hist.astype(float)
        if heights.max() > 0:
            heights = heights / heights.max() * 0.16

        x_grid = np.linspace(edges[0], edges[-1], 240)
        density = gaussian_kde(x_ends)(x_grid)
        density = density / density.max() * 0.16
        if y_end > 0:
            ax.bar(
                centers,
                heights,
                width=np.diff(edges) * 0.92,
                bottom=upper,
                color=color,
                alpha=0.18,
                edgecolor="none",
                align="center",
                zorder=1,
            )
            ax.plot(x_grid, upper + density, color=color, linewidth=1.25, zorder=5)
        else:
            ax.bar(
                centers,
                heights,
                width=np.diff(edges) * 0.92,
                bottom=lower - heights,
                color=color,
                alpha=0.18,
                edgecolor="none",
                align="center",
                zorder=1,
            )
            ax.plot(x_grid, lower - density, color=color, linewidth=1.25, zorder=5)

    ax.annotate(
        "drift rate (v)",
        xy=(0.53, 0.43 if slope_summaries["Action"]["mean"] >= 0 else -0.43),
        xytext=(0.34, 0.57),
        fontsize=8.8,
        color="#333333",
        style="italic",
        rotation=18,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.22),
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.9),
    )


def save_summary(output_dir, lmm_summaries, hddm_summaries):
    path = output_dir / "tables" / "physical_z_reversal_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "group", "estimate", "ci_low", "ci_high", "node"])
        for group in PLOT_ORDER:
            row = lmm_summaries[group]
            writer.writerow(
                ["lmm_simple_slope", DISPLAY_LABELS[group], row["estimate"], row["ci_low"], row["ci_high"], ""]
            )
        for group in PLOT_ORDER:
            row = hddm_summaries[group]
            writer.writerow(
                [
                    "hddm_posterior_slope",
                    DISPLAY_LABELS[group],
                    row["mean"],
                    row["hdi_2.5"],
                    row["hdi_97.5"],
                    row["node"],
                ]
            )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="all_dat_merged.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    root = Path.cwd()
    input_csv = root / args.input
    output_dir = root / args.output_dir

    empirical = load_empirical_physical_acc(input_csv)
    fixed_lines, lmm_summaries = fit_lmm_simple_slopes(input_csv)
    hddm_summaries = load_hddm_physical_slopes(output_dir, args.chains)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    plot_lmm_panel(axes[0], empirical, fixed_lines, lmm_summaries)
    plot_hddm_panel(axes[1], hddm_summaries)

    fig.suptitle(
        r"Reversal of $\mathrm{Physical}_{z}$ effect across observed accuracy and latent evidence accumulation",
        fontsize=13,
        y=0.98,
    )
    handles = [
        Line2D([0], [0], color=COLORS[group], lw=3.2, label=DISPLAY_LABELS[group]) for group in PLOT_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    fig_dir = _organized_fig_dir(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "physical_z_reversal_effect.png"
    pdf_path = fig_dir / "physical_z_reversal_effect.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = save_summary(output_dir, lmm_summaries, hddm_summaries)
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
