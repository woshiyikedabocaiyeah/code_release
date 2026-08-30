#!/usr/bin/env python
"""Plot a double-dissociation style figure for Visual_Z effects."""

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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf
GROUP_MAP = {
    "sensorimotor": "Action",
    "categorization": "Semantic",
    "Voe": "Voe",
}
GROUPS = ["Action", "Semantic", "Voe"]
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
LINESTYLES = {
    "Semantic": "-",
    "Voe": "-",
    "Action": "-",
}
VISUAL_LABEL = r"$\mathrm{Visual}_{z}$"


def load_empirical_visual_acc(input_csv):
    data = pd.read_csv(input_csv)
    data["group"] = data["condition"].map(GROUP_MAP)

    out = {}
    for group in GROUPS:
        subset = (
            data[data["group"] == group]
            .groupby("Visual_Z", as_index=False)
            .agg(acc=("ACC", "mean"), n=("ACC", "size"))
            .sort_values("Visual_Z")
        )
        out[group] = {
            "x": subset["Visual_Z"].to_numpy(dtype=float),
            "acc": subset["acc"].to_numpy(dtype=float),
            "n": subset["n"].to_numpy(dtype=int),
        }
    return out


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
            "q25": float(np.quantile(trace, 0.25)),
            "q75": float(np.quantile(trace, 0.75)),
            "hdi_2.5": float(np.quantile(trace, 0.025)),
            "hdi_97.5": float(np.quantile(trace, 0.975)),
        }
    return slopes


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(colors="#111111", width=1.0, labelsize=10)


def fit_lmm_visual_simple_slopes(input_csv):
    data = pd.read_csv(input_csv)
    data["condition"] = pd.Categorical(data["condition"], categories=["sensorimotor", "categorization", "Voe"])
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
    x_grid = np.linspace(float(data["Visual_Z"].min()), float(data["Visual_Z"].max()), 240)

    def design_row(group, visual_z):
        row = {name: 0.0 for name in fe_params.index}
        row["Intercept"] = 1.0
        row["Visual_Z"] = visual_z
        if group == "Semantic":
            row["C(condition)[T.categorization]"] = 1.0
            row["C(condition)[T.categorization]:Visual_Z"] = visual_z
        elif group == "Voe":
            row["C(condition)[T.Voe]"] = 1.0
            row["C(condition)[T.Voe]:Visual_Z"] = visual_z
        return row

    summaries = {}
    fixed_lines = {}
    slope_vectors = {
        "Action": {"Visual_Z": 1.0},
        "Semantic": {"Visual_Z": 1.0, "C(condition)[T.categorization]:Visual_Z": 1.0},
        "Voe": {"Visual_Z": 1.0, "C(condition)[T.Voe]:Visual_Z": 1.0},
    }

    for group in PLOT_ORDER:
        mean_vals = []
        lo_vals = []
        hi_vals = []
        for visual_z in x_grid:
            row = design_row(group, visual_z)
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


def _draw_visual_lmm_content(ax, empirical, fixed_lines):
    for group in PLOT_ORDER:
        row = fixed_lines[group]
        color = COLORS[group]
        ax.fill_between(row["x"], row["lo"], row["hi"], color=color, alpha=0.10, linewidth=0)
        ax.plot(
            row["x"],
            row["mean"],
            color=color,
            linewidth=2.8,
            linestyle=LINESTYLES[group],
            zorder=2.4 if group == "Voe" else 2.2,
        )

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
            zorder=3,
        )


def _add_visual_slope_notes(ax, slope_summaries):
    slope_note_positions = {
        "Action": 0.44,
        "Voe": 0.28,
        "Semantic": 0.12,
    }
    for group in ["Action", "Voe", "Semantic"]:
        slope = slope_summaries[group]
        label = f"{DISPLAY_LABELS[group]}: b = {slope['estimate']:.3f}"
        ax.text(
            0.98,
            slope_note_positions[group],
            label,
            transform=ax.transAxes,
            fontsize=9.4,
            color=COLORS[group],
            ha="right",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=0.18),
        )


def plot_visual_lmm_panel(ax, empirical, fixed_lines, slope_summaries):
    _draw_visual_lmm_content(ax, empirical, fixed_lines)
    _add_visual_slope_notes(ax, slope_summaries)

    ax.set_title(f"{VISUAL_LABEL} effect on ACC (LMM simple slopes)", fontsize=12, pad=10)
    ax.set_xlabel(VISUAL_LABEL)
    ax.set_ylabel("Predicted ACC (%)")
    ax.set_ylim(72.0, 102.5)
    ax.set_xlim(float(fixed_lines["Semantic"]["x"].min()) - 0.12, float(fixed_lines["Semantic"]["x"].max()) + 0.22)
    style_axis(ax)


def plot_visual_lmm_panel_broken(ax_top, ax_bottom, empirical, fixed_lines, slope_summaries):
    for ax in (ax_top, ax_bottom):
        _draw_visual_lmm_content(ax, empirical, fixed_lines)
        ax.set_xlim(float(fixed_lines["Semantic"]["x"].min()) - 0.12, float(fixed_lines["Semantic"]["x"].max()) + 0.22)
        style_axis(ax)

    _add_visual_slope_notes(ax_bottom, slope_summaries)

    ax_top.set_ylim(72.0, 102.5)
    ax_bottom.set_ylim(0.0, 60.0)

    ax_top.set_title(f"{VISUAL_LABEL} effect on ACC", fontsize=12, pad=10)
    ax_top.set_ylabel("Predicted ACC (%)")
    ax_bottom.set_xlabel(VISUAL_LABEL)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bottom.tick_params(top=False)


def plot_hddm_panel(ax, slopes):
    positions = np.arange(len(PLOT_ORDER), 0, -1)
    for pos, group in zip(positions, PLOT_ORDER):
        row = slopes[group]
        ax.hlines(
            pos,
            row["hdi_2.5"],
            row["hdi_97.5"],
            color="black",
            linewidth=1.8,
            alpha=0.95,
        )
        ax.scatter(row["mean"], pos, s=96, color=COLORS[group], edgecolor="black", linewidth=0.9, zorder=3)

    ax.axvline(0, color="black", linewidth=1.1, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels([DISPLAY_LABELS[group] for group in PLOT_ORDER])
    ax.set_title(f"HDDM {VISUAL_LABEL} effect on drift rate $v$", fontsize=12, pad=10)
    ax.set_xlabel("posterior slope on v")
    ax.set_ylabel("")
    xmin = min(slopes[group]["hdi_2.5"] for group in PLOT_ORDER)
    xmax = max(slopes[group]["hdi_97.5"] for group in PLOT_ORDER)
    span = xmax - xmin
    ax.set_xlim(xmin - span * 0.18, xmax + span * 0.12)
    style_axis(ax)


def plot_hddm_academic_panel(ax, slopes):
    rng = np.random.default_rng(7)
    x0 = 0.06
    upper = 0.78
    lower = -0.78
    center = 0.0
    t = np.linspace(0.0, 1.0, 150)
    abs_max = max(np.max(np.abs(slopes[group]["trace"])) for group in PLOT_ORDER)

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

    ax.set_title(f"HDDM {VISUAL_LABEL} modulation of drift rate $v$", fontsize=12, pad=10)
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
        row = slopes[group]
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
            x_grid = np.linspace(edges[0], edges[-1], 240)
            density = gaussian_kde(x_ends)(x_grid)
            density = density / density.max() * 0.16
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
            x_grid = np.linspace(edges[0], edges[-1], 240)
            density = gaussian_kde(x_ends)(x_grid)
            density = density / density.max() * 0.16
            ax.plot(x_grid, lower - density, color=color, linewidth=1.25, zorder=5)

    ax.annotate(
        "drift rate (v)",
        xy=(0.53, 0.43),
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


def plot_hddm_halfeye_panel(ax, slopes):
    positions = np.arange(len(PLOT_ORDER), 0, -1)
    all_vals = np.concatenate([slopes[group]["trace"] for group in PLOT_ORDER])
    xmin = float(np.min(all_vals))
    xmax = float(np.max(all_vals))
    span = xmax - xmin
    x_grid = np.linspace(xmin - span * 0.08, xmax + span * 0.08, 500)

    for pos, group in zip(positions, PLOT_ORDER):
        row = slopes[group]
        trace = row["trace"]
        kde = gaussian_kde(trace)
        dens = kde(x_grid)
        dens = dens / dens.max() * 0.34

        ax.fill_between(
            x_grid,
            pos,
            pos + dens,
            color=COLORS[group],
            alpha=0.28,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x_grid, pos + dens, color=COLORS[group], linewidth=1.6, zorder=2)

        q25, q50, q75 = np.quantile(trace, [0.25, 0.5, 0.75])
        ax.hlines(pos, row["hdi_2.5"], row["hdi_97.5"], color=COLORS[group], linewidth=2.2, zorder=3)
        ax.hlines(pos, q25, q75, color=COLORS[group], linewidth=5.2, zorder=4)
        ax.scatter(row["mean"], pos, s=92, color="white", edgecolor=COLORS[group], linewidth=2.2, zorder=5)
        ax.scatter(row["mean"], pos, s=20, color=COLORS[group], zorder=6)

    ax.axvline(0, color="black", linewidth=1.1, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels([DISPLAY_LABELS[group] for group in PLOT_ORDER])
    ax.set_title(f"HDDM {VISUAL_LABEL} effect on drift rate $v$", fontsize=12, pad=10)
    ax.set_xlabel("posterior slope on v")
    ax.set_ylabel("")
    ax.set_xlim(xmin - span * 0.18, xmax + span * 0.12)
    ax.set_ylim(0.8, len(PLOT_ORDER) + 0.45)
    style_axis(ax)


def save_summary(output_dir, empirical, slopes, lmm_summaries):
    path = output_dir / "tables" / "visual_z_double_dissociation_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "group", "x_or_node", "value_1", "value_2", "value_3"])
        for group in PLOT_ORDER:
            for x, acc in zip(empirical[group]["x"], empirical[group]["acc"]):
                writer.writerow(["empirical_acc", group, f"{x:.12g}", f"{acc:.12g}", "", ""])
        for group in PLOT_ORDER:
            row = lmm_summaries[group]
            writer.writerow(
                [
                    "lmm_visual_slope",
                    group,
                    "Visual_Z",
                    f"{row['estimate']:.12g}",
                    f"{row['ci_low']:.12g}",
                    f"{row['ci_high']:.12g}",
                ]
            )
        for group in PLOT_ORDER:
            row = slopes[group]
            writer.writerow(
                [
                    "hddm_visual_slope",
                    group,
                    row["node"],
                    f"{row['mean']:.12g}",
                    f"{row['hdi_2.5']:.12g}",
                    f"{row['hdi_97.5']:.12g}",
                ]
            )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="all_dat_merged.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--hddm-style", choices=["forest", "halfeye", "academic"], default="forest")
    args = parser.parse_args()

    root = Path.cwd()
    input_csv = root / args.input
    output_dir = root / args.output_dir

    empirical = load_empirical_visual_acc(input_csv)
    fixed_lines, lmm_summaries = fit_lmm_visual_simple_slopes(input_csv)
    slopes = load_visual_slopes(output_dir, args.chains)

    fig = plt.figure(figsize=(11.5, 5.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.34, 1.0],
        height_ratios=[4.6, 1.4],
        wspace=0.26,
        hspace=0.0,
    )
    ax_left_top = fig.add_subplot(gs[0, 0])
    ax_left_bottom = fig.add_subplot(gs[1, 0], sharex=ax_left_top)
    ax_right = fig.add_subplot(gs[:, 1])

    plot_visual_lmm_panel_broken(ax_left_top, ax_left_bottom, empirical, fixed_lines, lmm_summaries)
    if args.hddm_style == "halfeye":
        plot_hddm_halfeye_panel(ax_right, slopes)
    elif args.hddm_style == "academic":
        plot_hddm_academic_panel(ax_right, slopes)
    else:
        plot_hddm_panel(ax_right, slopes)

    fig.suptitle(
        "Visual complexity shows a double dissociation across observed accuracy and latent drift dynamics",
        fontsize=13,
        y=0.98,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[group],
            lw=3.0,
            linestyle=LINESTYLES[group],
            marker="o",
            markersize=6.0,
            label=DISPLAY_LABELS[group],
        )
        for group in PLOT_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=10,
        handlelength=1.4,
        handleheight=1.2,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    fig_dir = _organized_fig_dir(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix_map = {"forest": "", "halfeye": "_halfeye", "academic": "_academic"}
    suffix = suffix_map[args.hddm_style]
    png_path = fig_dir / f"visual_z_double_dissociation{suffix}.png"
    pdf_path = fig_dir / f"visual_z_double_dissociation{suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = save_summary(output_dir, empirical, slopes, lmm_summaries)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
