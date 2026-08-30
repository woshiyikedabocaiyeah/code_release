#!/usr/bin/env python3

from __future__ import annotations

import math
import pickle
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "hddm_results_4chains_2000samples"
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["Semantic", "Intuitive", "Action"]
COLORS = {
    "Semantic": "#4FA3C7",
    "Intuitive": "#2C9C84",
    "Action": "#E63E32",
}

CONDITION_TO_GROUP = {
    "categorization": "Semantic",
    "sensorimotor": "Action",
    "Voe": "Intuitive",
}

TRACE_KEYS = {
    "visual_z": {
        "Semantic": "v_C(group, Treatment('Semantic'))[Semantic]:visual_z",
        "Action": "v_C(group, Treatment('Semantic'))[Action]:visual_z",
        "Intuitive": "v_C(group, Treatment('Semantic'))[Intuitive]:visual_z",
    },
    "physical_z": {
        "Semantic": "v_C(group, Treatment('Semantic'))[Semantic]:physical_z",
        "Action": "v_C(group, Treatment('Semantic'))[Action]:physical_z",
        "Intuitive": "v_C(group, Treatment('Semantic'))[Intuitive]:physical_z",
    },
}


def load_trace(path: Path) -> dict:
    return pickle.loads(path.read_bytes())


def combined_trace(rt_key: str, key: str) -> np.ndarray:
    arrays = []
    for chain in range(1, 5):
        trace_path = RESULTS_DIR / rt_key / "regression" / f"chain_{chain}_trace.db"
        trace = load_trace(trace_path)
        arrays.append(np.asarray(trace[key][0], dtype=float))
    return np.concatenate(arrays)


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p))


def build_subject_bin_points(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    return build_subject_bin_mean_points(
        df=df,
        feature=feature,
        outcome_col="response",
        y_col="acc_pct",
        scale=100.0,
    )


def build_subject_bin_mean_points(
    df: pd.DataFrame,
    feature: str,
    outcome_col: str,
    y_col: str,
    scale: float,
) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        group_df = df[df["group"] == group].copy()
        for subj_idx, subj_df in group_df.groupby("subj_idx"):
            try:
                bins = pd.qcut(subj_df[feature], q=5, duplicates="drop")
            except ValueError:
                continue
            agg = (
                subj_df.groupby(bins, observed=True)
                .agg(x=(feature, "mean"), y=(outcome_col, "mean"))
                .reset_index(drop=True)
            )
            agg[y_col] = agg.pop("y") * scale
            agg["group"] = group
            agg["subj_idx"] = subj_idx
            rows.append(agg)
    scatter = pd.concat(rows, ignore_index=True)
    return scatter


def load_raw_behavior() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "all_dat_merged.csv")
    df["group"] = df["condition"].map(CONDITION_TO_GROUP)
    df["visual_z"] = pd.to_numeric(df["Visual_Z"], errors="coerce")
    df["physical_z"] = pd.to_numeric(df["Physical_Z"], errors="coerce")
    df["rt_onset"] = pd.to_numeric(df["RT_onset"], errors="coerce")
    df["rt_critical"] = pd.to_numeric(df["RT_critical"], errors="coerce")
    return df


def build_stimulus_rt_points(raw_df: pd.DataFrame, rt_col: str) -> pd.DataFrame:
    work = raw_df.dropna(subset=["Video", "group", "visual_z", rt_col]).copy()
    if rt_col == "rt_critical":
        work = work[work[rt_col] >= 0].copy()
    return (
        work.groupby(["Video", "group"], as_index=False)
        .agg(
            x=("visual_z", "mean"),
            rt=(rt_col, "mean"),
        )
        .reset_index(drop=True)
    )


def build_empirical_feature_points(
    raw_df: pd.DataFrame,
    feature_col: str,
    outcome_col: str,
    *,
    value_scale: float = 1.0,
    filter_rt: bool = False,
) -> pd.DataFrame:
    work = raw_df.dropna(subset=["condition", feature_col, outcome_col]).copy()
    if filter_rt:
        work = work[work[outcome_col] > 0.2].copy()
    work["group"] = work["condition"].map(CONDITION_TO_GROUP)

    points = (
        work.groupby(["group", feature_col], as_index=False)
        .agg(value=(outcome_col, "mean"), n=(outcome_col, "size"))
        .sort_values(["group", feature_col])
        .reset_index(drop=True)
    )
    points["x"] = points[feature_col].astype(float)
    points["value"] = points["value"].astype(float) * value_scale
    return points[["group", "x", "value", "n"]]


def fit_lmm_feature_effect(
    raw_df: pd.DataFrame,
    outcome_col: str,
    feature_col: str,
    covariate_col: str,
    *,
    prediction_scale: float = 1.0,
    filter_rt: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels is required for the mixed-model behavior panels. "
            "Run this script with the hddm_project .venv39 Python or install statsmodels."
        ) from exc

    data = raw_df.dropna(subset=["Subject", "condition", feature_col, covariate_col, outcome_col]).copy()
    if filter_rt:
        data = data[data[outcome_col] > 0.2].copy()
    data["condition"] = pd.Categorical(
        data["condition"],
        categories=["sensorimotor", "categorization", "Voe"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            f"{outcome_col} ~ C(condition) * {feature_col} + C(condition) * {covariate_col}",
            data=data,
            groups=data["Subject"],
        )
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)

    fe_params = result.fe_params.copy()
    cov_fe = result.cov_params().loc[fe_params.index, fe_params.index]
    x_grid = np.linspace(float(data[feature_col].min()), float(data[feature_col].max()), 240)

    def design_row(group: str, feature_value: float) -> dict[str, float]:
        row = {name: 0.0 for name in fe_params.index}
        row["Intercept"] = 1.0
        row[feature_col] = feature_value
        if group == "Semantic":
            row["C(condition)[T.categorization]"] = 1.0
            row[f"C(condition)[T.categorization]:{feature_col}"] = feature_value
        elif group == "Intuitive":
            row["C(condition)[T.Voe]"] = 1.0
            row[f"C(condition)[T.Voe]:{feature_col}"] = feature_value
        return row

    slope_vectors = {
        "Action": {feature_col: 1.0},
        "Semantic": {
            feature_col: 1.0,
            f"C(condition)[T.categorization]:{feature_col}": 1.0,
        },
        "Intuitive": {
            feature_col: 1.0,
            f"C(condition)[T.Voe]:{feature_col}": 1.0,
        },
    }

    fixed_lines: dict[str, dict[str, np.ndarray]] = {}
    slope_summaries: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        mean_vals = []
        low_vals = []
        high_vals = []
        for feature_value in x_grid:
            row = design_row(group, float(feature_value))
            vec = np.array([row[name] for name in fe_params.index], dtype=float)
            mean = float(vec @ fe_params.values)
            se = float(np.sqrt(vec @ cov_fe.values @ vec))
            mean_vals.append(mean)
            low_vals.append(mean - 1.96 * se)
            high_vals.append(mean + 1.96 * se)

        fixed_lines[group] = {
            "x": x_grid,
            "mean": np.array(mean_vals) * prediction_scale,
            "low": np.array(low_vals) * prediction_scale,
            "high": np.array(high_vals) * prediction_scale,
        }

        slope_row = {name: 0.0 for name in fe_params.index}
        for name, weight in slope_vectors[group].items():
            slope_row[name] = weight
        vec = np.array([slope_row[name] for name in fe_params.index], dtype=float)
        estimate = float(vec @ fe_params.values)
        se = float(np.sqrt(vec @ cov_fe.values @ vec))
        slope_summaries[group] = {
            "estimate": estimate,
            "ci_low": estimate - 1.96 * se,
            "ci_high": estimate + 1.96 * se,
        }

    return fixed_lines, slope_summaries


def dynamic_ylim(
    scatter: pd.DataFrame,
    curves: dict[str, dict[str, np.ndarray]],
    y_col: str,
    *,
    pad_fraction: float = 0.12,
) -> tuple[float, float]:
    values = [scatter[y_col].to_numpy(dtype=float)]
    for group in GROUPS:
        values.extend(
            [
                curves[group]["mean"],
                curves[group]["low"],
                curves[group]["high"],
            ]
        )
    all_values = np.concatenate(values)
    ymin = float(np.nanmin(all_values))
    ymax = float(np.nanmax(all_values))
    pad = (ymax - ymin) * pad_fraction if ymax > ymin else 0.5
    return ymin - pad * 0.45, ymax + pad


def hddm_acc_curves(
    df: pd.DataFrame,
    feature: str,
    slope_draws: dict[str, np.ndarray],
    scale_factor: float,
    x_grid: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    curves: dict[str, dict[str, np.ndarray]] = {}
    for group in GROUPS:
        group_df = df[df["group"] == group]
        mean_acc = float(group_df["response"].mean())
        mean_feature = float(group_df[feature].mean())
        centered = x_grid - mean_feature

        draws = slope_draws[group][:, None]
        logits = logit(mean_acc) + scale_factor * draws * centered[None, :]
        probs = logistic(logits) * 100.0
        curves[group] = {
            "mean": probs.mean(axis=0),
            "low": np.percentile(probs, 2.5, axis=0),
            "high": np.percentile(probs, 97.5, axis=0),
        }
    return curves


def linear_best_fit_curves(
    scatter: pd.DataFrame,
    x_grid: np.ndarray,
    y_col: str = "acc_pct",
    clip: tuple[float, float] | None = (0.0, 100.0),
) -> dict[str, dict[str, np.ndarray]]:
    curves: dict[str, dict[str, np.ndarray]] = {}
    for group in GROUPS:
        g = scatter[scatter["group"] == group]
        x = g["x"].to_numpy(dtype=float)
        y = g[y_col].to_numpy(dtype=float)

        X = np.column_stack([np.ones_like(x), x])
        beta0, beta1 = np.linalg.lstsq(X, y, rcond=None)[0]
        mean = beta0 + beta1 * x_grid

        fitted = beta0 + beta1 * x
        resid = y - fitted
        n = len(x)
        dof = max(n - 2, 1)
        s2 = float(np.sum(resid**2) / dof)
        xbar = float(x.mean())
        sxx = float(np.sum((x - xbar) ** 2))
        if sxx > 0 and n > 2:
            se_mean = np.sqrt(s2 * (1.0 / n + ((x_grid - xbar) ** 2) / sxx))
            delta = 1.96 * se_mean
        else:
            delta = np.zeros_like(x_grid)

        curves[group] = {
            "mean": mean,
            "low": mean - delta,
            "high": mean + delta,
            "slope": float(beta1),
            "intercept": float(beta0),
        }
        if clip is not None:
            curves[group]["low"] = np.clip(curves[group]["low"], clip[0], clip[1])
            curves[group]["high"] = np.clip(curves[group]["high"], clip[0], clip[1])
    return curves


def noisy_trajectory(
    x0: float,
    x1: float,
    y1: float,
    n_points: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x0, x1, n_points)
    base = np.linspace(0.0, y1, n_points)
    noise = np.cumsum(rng.normal(0.0, noise_scale, size=n_points))
    noise = noise - np.linspace(0.0, noise[-1], n_points)
    taper = np.sin(np.linspace(0.0, np.pi, n_points))
    ys = base + noise * taper
    ys[0] = 0.0
    ys[-1] = y1
    return xs, ys


def representative_path(x0: float, x1: float, y1: float, n_points: int = 120) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x0, x1, n_points)
    ys = np.linspace(0.0, y1, n_points)
    return xs, ys


def density_bump(
    ax: plt.Axes,
    x_center: float,
    y_anchor: float,
    color: str,
    direction: str,
    width: float,
    amplitude: float,
) -> None:
    xs = np.linspace(x_center - width, x_center + width, 200)
    ys = np.exp(-0.5 * ((xs - x_center) / (width / 3.0)) ** 2)
    ys = ys / ys.max() * amplitude
    if direction == "up":
        ax.fill_between(xs, y_anchor, y_anchor + ys, color=color, alpha=0.20, linewidth=0)
        ax.plot(xs, y_anchor + ys, color=color, linewidth=1.6)
    else:
        ax.fill_between(xs, y_anchor, y_anchor - ys, color=color, alpha=0.20, linewidth=0)
        ax.plot(xs, y_anchor - ys, color=color, linewidth=1.6)


def draw_right_panel(
    ax: plt.Axes,
    slopes: dict[str, float],
    title: str,
    title_fontsize: float = 19,
) -> None:
    ax.set_facecolor("white")
    y_top, y_mid, y_bottom = 0.72, 0.0, -0.72
    x_left, x_right = 0.55, 7.9
    x_start = 1.0

    ax.hlines([y_top, y_mid, y_bottom], xmin=x_left, xmax=x_right, colors="#8A8A8A", linewidth=1.5)
    ax.scatter([x_start], [y_mid], s=34, color="#111111", zorder=5)

    mags = np.array([abs(slopes[g]) for g in GROUPS], dtype=float)
    mag_min = float(mags.min())
    mag_max = float(mags.max())

    def display_mag(group: str) -> float:
        mag = abs(slopes[group])
        if mag_max - mag_min < 1e-9:
            return 0.6
        return 0.35 + 0.55 * ((mag - mag_min) / (mag_max - mag_min))

    rng = np.random.default_rng(20260511 + int(sum(abs(v) * 10000 for v in slopes.values())))

    # Draw negative-slope groups first so the positive action path stays visually on top.
    draw_order = sorted(GROUPS, key=lambda g: slopes[g] > 0)
    top_paths: list[tuple[str, float]] = []
    bottom_paths: list[tuple[str, float]] = []

    for group in draw_order:
        color = COLORS[group]
        sign = 1 if slopes[group] >= 0 else -1
        disp = display_mag(group)
        x_end = 6.25 - 0.65 * disp
        y_target = y_top if sign > 0 else y_bottom

        for _ in range(12):
            jitter_end = x_end + rng.normal(0.0, 0.16)
            xs, ys = noisy_trajectory(x_start, jitter_end, y_target, 96, rng, noise_scale=0.010)
            ax.plot(xs, ys, color=color, alpha=0.22, linewidth=1.5)

        xs, ys = representative_path(x_start, x_end, y_target)
        ax.plot(xs, ys, color=color, linewidth=3.0, alpha=0.95)

        if sign > 0:
            top_paths.append((group, x_end))
        else:
            bottom_paths.append((group, x_end))

    for group, x_end in top_paths:
        density_bump(ax, x_end, y_top, COLORS[group], "up", width=0.44, amplitude=0.12)
    for group, x_end in bottom_paths:
        density_bump(ax, x_end, y_bottom, COLORS[group], "down", width=0.44, amplitude=0.10)

    # Drift-rate guide
    ax.annotate(
        "drift rate (v)",
        xy=(4.1, 0.47),
        xytext=(3.4, 0.53),
        fontsize=14,
        color="#3A3A3A",
        rotation=16,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.2, color="#555555"),
    )

    # Threshold arrow and labels
    ax.annotate(
        "",
        xy=(7.45, y_top - 0.02),
        xytext=(7.45, y_bottom + 0.02),
        arrowprops=dict(arrowstyle="<->", lw=1.4, linestyle=(0, (4, 3)), color="#444444"),
    )
    ax.text(7.58, 0.0, "threshold (a)", rotation=90, fontsize=15, color="#222222", ha="left", va="center")
    ax.text(6.98, y_top + 0.06, "upper threshold", fontsize=13, color="#222222", style="italic")
    ax.text(6.98, y_bottom - 0.09, "lower threshold", fontsize=13, color="#222222", style="italic")

    # Non-decision-time arrow and time arrow
    ax.annotate(
        "",
        xy=(0.62, -0.12),
        xytext=(1.03, -0.12),
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="#333333"),
    )
    ax.text(0.58, -0.27, "non-decision\n time (t)", fontsize=15, color="#222222", ha="left", va="top")
    ax.annotate(
        "",
        xy=(2.95, -1.02),
        xytext=(1.72, -1.02),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"),
    )
    ax.text(2.18, -0.93, "time", fontsize=15, color="#222222", style="italic")

    ax.set_title(title, fontsize=title_fontsize, pad=12)
    ax.set_xlim(0.2, 8.05)
    ax.set_ylim(-1.08, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_observed_effect_panel(
    ax: plt.Axes,
    scatter: pd.DataFrame,
    curves: dict[str, dict[str, np.ndarray]],
    x_grid: np.ndarray,
    title: str,
    ylabel: str,
    y_col: str,
    ylim: tuple[float, float],
    slope_label_loc: str,
    slope_values: dict[str, float] | None = None,
    marker_size: float = 10,
    scatter_alpha: float = 0.85,
    x_label: str = "Visual_Z",
) -> None:
    ax.set_facecolor("white")
    for group in GROUPS:
        color = COLORS[group]
        g_scatter = scatter[scatter["group"] == group]
        ax.scatter(
            g_scatter["x"],
            g_scatter[y_col],
            s=marker_size,
            color=color,
            alpha=scatter_alpha,
            edgecolor="none",
            zorder=2,
        )
        curve_x = curves[group].get("x", x_grid)
        ax.fill_between(
            curve_x,
            curves[group]["low"],
            curves[group]["high"],
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ax.plot(curve_x, curves[group]["mean"], color=color, linewidth=2.5, zorder=3)

    ax.set_title(title, fontsize=13.5, pad=11)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    ax.set_xlim(float(x_grid.min()) - 0.05, float(x_grid.max()) + 0.05)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=10.5, width=1.1, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    if slope_label_loc == "upper_left":
        x_text, y_text, ha, y_step = 0.10, 0.88, "left", -0.075
    elif slope_label_loc == "lower_right":
        x_text, y_text, ha, y_step = 0.96, 0.22, "right", -0.075
    else:
        raise ValueError(f"Unsupported slope label location: {slope_label_loc}")

    for i, group in enumerate(["Action", "Intuitive", "Semantic"]):
        slope = slope_values[group] if slope_values is not None else curves[group]["slope"]
        ax.text(
            x_text,
            y_text + i * y_step,
            f"{group}: b = {slope:.3f}",
            transform=ax.transAxes,
            fontsize=9.5,
            color=COLORS[group],
            ha=ha,
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.2),
        )


def make_behavior_feature_figure(
    *,
    feature_col: str,
    covariate_col: str,
    feature_label: str,
    outfile_stem: str,
) -> tuple[Path, Path]:
    raw_df = load_raw_behavior()

    feature_lo = float(raw_df[feature_col].min())
    feature_hi = float(raw_df[feature_col].max())
    x_grid = np.linspace(feature_lo, feature_hi, 220)

    rt_onset_scatter = build_empirical_feature_points(raw_df, feature_label, "RT_onset", filter_rt=True)
    rt_critical_scatter = build_empirical_feature_points(raw_df, feature_label, "RT_critical", filter_rt=True)
    acc_scatter = build_empirical_feature_points(raw_df, feature_label, "ACC", value_scale=100.0)

    rt_onset_curves, rt_onset_slopes = fit_lmm_feature_effect(
        raw_df,
        "RT_onset",
        feature_label,
        covariate_col,
        filter_rt=True,
    )
    rt_critical_curves, rt_critical_slopes = fit_lmm_feature_effect(
        raw_df,
        "RT_critical",
        feature_label,
        covariate_col,
        filter_rt=True,
    )
    acc_curves, acc_slopes = fit_lmm_feature_effect(
        raw_df,
        "ACC",
        feature_label,
        covariate_col,
        prediction_scale=100.0,
    )

    fig = plt.figure(figsize=(16.0, 5.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    draw_observed_effect_panel(
        axes[0],
        rt_onset_scatter,
        rt_onset_curves,
        x_grid,
        f"{feature_label} effect on RT_onset",
        "Predicted RT_onset (s)",
        "value",
        dynamic_ylim(rt_onset_scatter, rt_onset_curves, "value"),
        "upper_left",
        slope_values={group: rt_onset_slopes[group]["estimate"] for group in GROUPS},
        marker_size=20,
        scatter_alpha=0.92,
        x_label=feature_label,
    )
    draw_observed_effect_panel(
        axes[1],
        rt_critical_scatter,
        rt_critical_curves,
        x_grid,
        f"{feature_label} effect on RT_critical",
        "Predicted RT_critical (s)",
        "value",
        dynamic_ylim(rt_critical_scatter, rt_critical_curves, "value"),
        "upper_left",
        slope_values={group: rt_critical_slopes[group]["estimate"] for group in GROUPS},
        marker_size=20,
        scatter_alpha=0.92,
        x_label=feature_label,
    )
    draw_observed_effect_panel(
        axes[2],
        acc_scatter,
        acc_curves,
        x_grid,
        f"{feature_label} effect on ACC",
        "Predicted ACC (%)",
        "value",
        (72, 102.5),
        "lower_right",
        slope_values={group: acc_slopes[group]["estimate"] for group in GROUPS},
        marker_size=20,
        scatter_alpha=0.92,
        x_label=feature_label,
    )

    handles = [Line2D([0], [0], color=COLORS[group], lw=3, marker="o", markersize=7) for group in GROUPS]
    fig.legend(
        handles,
        GROUPS,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=3,
        frameon=False,
        fontsize=13,
        handlelength=1.7,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.17)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{outfile_stem}.png"
    svg_path = OUT_DIR / f"{outfile_stem}.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def make_visual_z_behavior_hddm_figure() -> tuple[Path, Path]:
    return make_behavior_feature_figure(
        feature_col="visual_z",
        covariate_col="Physical_Z",
        feature_label="Visual_Z",
        outfile_stem="paper_figure_visual_z_modulation",
    )


def make_physical_z_behavior_hddm_figure() -> tuple[Path, Path]:
    return make_behavior_feature_figure(
        feature_col="physical_z",
        covariate_col="Visual_Z",
        feature_label="Physical_Z",
        outfile_stem="paper_figure_physical_z_modulation",
    )


def make_posterior_coefficient_distribution_figure(
    *,
    rt_key: str,
    feature: str,
    title: str,
    outfile_stem: str,
) -> tuple[Path, Path]:
    from scipy.stats import gaussian_kde

    draws_by_group = {
        group: combined_trace(rt_key, TRACE_KEYS[feature][group])
        for group in GROUPS
    }
    all_draws = np.concatenate([draws_by_group[group] for group in GROUPS])
    x_min = float(all_draws.min())
    x_max = float(all_draws.max())
    pad = max((x_max - x_min) * 0.18, 0.02)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="white")
    max_density = 0.0

    for group in GROUPS:
        color = COLORS[group]
        draws = draws_by_group[group]
        ax.hist(
            draws,
            bins=46,
            density=True,
            color=color,
            alpha=0.15,
            edgecolor="white",
            linewidth=0.25,
        )

        kde = gaussian_kde(draws)
        density = kde(x_grid)
        max_density = max(max_density, float(density.max()))
        ax.fill_between(x_grid, 0, density, color=color, alpha=0.08, linewidth=0)
        ax.plot(x_grid, density, color=color, linewidth=2.8, label=group)
        ax.axvline(float(draws.mean()), color=color, linewidth=1.2, alpha=0.95)

    ax.axhline(0, color="#111111", linewidth=1.0)
    ax.set_title(title, fontsize=19, pad=14)
    ax.set_xlabel("Beta coefficient", fontsize=14)
    ax.set_ylabel("Posterior density", fontsize=14)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(0, max_density * 1.13)
    ax.tick_params(axis="both", labelsize=12, width=1.1, length=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.legend(frameon=False, loc="upper right", fontsize=13)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{outfile_stem}.png"
    svg_path = OUT_DIR / f"{outfile_stem}.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def make_feature_figure(
    df: pd.DataFrame,
    rt_key: str,
    feature: str,
    title_left: str,
    title_right: str,
    scale_factor: float,
    outfile_stem: str,
) -> tuple[Path, Path]:
    summary_path = RESULTS_DIR / rt_key / "regression" / "summary.json"
    summary = json.loads(summary_path.read_text())
    slope_block = "visual_slopes" if feature == "visual_z" else "physical_slopes"

    slope_means = {
        group: float(summary["hypotheses"][slope_block][group.lower()]["mean"])
        for group in ["Semantic", "Action", "Intuitive"]
    }
    slope_draws = {
        group: combined_trace(rt_key, TRACE_KEYS[feature][group]) for group in ["Semantic", "Action", "Intuitive"]
    }

    feature_lo = float(df[feature].quantile(0.01))
    feature_hi = float(df[feature].quantile(0.99))
    x_grid = np.linspace(feature_lo, feature_hi, 220)
    scatter = build_subject_bin_points(df, feature)
    curves = linear_best_fit_curves(scatter, x_grid)

    fig = plt.figure(figsize=(15.4, 7.1), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.16)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax_left.set_facecolor("white")
    for group in GROUPS:
        color = COLORS[group]
        g_scatter = scatter[scatter["group"] == group]
        ax_left.scatter(
            g_scatter["x"],
            g_scatter["acc_pct"],
            s=18,
            color=color,
            alpha=0.75,
            edgecolor="none",
            zorder=2,
        )
        ax_left.fill_between(
            x_grid,
            curves[group]["low"],
            curves[group]["high"],
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ax_left.plot(x_grid, curves[group]["mean"], color=color, linewidth=3.0, zorder=3)

    ax_left.set_title(title_left, fontsize=19, pad=14)
    ax_left.set_xlabel("Visual_Z" if feature == "visual_z" else "Physical_Z", fontsize=17)
    ax_left.set_ylabel("Predicted ACC (%)", fontsize=17, labelpad=14)
    ax_left.set_xlim(feature_lo - 0.05, feature_hi + 0.05)
    ax_left.set_ylim(72, 101)
    ax_left.tick_params(axis="both", labelsize=15, width=1.4, length=5)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.spines["left"].set_linewidth(1.8)
    ax_left.spines["bottom"].set_linewidth(1.8)

    # HDDM slope text block
    slope_lines = [
        ("Action", slope_means["Action"]),
        ("Intuitive", slope_means["Intuitive"]),
        ("Semantic", slope_means["Semantic"]),
    ]
    y0 = 74.5 if feature == "physical_z" else 74.2
    for i, (group, value) in enumerate(slope_lines):
        ax_left.text(
            feature_hi - 0.02,
            y0 + (2 - i) * 1.15,
            f"{group}: b = {value:.3f}",
            fontsize=14.5,
            color=COLORS[group],
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.25),
        )

    draw_right_panel(ax_right, slope_means, title_right)

    handles = [Line2D([0], [0], color=COLORS[group], lw=4, marker="o", markersize=10) for group in GROUPS]
    fig.legend(
        handles,
        GROUPS,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=3,
        frameon=False,
        fontsize=17,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{outfile_stem}.png"
    svg_path = OUT_DIR / f"{outfile_stem}.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    outputs = []
    outputs.extend(
        make_feature_figure(
            df=pd.read_csv(RESULTS_DIR / "rt_onset" / "rt_onset_cleaned.csv"),
            rt_key="rt_onset",
            feature="visual_z",
            title_left="Visual_Z effect on ACC (RT_onset)",
            title_right="HDDM Visual_Z modulation of drift rate v (RT_onset)",
            scale_factor=0.80,
            outfile_stem="paper_figure_visual_z_modulation_rt_onset",
        )
    )
    outputs.extend(
        make_feature_figure(
            df=pd.read_csv(RESULTS_DIR / "rt_critical" / "rt_critical_cleaned.csv"),
            rt_key="rt_critical",
            feature="visual_z",
            title_left="Visual_Z effect on ACC (RT_critical)",
            title_right="HDDM Visual_Z modulation of drift rate v (RT_critical)",
            scale_factor=0.80,
            outfile_stem="paper_figure_visual_z_modulation_rt_critical",
        )
    )
    outputs.extend(make_visual_z_behavior_hddm_figure())
    outputs.extend(make_physical_z_behavior_hddm_figure())
    outputs.extend(
        make_posterior_coefficient_distribution_figure(
            rt_key="rt_critical",
            feature="visual_z",
            title="Model results: posterior distributions of v_Visual_Z coefficients",
            outfile_stem="paper_figure_visual_z_posterior_coefficients",
        )
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
