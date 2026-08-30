#!/usr/bin/env python3

from __future__ import annotations

import pickle
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "hddm_results_4chains_2000samples"
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT / "rt_condition_sample_style"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["Semantic", "Action", "Intuitive"]
RT_KINDS = ["rt_onset", "rt_critical"]
DISPLAY_GROUP_NAMES = {
    "Semantic": "Concept Verification",
    "Intuitive": "Plausibility Assessment",
    "Action": "Affordance Recognition",
}
RT_DISPLAY_NAMES = {
    "rt_onset": r"$RT_{\mathrm{onset}}$",
    "rt_critical": r"$RT_{\mathrm{critical}}$",
}

COLORS = {
    "Semantic": {"line": "#4FA3C7", "fill": "#A9D9EE"},
    "Action": {"line": "#E63E32", "fill": "#F3AAA4"},
    "Intuitive": {"line": "#2C9C84", "fill": "#A7DBD0"},
}


def load_trace(path: Path) -> dict:
    return pickle.loads(path.read_bytes())


@lru_cache(maxsize=None)
def group_posterior_means(rt_kind: str) -> dict[str, float]:
    arrays = {group: [] for group in GROUPS}
    for chain in range(1, 5):
        trace_path = RESULTS_DIR / rt_kind / "group" / f"chain_{chain}_trace.db"
        trace = load_trace(trace_path)
        for group in GROUPS:
            arrays[group].append(np.asarray(trace[f"v({group})"][0], dtype=float))
    return {
        group: float(np.concatenate(group_arrays).mean())
        for group, group_arrays in arrays.items()
    }


def posterior_mean(rt_kind: str, key: str) -> float:
    group = key[len("v(") : -1]
    return group_posterior_means(rt_kind)[group]


def regression_baseline_v_mean(rt_kind: str, group: str) -> float:
    summary_path = RESULTS_DIR / rt_kind / "rt_summary.json"
    summary = json.loads(summary_path.read_text())
    drift_intercepts = summary["models"]["regression"]["hypotheses"]["drift_intercepts"]
    return float(drift_intercepts[group.lower()]["mean"])


def map_rt_to_x(rt_values: np.ndarray, x_start: float, x_end: float, rt_max: float) -> np.ndarray:
    clipped = np.clip(rt_values, 0, rt_max)
    return x_start + (clipped / rt_max) * (x_end - x_start)


def kde_band(ax: plt.Axes, x_values: np.ndarray, y_anchor: float, color: str, direction: str) -> None:
    if len(x_values) < 5:
        return
    xs = np.linspace(x_values.min(), x_values.max(), 400)
    ys = gaussian_kde(x_values)(xs)
    ys = ys / ys.max() * 0.13
    if direction == "up":
        ax.fill_between(xs, y_anchor, y_anchor + ys, color=color, alpha=0.35, linewidth=0)
        ax.plot(xs, y_anchor + ys, color=color, linewidth=2.0)
    else:
        ax.fill_between(xs, y_anchor, y_anchor - ys, color=color, alpha=0.35, linewidth=0)
        ax.plot(xs, y_anchor - ys, color=color, linewidth=2.0)


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


def plot_condition(rt_kind: str, group: str) -> Path:
    color_line = COLORS[group]["line"]
    color_fill = COLORS[group]["fill"]

    csv_path = RESULTS_DIR / rt_kind / f"{rt_kind}_cleaned.csv"
    df = pd.read_csv(csv_path)
    df = df[df["group"] == group].copy()

    correct_rts = df.loc[df["response"] == 1, "rt"].to_numpy()
    error_rts = df.loc[df["response"] == 0, "rt"].to_numpy()

    mean_correct_rt = float(correct_rts.mean())
    mean_error_rt = float(error_rts.mean())
    rt_max = float(np.quantile(df["rt"], 0.90))
    rt_max = max(rt_max, max(mean_correct_rt, mean_error_rt) * 1.35)
    x_start = 1.25
    x_end = 6.75

    correct_x = map_rt_to_x(correct_rts, x_start, x_end, rt_max)
    error_x = map_rt_to_x(error_rts, x_start, x_end, rt_max)
    mean_correct_x = float(correct_x.mean())
    mean_error_x = float(error_x.mean())

    # Use the regression drift intercept for both RT definitions so the label
    # matches the model family used for the reported HDDM results.
    v_mean = regression_baseline_v_mean(rt_kind, group)

    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_top = 1.0
    y_mid = 0.0
    y_bottom = -1.0
    x_left = 0.72
    x_right = 7.15

    ax.hlines(
        [y_top, y_mid, y_bottom],
        xmin=x_left,
        xmax=x_right,
        colors="#8A8A8A",
        linewidth=1.8,
        zorder=1,
    )

    seed = {"rt_onset": 1201, "rt_critical": 2201}[rt_kind] + GROUPS.index(group)
    rng = np.random.default_rng(seed)

    # Draw a limited set of pale trajectories sampled from observed RTs.
    correct_sample = np.quantile(correct_rts, np.linspace(0.10, 0.92, 14))
    error_sample = np.quantile(error_rts, np.linspace(0.10, 0.90, min(12, len(error_rts)))) if len(error_rts) >= 4 else error_rts

    for rt in correct_sample:
        x1 = map_rt_to_x(np.array([rt]), x_start, x_end, rt_max)[0]
        xs, ys = noisy_trajectory(x_start, x1, y_top, 90, rng, noise_scale=0.012)
        ax.plot(xs, ys, color=color_line, alpha=0.33, linewidth=2.0, zorder=2)

    for rt in error_sample:
        x1 = map_rt_to_x(np.array([rt]), x_start, x_end, rt_max)[0]
        xs, ys = noisy_trajectory(x_start, x1, y_bottom, 90, rng, noise_scale=0.012)
        ax.plot(xs, ys, color=color_line, alpha=0.22, linewidth=2.0, zorder=2)

    # Draw the RT densities before the highlighted paths so the path endpoints
    # remain visibly attached to the correct/error thresholds.
    kde_band(ax, correct_x, y_top, color_line, "up")
    kde_band(ax, error_x, y_bottom, color_line, "down")

    # Mean RT markers use the exact threshold endpoints.
    line_y_low = y_bottom
    line_y_high = y_top
    for mean_x in (mean_correct_x, mean_error_x):
        ax.plot(
            [mean_x, mean_x],
            [line_y_low, line_y_high],
            color="#4A4A4A",
            linestyle=(0, (5, 4)),
            linewidth=1.5,
            alpha=0.95,
            solid_capstyle="butt",
            dash_capstyle="butt",
            zorder=3,
            clip_on=True,
        )

    # Highlight representative straight paths.
    xs, ys = representative_path(x_start, mean_correct_x, y_top)
    ax.plot(
        xs,
        ys,
        color=color_line,
        linewidth=4.0,
        alpha=0.95,
        zorder=5,
        solid_capstyle="butt",
    )

    xs, ys = representative_path(x_start, mean_error_x, y_bottom)
    ax.plot(
        xs,
        ys,
        color=color_line,
        linewidth=4.0,
        alpha=0.55,
        zorder=5,
        solid_capstyle="butt",
    )
    seg_mask = (xs >= x_start + 0.28 * (mean_error_x - x_start)) & (xs <= x_start + 0.64 * (mean_error_x - x_start))
    ax.plot(
        xs[seg_mask],
        ys[seg_mask],
        color="#1E1E1E",
        linewidth=2.0,
        linestyle=(0, (7, 6)),
        alpha=0.85,
        zorder=6,
    )

    # Left labels
    ax.text(0.42, y_top - 0.012, "correct", ha="right", va="center", fontsize=18, color="#222222")
    ax.text(0.42, y_bottom + 0.01, "error", ha="right", va="center", fontsize=18, color="#222222")

    # Starting point and non-decision time.
    ax.scatter([x_start], [0], s=160, color="#111111", zorder=5)
    ax.annotate(
        "",
        xy=(x_start - 0.36, -0.20),
        xytext=(x_start - 0.02, -0.20),
        arrowprops=dict(arrowstyle="<->", linewidth=2.2, color="#333333"),
    )
    ax.text(x_start - 0.50, -0.44, "non-decision\n time (t)", ha="left", va="top", fontsize=17, color="#222222")

    # Time arrow
    ax.annotate(
        "",
        xy=(x_start + 0.98, -1.30),
        xytext=(x_start + 0.20, -1.30),
        arrowprops=dict(arrowstyle="->", linewidth=2.2, color="#333333"),
    )
    ax.text(x_start + 0.42, -1.20, "time", ha="left", va="center", fontsize=18, color="#222222", style="italic")

    # Labels for the dashed decision-time markers
    ax.text(mean_correct_x - 0.08, y_mid + 0.045, "correct", ha="right", va="bottom", fontsize=13, color="#222222")
    ax.text(mean_error_x + 0.08, y_mid + 0.045, "error", ha="left", va="bottom", fontsize=13, color="#222222")

    # Parameter text: keep only for rt_critical, and use regression baseline v.
    if rt_kind == "rt_critical":
        ax.text(
            5.35,
            0.40,
            f"HDDM\nv = {v_mean:.3f}",
            ha="left",
            va="center",
            fontsize=18,
            color="#4A4A4A",
        )

    ax.set_title(
        f"{DISPLAY_GROUP_NAMES[group]} ({RT_DISPLAY_NAMES[rt_kind]})",
        fontsize=28,
        pad=18,
        color="#111111",
    )

    ax.set_xlim(0.0, 7.4)
    ax.set_ylim(-1.32, 1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{rt_kind}_{group.lower()}_sample_style"
    png_path = OUT_DIR / f"{stem}.png"
    svg_path = OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    outputs = []
    for rt_kind in RT_KINDS:
        for group in GROUPS:
            outputs.append(plot_condition(rt_kind, group))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
