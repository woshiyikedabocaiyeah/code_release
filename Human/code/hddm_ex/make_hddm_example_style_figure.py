#!/usr/bin/env python3

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "hddm_results_4chains_2000samples"
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["Semantic", "Action", "Intuitive"]
FILL_COLORS = {
    "Semantic": "#9FD3F2",
    "Action": "#F7C59F",
    "Intuitive": "#B8E0D2",
}
LINE_COLORS = {
    "Semantic": "#4C8DB5",
    "Action": "#D97706",
    "Intuitive": "#2A9D8F",
}
TEXT_DARK = "#243447"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_trace(path: Path) -> dict:
    return pickle.loads(path.read_bytes())


def stacked_trace_arrays(trace_paths: list[Path], key: str) -> np.ndarray:
    arrays = []
    for path in trace_paths:
        obj = load_trace(path)
        arrays.append(np.asarray(obj[key][0], dtype=float))
    return np.concatenate(arrays)


def kde_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    low = np.quantile(values, 0.005)
    high = np.quantile(values, 0.995)
    pad = max((high - low) * 0.2, values.std() * 0.5, 1e-3)
    x = np.linspace(low - pad, high + pad, 500)
    kde = gaussian_kde(values)
    y = kde(x)
    return x, y


def draw_density(ax: plt.Axes, values: np.ndarray, group: str) -> tuple[float, float]:
    x, y = kde_curve(values)
    ax.fill_between(x, y, 0, color=FILL_COLORS[group], alpha=0.55, linewidth=0)
    ax.plot(x, y, color=LINE_COLORS[group], linewidth=2.2, label=group)
    return float(values.mean()), float(y.max())


def draw_mean_marker(ax: plt.Axes, x: float, ymax: float, group: str, frac: float) -> None:
    ax.vlines(
        x,
        0,
        ymax * frac,
        colors=LINE_COLORS[group],
        linestyles="--",
        linewidth=1.4,
        alpha=0.9,
    )


def style_axis(ax: plt.Axes, panel: str, title: str, xlabel: str, zero_line: bool = False) -> None:
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    if zero_line:
        ax.axvline(0, color="#6B7280", linestyle=":", linewidth=1.1, alpha=0.9)
    ax.text(
        -0.11,
        1.05,
        panel,
        transform=ax.transAxes,
        fontsize=14,
        weight="bold",
        color=TEXT_DARK,
        ha="left",
        va="bottom",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    rt_summary = load_json(RESULTS_DIR / "rt_critical" / "rt_summary.json")

    group_trace_paths = [
        RESULTS_DIR / "rt_critical" / "group" / f"chain_{i}_trace.db"
        for i in range(1, 5)
    ]
    reg_trace_paths = [
        RESULTS_DIR / "rt_critical" / "regression" / f"chain_{i}_trace.db"
        for i in range(1, 5)
    ]

    cleaned = pd.read_csv(rt_summary["cleaned_csv"])
    observed_rt = {
        group: cleaned.loc[cleaned["group"] == group, "rt"].to_numpy()
        for group in GROUP_ORDER
    }

    group_a = {
        group: stacked_trace_arrays(group_trace_paths, f"a({group})")
        for group in GROUP_ORDER
    }
    group_v = {
        group: stacked_trace_arrays(group_trace_paths, f"v({group})")
        for group in GROUP_ORDER
    }

    v_intercept = stacked_trace_arrays(reg_trace_paths, "v_Intercept")
    v_action_delta = stacked_trace_arrays(reg_trace_paths, "v_C(group, Treatment('Semantic'))[T.Action]")
    v_intuitive_delta = stacked_trace_arrays(reg_trace_paths, "v_C(group, Treatment('Semantic'))[T.Intuitive]")
    drift_intercepts = {
        "Semantic": v_intercept,
        "Action": v_intercept + v_action_delta,
        "Intuitive": v_intercept + v_intuitive_delta,
    }

    visual_slopes = {
        "Semantic": stacked_trace_arrays(
            reg_trace_paths, "v_C(group, Treatment('Semantic'))[Semantic]:visual_z"
        ),
        "Action": stacked_trace_arrays(
            reg_trace_paths, "v_C(group, Treatment('Semantic'))[Action]:visual_z"
        ),
        "Intuitive": stacked_trace_arrays(
            reg_trace_paths, "v_C(group, Treatment('Semantic'))[Intuitive]:visual_z"
        ),
    }
    physical_slopes = {
        "Semantic": stacked_trace_arrays(
            reg_trace_paths, "v_C(group, Treatment('Semantic'))[Semantic]:physical_z"
        ),
        "Action": stacked_trace_arrays(
            reg_trace_paths, "v_C(group, Treatment('Semantic'))[Action]:physical_z"
        ),
        "Intuitive": stacked_trace_arrays(
            reg_trace_paths, "v_C(group, Treatment('Semantic'))[Intuitive]:physical_z"
        ),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.87, wspace=0.18, hspace=0.34)

    fig.suptitle(
        "HDDM posterior summary for rt_critical",
        fontsize=18,
        weight="bold",
        color=TEXT_DARK,
        y=0.975,
    )
    fig.text(
        0.5,
        0.935,
        "Three-group density panels; dashed lines indicate posterior means",
        ha="center",
        fontsize=10.5,
        color="#4B5563",
    )

    panels = [
        ("(A)", "Observed RT distribution", "Reaction time (s)", observed_rt, False),
        ("(B)", "Decision boundary (a)", "Boundary parameter", group_a, False),
        ("(C)", "Group drift rate (v)", "Drift parameter", group_v, False),
        ("(D)", "Regression drift intercept", "Drift intercept", drift_intercepts, False),
        ("(E)", "Visual feature effect", "Visual slope", visual_slopes, True),
        ("(F)", "Physical feature effect", "Physical slope", physical_slopes, True),
    ]

    line_height_fracs = {"Semantic": 0.88, "Action": 0.94, "Intuitive": 0.82}

    for ax, (panel, title, xlabel, values_dict, zero_line) in zip(axes.flat, panels):
        means = {}
        peak_y = 0.0
        for group in GROUP_ORDER:
            mean_x, max_y = draw_density(ax, values_dict[group], group)
            means[group] = mean_x
            peak_y = max(peak_y, max_y)
        for group in GROUP_ORDER:
            draw_mean_marker(ax, means[group], peak_y, group, line_height_fracs[group])
        style_axis(ax, panel, title, xlabel, zero_line=zero_line)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.895),
        fontsize=10.5,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "figure_5_example_style_rt_critical"
    fig.savefig(f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
