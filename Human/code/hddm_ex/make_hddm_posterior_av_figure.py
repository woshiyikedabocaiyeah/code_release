#!/usr/bin/env python3

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "hddm_results_4chains_2000samples"
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORDER = ["Semantic", "Intuitive", "Action"]
DISPLAY_NAMES = {
    "Semantic": "Concept Verification",
    "Intuitive": "Plausibility Assessment",
    "Action": "Affordance Recognition",
}
COLORS = {
    "Semantic": "#78B9DA",
    "Intuitive": "#74C0B0",
    "Action": "#F07067",
}


@lru_cache(maxsize=None)
def load_trace(path: Path) -> dict:
    """Load each chain once so multiple posterior parameters share the cache."""
    return pickle.loads(path.read_bytes())


def concat_group_trace(key: str) -> np.ndarray:
    arrays = []
    for chain in range(1, 5):
        trace_path = RESULTS_DIR / "rt_critical" / "group" / f"chain_{chain}_trace.db"
        trace = load_trace(trace_path)
        arrays.append(np.asarray(trace[key][0], dtype=float))
    return np.concatenate(arrays)


def concat_reg_trace(key: str) -> np.ndarray:
    arrays = []
    for chain in range(1, 5):
        trace_path = RESULTS_DIR / "rt_critical" / "regression" / f"chain_{chain}_trace.db"
        trace = load_trace(trace_path)
        arrays.append(np.asarray(trace[key][0], dtype=float))
    return np.concatenate(arrays)


def posterior_draws() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    a_draws = {
        "Action": concat_group_trace("a(Action)"),
        "Semantic": concat_group_trace("a(Semantic)"),
        "Intuitive": concat_group_trace("a(Intuitive)"),
    }

    intercept = concat_reg_trace("v_Intercept")
    coef_action = concat_reg_trace("v_C(group, Treatment('Semantic'))[T.Action]")
    coef_intuitive = concat_reg_trace("v_C(group, Treatment('Semantic'))[T.Intuitive]")
    v_draws = {
        "Action": intercept + coef_action,
        "Semantic": intercept.copy(),
        "Intuitive": intercept + coef_intuitive,
    }
    return a_draws, v_draws


def pair_prob(draw_a: np.ndarray, draw_b: np.ndarray, op: str) -> float:
    if op == "<":
        return float(np.mean(draw_a < draw_b))
    if op == ">":
        return float(np.mean(draw_a > draw_b))
    raise ValueError(op)


def posterior_prob_to_stars(prob: float) -> str:
    tail_p = max(1e-12, 2 * (1 - prob))
    if tail_p < 0.001:
        return "***"
    if tail_p < 0.01:
        return "**"
    if tail_p < 0.05:
        return "*"
    return "ns"


def add_violin_panel(
    ax: plt.Axes,
    draws_by_group: dict[str, np.ndarray],
    title: str,
    ylim: tuple[float, float],
    *,
    title_fontsize: int = 15,
    title_pad: int = 10,
    label_fontsize: int = 13,
    tick_fontsize: int = 12,
    violin_width: float = 0.82,
    bw_method=None,
) -> None:
    positions = np.arange(1, len(ORDER) + 1)
    data = [draws_by_group[g] for g in ORDER]

    vp = ax.violinplot(
        data,
        positions=positions,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=bw_method,
    )
    for body, group in zip(vp["bodies"], ORDER):
        body.set_facecolor(COLORS[group])
        body.set_edgecolor(COLORS[group])
        body.set_alpha(0.45)
        body.set_linewidth(1.0)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.08,
        patch_artist=True,
        showfliers=False,
        whis=(2.5, 97.5),
    )
    for box in bp["boxes"]:
        box.set(facecolor="#1C1C1C", edgecolor="#1C1C1C", linewidth=1.0)
    for whisker in bp["whiskers"]:
        whisker.set(color="#1C1C1C", linewidth=1.3)
    for cap in bp["caps"]:
        cap.set_visible(False)
    for median in bp["medians"]:
        median.set(color="#1C1C1C", linewidth=0)

    means = [np.mean(draws_by_group[g]) for g in ORDER]
    ax.scatter(positions, means, s=34, color="white", edgecolor="#1C1C1C", linewidth=0.8, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_NAMES[group] for group in ORDER], fontsize=tick_fontsize)
    ax.set_ylabel("posterior value", fontsize=label_fontsize)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="y", labelsize=tick_fontsize, colors="#333333")
    ax.tick_params(axis="x", colors="#333333")


def add_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    *,
    axes_fraction: bool = False,
    text_offset: float | None = None,
    fontsize: int = 12,
    linewidth: float = 1.0,
    bbox_pad: float = 0.12,
) -> None:
    transform = blended_transform_factory(ax.transData, ax.transAxes) if axes_fraction else ax.transData
    if text_offset is None:
        text_offset = 0.03 if axes_fraction else 0.018
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="#333333",
        linewidth=linewidth,
        transform=transform,
        clip_on=not axes_fraction,
    )
    ax.text(
        (x1 + x2) / 2,
        y + h + text_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#222222",
        bbox=dict(facecolor="white", edgecolor="none", pad=bbox_pad),
        transform=transform,
        clip_on=False,
    )


def main() -> None:
    a_draws, v_draws = posterior_draws()

    prob_action_sem = pair_prob(v_draws["Action"], v_draws["Semantic"], "<")
    prob_action_int = pair_prob(v_draws["Action"], v_draws["Intuitive"], "<")
    prob_sem_int = pair_prob(v_draws["Semantic"], v_draws["Intuitive"], "<")

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.0), facecolor="white")

    fig.suptitle(
        "HDDM posterior distributions across the three task conditions\n"
        "Boundary separation a overlaps across groups, whereas baseline drift rate v separates reliably",
        fontsize=16,
        y=0.875,
    )

    add_violin_panel(
        axes[0],
        a_draws,
        "Substantial overlap in boundary separation (a)",
        ylim=(2.85, 4.42),
        title_fontsize=14,
        title_pad=14,
    )
    add_violin_panel(
        axes[1],
        v_draws,
        "Significant separation in baseline drift rate (v)",
        ylim=(0.50, 1.40),
        title_fontsize=14,
        title_pad=14,
        violin_width=0.92,
        bw_method=0.45,
    )
    axes[1].set_yticks(np.arange(0.5, 1.41, 0.1))

    add_bracket(
        axes[1], 1, 2, 1.290, 0.008, posterior_prob_to_stars(prob_sem_int),
        text_offset=0.002, fontsize=11.5, linewidth=1.0, bbox_pad=0.08
    )
    add_bracket(
        axes[1], 2, 3, 1.340, 0.008, posterior_prob_to_stars(prob_action_int),
        text_offset=0.001, fontsize=11.5, linewidth=1.0, bbox_pad=0.08
    )
    add_bracket(
        axes[1], 1, 3, 1.385, 0.004, posterior_prob_to_stars(prob_action_sem),
        text_offset=0.002, fontsize=11.5, linewidth=1.0, bbox_pad=0.08
    )

    fig.tight_layout(rect=[0, 0, 1, 0.86], w_pad=1.8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "paper_figure_posterior_a_v_comparison.png"
    svg_path = OUT_DIR / "paper_figure_posterior_a_v_comparison.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(svg_path)

    fig2, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="white")
    fig2.suptitle("Significant separation in baseline drift rate (v)", fontsize=12.5, y=0.95)
    add_violin_panel(
        ax,
        v_draws,
        "",
        ylim=(0.50, 1.40),
        title_fontsize=1,
        title_pad=0,
        label_fontsize=11,
        tick_fontsize=10,
        violin_width=0.92,
        bw_method=0.45,
    )
    ax.set_yticks(np.arange(0.5, 1.41, 0.1))
    add_bracket(
        ax, 1, 2, 1.12, 0.018, posterior_prob_to_stars(prob_sem_int),
        axes_fraction=True, text_offset=0.002, fontsize=11.5, linewidth=1.0, bbox_pad=0.08
    )
    add_bracket(
        ax, 2, 3, 1.18, 0.018, posterior_prob_to_stars(prob_action_int),
        axes_fraction=True, text_offset=0.002, fontsize=11.5, linewidth=1.0, bbox_pad=0.08
    )
    add_bracket(
        ax, 1, 3, 1.24, 0.018, posterior_prob_to_stars(prob_action_sem),
        axes_fraction=True, text_offset=0.002, fontsize=11.5, linewidth=1.0, bbox_pad=0.08
    )
    fig2.subplots_adjust(left=0.13, right=0.99, bottom=0.14, top=0.73)
    png_path2 = OUT_DIR / "paper_figure_posterior_v_only.png"
    svg_path2 = OUT_DIR / "paper_figure_posterior_v_only.svg"
    fig2.savefig(png_path2, dpi=320, bbox_inches="tight")
    fig2.savefig(svg_path2, bbox_inches="tight")
    plt.close(fig2)
    print(png_path2)
    print(svg_path2)


if __name__ == "__main__":
    main()
