#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "hddm_results_4chains_2000samples"
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
FIGURES_DIR = FIG_ROOT
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["Semantic", "Action", "Intuitive"]
GROUP_COLORS = {
    "Semantic": "#2A9D8F",
    "Action": "#E76F51",
    "Intuitive": "#264653",
}
RT_KIND_LABELS = {
    "rt_onset": "RT onset",
    "rt_critical": "RT critical",
}
RT_KIND_COLORS = {
    "rt_onset": "#457B9D",
    "rt_critical": "#E9C46A",
}
MODEL_ORDER = ["null", "group", "regression"]
MODEL_COLORS = {
    "null": "#8D99AE",
    "group": "#2A9D8F",
    "regression": "#E76F51",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(FIGURES_DIR / f"{stem}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def subject_level_behavior(cleaned_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cleaned_csv)
    subj = (
        df.groupby(["subj_idx", "group"], as_index=False)
        .agg(mean_rt=("rt", "mean"), accuracy=("response", "mean"))
    )
    summary = (
        subj.groupby("group", as_index=False)
        .agg(
            mean_rt=("mean_rt", "mean"),
            mean_rt_se=("mean_rt", "sem"),
            accuracy=("accuracy", "mean"),
            accuracy_se=("accuracy", "sem"),
            n_subjects=("subj_idx", "nunique"),
        )
    )
    summary["mean_rt_ci95"] = 1.96 * summary["mean_rt_se"].fillna(0.0)
    summary["accuracy_ci95"] = 1.96 * summary["accuracy_se"].fillna(0.0)
    summary["group"] = pd.Categorical(summary["group"], GROUP_ORDER, ordered=True)
    return summary.sort_values("group").reset_index(drop=True)


def interval_items_from_regression(model_summary: dict, block: str) -> list[dict]:
    values = model_summary["hypotheses"][block]
    items = []
    for group_key in ("semantic", "action", "intuitive"):
        label = group_key.capitalize()
        vals = values[group_key]
        items.append(
            {
                "label": label,
                "mean": vals["mean"],
                "low": vals["hdi_2.5"],
                "high": vals["hdi_97.5"],
                "color": GROUP_COLORS[label],
            }
        )
    return items


def interval_items_from_group(model_summary: dict, prefix: str) -> list[dict]:
    items = []
    for group_key in ("semantic", "action"):
        label = group_key.capitalize()
        vals = model_summary["hypotheses"][f"{prefix}_{group_key}_summary"]
        items.append(
            {
                "label": label,
                "mean": vals["mean"],
                "low": vals["hdi_2.5"],
                "high": vals["hdi_97.5"],
                "color": GROUP_COLORS[label],
            }
        )
    return items


def draw_interval_panel(
    ax: plt.Axes,
    items: list[dict],
    title: str,
    subtitle: str | None = None,
    zero_line: bool = False,
) -> None:
    y_positions = np.arange(len(items))[::-1]
    for y, item in zip(y_positions, items):
        ax.hlines(y, item["low"], item["high"], color=item["color"], linewidth=3, alpha=0.9)
        ax.plot(item["mean"], y, "o", color=item["color"], markersize=8)
        ax.text(
            item["high"],
            y + 0.08,
            f"{item['mean']:.3f} [{item['low']:.3f}, {item['high']:.3f}]",
            fontsize=9,
            color=item["color"],
            ha="left",
            va="bottom",
        )

    if zero_line:
        ax.axvline(0, color="#6C757D", linestyle="--", linewidth=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([item["label"] for item in items], fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color="#495057",
            ha="left",
            va="bottom",
        )
    ax.grid(axis="x", color="#DEE2E6", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_behavior_figure(rt_summaries: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    width = 0.34
    x = np.arange(len(GROUP_ORDER))

    for i, (metric, ci_metric, ylabel, title) in enumerate(
        [
            ("mean_rt", "mean_rt_ci95", "Mean RT (s)", "Behavioral mean RT"),
            ("accuracy", "accuracy_ci95", "Accuracy", "Behavioral accuracy"),
        ]
    ):
        ax = axes[i]
        for offset, rt_kind in zip((-width / 2, width / 2), ("rt_onset", "rt_critical")):
            summary = subject_level_behavior(Path(rt_summaries[rt_kind]["cleaned_csv"]))
            vals = summary[metric].to_numpy()
            cis = summary[ci_metric].to_numpy()
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=RT_KIND_COLORS[rt_kind],
                alpha=0.9,
                label=RT_KIND_LABELS[rt_kind],
                yerr=cis,
                capsize=4,
                edgecolor="white",
                linewidth=1,
            )
            for xpos, val in zip(x + offset, vals):
                ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.grid(axis="y", color="#DEE2E6", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[1].set_ylim(0.75, 1.0)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Figure 1. Behavioral summary across the two RT definitions", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, "figure_1_behavioral_summary")


def build_model_comparison_figure(run_summary: dict, rt_summaries: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    width = 0.34
    x = np.arange(len(MODEL_ORDER))

    for offset, rt_kind in zip((-width / 2, width / 2), ("rt_onset", "rt_critical")):
        models = rt_summaries[rt_kind]["models"]
        dic_vals = [np.nanmean([chain["dic"] for chain in models[name]["chains"]]) for name in MODEL_ORDER]
        rhat_vals = [models[name]["rhat_max"] for name in MODEL_ORDER]

        axes[0].bar(
            x + offset,
            dic_vals,
            width=width,
            color=RT_KIND_COLORS[rt_kind],
            alpha=0.9,
            label=RT_KIND_LABELS[rt_kind],
            edgecolor="white",
            linewidth=1,
        )
        axes[1].bar(
            x + offset,
            rhat_vals,
            width=width,
            color=RT_KIND_COLORS[rt_kind],
            alpha=0.9,
            label=RT_KIND_LABELS[rt_kind],
            edgecolor="white",
            linewidth=1,
        )

    axes[0].set_title("Model fit (mean DIC)", fontsize=13, weight="bold")
    axes[0].set_ylabel("Mean DIC")
    axes[1].set_title("Convergence (max R-hat)", fontsize=13, weight="bold")
    axes[1].set_ylabel("Max R-hat")
    axes[1].set_yscale("log")
    axes[1].axhline(1.1, color="#D62828", linestyle="--", linewidth=1.2)
    axes[1].text(0.02, 1.1, "target = 1.1", color="#D62828", fontsize=9, va="bottom")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([name.capitalize() for name in MODEL_ORDER])
        ax.grid(axis="y", color="#DEE2E6", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    runtime_hours = run_summary["runtime_seconds_total"] / 3600.0
    axes[0].text(
        0.01,
        0.98,
        f"Total rerun time: {runtime_hours:.1f} h",
        transform=axes[0].transAxes,
        fontsize=9,
        color="#495057",
        ha="left",
        va="top",
    )
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Figure 2. Model comparison and convergence diagnostics", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, "figure_2_model_comparison")


def build_regression_parameter_figure(model_summaries: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9), sharex="col")
    panel_specs = [
        ("drift_intercepts", "Drift intercepts", False),
        ("visual_slopes", "Visual slopes", True),
        ("physical_slopes", "Physical slopes", True),
    ]

    for row, rt_kind in enumerate(("rt_onset", "rt_critical")):
        model_summary = model_summaries[rt_kind]
        for col, (block, title, zero_line) in enumerate(panel_specs):
            subtitle = None
            if block == "physical_slopes":
                prob = model_summary["hypotheses"]["posterior_probability_action_physical_gt_0"]
                subtitle = f"P(Action physical slope > 0) = {prob:.3f}"
            elif block == "visual_slopes":
                prob = model_summary["hypotheses"]["posterior_probability_semantic_visual_gt_0"]
                subtitle = f"P(Semantic visual slope > 0) = {prob:.3f}"

            draw_interval_panel(
                axes[row, col],
                interval_items_from_regression(model_summary, block),
                f"{RT_KIND_LABELS[rt_kind]}: {title}",
                subtitle=subtitle,
                zero_line=zero_line,
            )

    fig.suptitle("Figure 3. Regression-model parameter estimates", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, "figure_3_regression_parameter_summary")


def build_group_parameter_figure(model_summaries: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panel_specs = [
        ("a", "Decision boundary a", "posterior_probability_a_action_gt_semantic"),
        ("v", "Drift rate v", "posterior_probability_v_action_lt_semantic"),
    ]

    for row, rt_kind in enumerate(("rt_onset", "rt_critical")):
        model_summary = model_summaries[rt_kind]
        for col, (prefix, title, prob_key) in enumerate(panel_specs):
            prob = model_summary["hypotheses"][prob_key]
            comparator = ">" if prefix == "a" else "<"
            subtitle = f"P({prefix}_Action {comparator} {prefix}_Semantic) = {prob:.3f}"
            draw_interval_panel(
                axes[row, col],
                interval_items_from_group(model_summary, prefix),
                f"{RT_KIND_LABELS[rt_kind]}: {title}",
                subtitle=subtitle,
                zero_line=False,
            )

    fig.suptitle("Figure 4. Group-model contrasts for Action vs Semantic", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, "figure_4_group_model_summary")


def main() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    run_summary = load_json(RESULTS_DIR / "run_summary.json")
    rt_summaries = {
        rt_kind: load_json(RESULTS_DIR / rt_kind / "rt_summary.json")
        for rt_kind in ("rt_onset", "rt_critical")
    }
    regression_summaries = {
        rt_kind: load_json(RESULTS_DIR / rt_kind / "regression" / "summary.json")
        for rt_kind in ("rt_onset", "rt_critical")
    }
    group_summaries = {
        rt_kind: load_json(RESULTS_DIR / rt_kind / "group" / "summary.json")
        for rt_kind in ("rt_onset", "rt_critical")
    }

    build_behavior_figure(rt_summaries)
    build_model_comparison_figure(run_summary, rt_summaries)
    build_regression_parameter_figure(regression_summaries)
    build_group_parameter_figure(group_summaries)

    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
