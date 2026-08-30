#!/usr/bin/env python
"""Create 6 clean sample-style condition figures: RT_onset/RT_critical x 3 tasks."""

import argparse
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
from matplotlib.transforms import blended_transform_factory


PLOT_ORDER = ["Semantic", "Voe", "Action"]
DISPLAY_LABELS = {
    "Semantic": "Concept Verification",
    "Voe": "Plausibility Assessment",
    "Action": "Affordance Recognition",
}
FILE_LABELS = {
    "Semantic": "semantic",
    "Voe": "intuitive",
    "Action": "action",
}
RT_LABELS = {
    "RT_onset": r"RT$_{onset}$",
    "RT_critical": r"RT$_{critical}$",
}
RAW_CONDITION = {
    "Semantic": "categorization",
    "Voe": "Voe",
    "Action": "sensorimotor",
}
COLORS = {
    "Semantic": "#4EA6C6",
    "Voe": "#1F987B",
    "Action": "#E3362A",
}


def lighten_color(color, mix=0.35):
    base = np.array(mcolors.to_rgb(color), dtype=float)
    return tuple(base * (1.0 - mix) + np.ones(3) * mix)


def load_models(output_dir, model_name, chains):
    model_dir = output_dir / "models" / model_name
    return [hddm.load(str(model_dir / f"{model_name}_chain{chain}.pkl")) for chain in range(1, chains + 1)]


def pooled_trace(models, node_name):
    return np.concatenate(
        [np.asarray(model.nodes_db.node[node_name].trace(), dtype=float) for model in models]
    )


def load_group_hddm_summaries(output_dir, chains):
    models = load_models(output_dir, "group", chains)
    summaries = {}
    for group in PLOT_ORDER:
        a_trace = pooled_trace(models, f"a({group})")
        v_trace = pooled_trace(models, f"v({group})")
        summaries[group] = {
            "a_mean": float(np.mean(a_trace)),
            "v_mean": float(np.mean(v_trace)),
            "a_hdi": tuple(np.quantile(a_trace, [0.025, 0.975])),
            "v_hdi": tuple(np.quantile(v_trace, [0.025, 0.975])),
        }
    return summaries


def load_regression_baseline_v_summaries(output_dir, chains):
    models = load_models(output_dir, "regression", chains)
    intercept = pooled_trace(models, "v_Intercept")
    semantic_offset = pooled_trace(models, "v_C(group)[T.Semantic]")
    voe_offset = pooled_trace(models, "v_C(group)[T.Voe]")
    traces = {
        "Action": intercept,
        "Semantic": intercept + semantic_offset,
        "Voe": intercept + voe_offset,
    }
    summaries = {}
    for group, trace in traces.items():
        summaries[group] = {
            "v_mean": float(np.mean(trace)),
            "v_hdi": tuple(np.quantile(trace, [0.025, 0.975])),
        }
    return summaries


def load_rt_trials(raw_csv, rt_column, min_rt=0.2):
    data = pd.read_csv(raw_csv)
    data = data[np.isfinite(data[rt_column])].copy()
    data = data[data[rt_column] > min_rt].copy()
    out = {}
    for group in PLOT_ORDER:
        subset = data[data["condition"] == RAW_CONDITION[group]].copy()
        out[group] = {
            "correct": np.asarray(subset.loc[subset["ACC"] == 1, rt_column], dtype=float),
            "error": np.asarray(subset.loc[subset["ACC"] == 0, rt_column], dtype=float),
        }
    return out


def rt_to_x(rt_values, lo, hi, x0):
    vals = np.clip(np.asarray(rt_values, dtype=float), lo, hi)
    scaled = (vals - lo) / (hi - lo)
    return x0 + 0.26 + 0.50 * scaled


def make_trajectory(rng, x0, x_end, y_end, t, noise_scale):
    x = x0 + (x_end - x0) * t
    base = y_end * np.power(t, 1.02)
    noise = np.cumsum(rng.normal(0.0, 1.0, size=t.size))
    bridge = np.linspace(noise[0], noise[-1], t.size)
    noise = noise - bridge
    max_abs = np.max(np.abs(noise))
    if max_abs > 0:
        noise = noise / max_abs
    y = base + noise * noise_scale * (1.0 - t) * (0.58 + 0.42 * t)
    y_lo = min(0.0, y_end)
    y_hi = max(0.0, y_end)
    y = np.clip(y, y_lo, y_hi)
    y[0] = 0.0
    y[-1] = y_end
    return x, y


def draw_rt_distribution(ax, x_values, baseline, color, direction="up", height=0.11):
    x_values = np.asarray(x_values, dtype=float)
    if x_values.size < 2:
        return

    bw = max(float(np.std(x_values)) * 0.28, 0.02)
    pad = max(0.05, bw * 2.0)
    x_min = max(0.12, float(np.min(x_values)) - pad)
    x_max = min(0.96, float(np.max(x_values)) + pad)
    x_grid = np.linspace(x_min, x_max, 220)

    diffs = (x_grid[:, None] - x_values[None, :]) / bw
    density = np.exp(-0.5 * diffs**2).sum(axis=1)
    if np.max(density) <= 0:
        return
    density = density / np.max(density)
    density[0] = 0.0
    density[-1] = 0.0
    offset = density * height
    y_grid = baseline + offset if direction == "up" else baseline - offset

    ax.fill_between(
        x_grid,
        baseline,
        y_grid,
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax.plot(x_grid, y_grid, color=color, linewidth=1.4, alpha=0.95, zorder=2)


def draw_condition_figure(group, rt_label, rt_data, hddm_summary, out_png, out_pdf):
    rng = np.random.default_rng(23)
    color = COLORS[group]
    faint = lighten_color(color, mix=0.42)

    correct = rt_data["correct"]
    error = rt_data["error"]
    all_rt = np.concatenate([correct, error])
    lo = float(np.quantile(all_rt, 0.05))
    hi = float(np.quantile(all_rt, 0.95))
    if hi <= lo:
        hi = lo + 1.0

    # Very light geometry tie-in to HDDM a: larger a means slightly wider separation.
    a_val = hddm_summary["a_mean"]
    a_norm = np.clip((a_val - 1.0) / 0.2, -1.0, 1.0)
    upper = 0.70 + 0.04 * a_norm
    lower = -upper
    center = 0.0
    x0 = 0.10
    t = np.linspace(0.0, 1.0, 150)

    x_correct = rt_to_x(correct, lo, hi, x0)
    x_error = rt_to_x(error, lo, hi, x0)
    median_correct = float(np.median(x_correct))
    median_error = float(np.median(x_error))
    display_correct = median_correct
    display_error = median_error
    min_display_gap = 0.14
    if abs(display_error - display_correct) < min_display_gap:
        midpoint = 0.5 * (display_correct + display_error)
        display_correct = midpoint - 0.5 * min_display_gap
        display_error = midpoint + 0.5 * min_display_gap

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(-0.06, 1.02)
    ax.set_ylim(-1.00, 1.00)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"{DISPLAY_LABELS[group]} ({RT_LABELS.get(rt_label, rt_label)})", fontsize=18.5, y=0.95, pad=0)
    ax.hlines([upper, center, lower], 0.02, 0.96, colors="#7B7B7B", linewidth=1.05, zorder=0)
    ax.scatter([x0], [0.0], s=22, color="#111111", zorder=6)

    label_transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.05,
        upper,
        "correct",
        transform=label_transform,
        ha="right",
        va="center",
        fontsize=14,
        color="#222222",
        clip_on=False,
    )
    ax.text(
        0.05,
        lower,
        "error",
        transform=label_transform,
        ha="right",
        va="center",
        fontsize=14,
        color="#222222",
        clip_on=False,
    )

    ax.annotate(
        "",
        xy=(x0, -0.14),
        xytext=(0.04, -0.14),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0),
    )
    ax.text(0.02, -0.30, "non-decision\n time (t)", ha="left", va="top", fontsize=13)

    ax.annotate(
        "",
        xy=(0.24, -0.95),
        xytext=(0.12, -0.95),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
    )
    ax.text(0.18, -0.91, "time", ha="center", va="bottom", fontsize=13, style="italic")

    for x_med in [display_correct, display_error]:
        ax.plot(
            [x_med, x_med],
            [lower, upper],
            color="#444444",
            linewidth=1.0,
            linestyle=(0, (5, 4)),
            dash_capstyle="butt",
            solid_capstyle="butt",
            zorder=1,
        )

    label_y = lower - 0.12
    ax.text(
        display_correct,
        label_y,
        "correct",
        ha="center",
        va="top",
        fontsize=12.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.06),
    )
    ax.text(
        display_error,
        label_y,
        "error",
        ha="center",
        va="top",
        fontsize=12.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.06),
    )

    def draw_bundle(x_ends, y_end, line_color, alpha, noise_scale, target_x):
        sampled = rng.choice(x_ends, size=min(18, len(x_ends)), replace=False)
        for x_end in sampled:
            x, y = make_trajectory(rng, x0, float(x_end), y_end, t, noise_scale=noise_scale)
            ax.plot(
                x,
                y,
                color=line_color,
                linewidth=1.3,
                alpha=alpha,
                zorder=2,
                solid_capstyle="butt",
                dash_capstyle="butt",
            )
        ax.plot(
            [x0, target_x],
            [0.0, y_end],
            color="#2F2F2F",
            linewidth=1.2,
            linestyle=(0, (5, 4)),
            zorder=3,
            solid_capstyle="butt",
            dash_capstyle="butt",
        )
        x_mean, y_mean = make_trajectory(rng, x0, target_x, y_end, t, noise_scale=noise_scale * 0.35)
        ax.plot(
            x_mean,
            y_mean,
            color=line_color,
            linewidth=2.8,
            alpha=0.95,
            zorder=4,
            solid_capstyle="butt",
            dash_capstyle="butt",
        )

    draw_bundle(x_correct, upper, color, alpha=0.34, noise_scale=0.07, target_x=display_correct)
    draw_bundle(x_error, lower, faint, alpha=0.36, noise_scale=0.07, target_x=display_error)
    draw_rt_distribution(ax, x_correct, upper, color, direction="up", height=0.10)
    draw_rt_distribution(ax, x_error, lower, faint, direction="down", height=0.10)

    if rt_label == "RT_onset":
        summary_text = f"$a$ = {hddm_summary['a_mean']:.3f}\n$v$ = {hddm_summary['v_mean']:.3f}"
        text_x, text_y, text_size = 0.76, 0.30, 13
    else:
        summary_text = f"HDDM\n$v$ = {hddm_summary['v_mean']:.3f}"
        text_x, text_y, text_size = 0.70, 0.28, 13

    ax.text(
        text_x,
        text_y,
        summary_text,
        ha="left",
        va="center",
        fontsize=text_size,
        color="#444444",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.18),
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="all_dat_merged.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--onset-hddm-dir", default=None)
    parser.add_argument("--critical-hddm-dir", default=None)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = root / args.output_dir
    raw_csv = root / args.input
    onset_hddm_dir = root / args.onset_hddm_dir if args.onset_hddm_dir else output_dir
    critical_hddm_dir = root / args.critical_hddm_dir if args.critical_hddm_dir else output_dir
    hddm_summaries = {
        "RT_onset": load_group_hddm_summaries(onset_hddm_dir, args.chains),
        "RT_critical": load_group_hddm_summaries(critical_hddm_dir, args.chains),
    }
    critical_v = load_regression_baseline_v_summaries(critical_hddm_dir, args.chains)
    for group in PLOT_ORDER:
        hddm_summaries["RT_critical"][group]["v_mean"] = critical_v[group]["v_mean"]
        hddm_summaries["RT_critical"][group]["v_hdi"] = critical_v[group]["v_hdi"]

    folder = _organized_fig_dir(output_dir) / "rt_condition_sample_style"
    folder.mkdir(parents=True, exist_ok=True)

    saved = []
    for rt_label in ["RT_onset", "RT_critical"]:
        rt_data = load_rt_trials(raw_csv, rt_label)
        for group in PLOT_ORDER:
            label = FILE_LABELS[group]
            png = folder / f"{rt_label.lower()}_{label}_sample.png"
            pdf = folder / f"{rt_label.lower()}_{label}_sample.pdf"
            draw_condition_figure(group, rt_label, rt_data[group], hddm_summaries[rt_label][group], png, pdf)
            saved.extend([png, pdf])

    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
