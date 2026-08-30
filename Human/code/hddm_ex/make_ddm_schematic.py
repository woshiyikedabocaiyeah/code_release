#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


BASE_DIR = Path(__file__).resolve().parent
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_ddm(
    n_trials: int = 2000,
    drift: float = 0.65,
    boundary: float = 1.2,
    start_bias: float = 0.0,
    noise: float = 1.0,
    ndt: float = 0.22,
    dt: float = 0.005,
    max_t: float = 3.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    max_steps = int(max_t / dt)

    upper_times = []
    lower_times = []
    upper_paths = []
    lower_paths = []

    upper = boundary / 2
    lower = -boundary / 2
    start = start_bias

    for _ in range(n_trials):
        x = start
        path = [x]
        hit = None
        t_hit = None
        for step in range(max_steps):
            x += drift * dt + noise * np.sqrt(dt) * rng.normal()
            path.append(x)
            if x >= upper:
                hit = "upper"
                t_hit = ndt + (step + 1) * dt
                break
            if x <= lower:
                hit = "lower"
                t_hit = ndt + (step + 1) * dt
                break
        if hit == "upper":
            upper_times.append(t_hit)
            if len(upper_paths) < 12:
                upper_paths.append(np.array(path))
        elif hit == "lower":
            lower_times.append(t_hit)
            if len(lower_paths) < 8:
                lower_paths.append(np.array(path))

    return {
        "upper_times": np.array(upper_times),
        "lower_times": np.array(lower_times),
        "upper_paths": upper_paths,
        "lower_paths": lower_paths,
        "upper": upper,
        "lower": lower,
        "start": start,
        "ndt": ndt,
        "dt": dt,
    }


def density_strip(ax: plt.Axes, values: np.ndarray, y0: float, color: str, direction: str = "up") -> None:
    if len(values) < 10:
        return
    xs = np.linspace(values.min() - 0.05, values.max() + 0.05, 300)
    ys = gaussian_kde(values)(xs)
    ys = ys / ys.max() * 0.18
    if direction == "up":
        ax.fill_between(xs, y0, y0 + ys, color=color, alpha=0.35, linewidth=0)
        ax.plot(xs, y0 + ys, color=color, linewidth=1.8, alpha=0.9)
    else:
        ax.fill_between(xs, y0, y0 - ys, color=color, alpha=0.35, linewidth=0)
        ax.plot(xs, y0 - ys, color=color, linewidth=1.8, alpha=0.9)


def main() -> None:
    sim = simulate_ddm()
    upper = sim["upper"]
    lower = sim["lower"]
    start = sim["start"]
    ndt = sim["ndt"]
    dt = sim["dt"]

    fig, ax = plt.subplots(figsize=(10.2, 7.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Boundaries and start line
    ax.hlines([upper, start, lower], xmin=0.0, xmax=2.8, colors="#8E8E8E", linewidth=1.6)

    # Trajectories
    upper_paths = sim["upper_paths"][:8]
    lower_paths = sim["lower_paths"][:5]
    upper_colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(upper_paths)))
    lower_colors = plt.cm.Reds(np.linspace(0.45, 0.9, len(lower_paths)))

    for path, color in zip(upper_paths, upper_colors):
        ts = ndt + np.arange(len(path)) * dt
        ax.plot(ts, path, color=color, linewidth=1.8, alpha=0.92)

    for path, color in zip(lower_paths, lower_colors):
        ts = ndt + np.arange(len(path)) * dt
        ax.plot(ts, path, color=color, linewidth=1.8, alpha=0.92)

    # RT densities along thresholds
    density_strip(ax, sim["upper_times"], upper, "#5AA4D6", direction="up")
    density_strip(ax, sim["lower_times"], lower, "#FF6B6B", direction="down")

    # Mean drift guideline
    ax.plot([ndt, 2.05], [start, 0.23], linestyle="--", color="#3A3A3A", linewidth=1.5)
    ax.text(
        1.35,
        0.20,
        "drift rate (v)",
        fontsize=12,
        color="#3A3A3A",
        rotation=17,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2),
    )

    # Non-decision time
    ax.annotate(
        "",
        xy=(ndt, -0.06),
        xytext=(0.0, -0.06),
        arrowprops=dict(arrowstyle="<->", color="#3A3A3A", linewidth=1.4),
    )
    ax.text(
        ndt / 2,
        -0.18,
        "non-\ndecision\ntime (t)",
        ha="center",
        va="center",
        fontsize=11,
        color="#3A3A3A",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.2),
    )

    # Bias
    ax.annotate(
        "",
        xy=(0.09, upper),
        xytext=(0.09, start),
        arrowprops=dict(arrowstyle="<->", color="#3A3A3A", linewidth=1.4, linestyle="--"),
    )
    ax.text(
        0.16,
        (upper + start) / 2,
        "bias (z)",
        rotation=90,
        va="center",
        fontsize=11,
        color="#3A3A3A",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.2),
    )

    # Threshold
    ax.annotate(
        "",
        xy=(2.65, upper),
        xytext=(2.65, lower),
        arrowprops=dict(arrowstyle="<->", color="#3A3A3A", linewidth=1.5, linestyle="--"),
    )
    ax.text(
        2.73,
        0,
        "threshold (a)",
        rotation=90,
        va="center",
        fontsize=11,
        color="#3A3A3A",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.2),
    )

    # Time arrow
    ax.annotate(
        "",
        xy=(2.00, -0.31),
        xytext=(1.55, -0.31),
        arrowprops=dict(arrowstyle="->", color="#3A3A3A", linewidth=1.5),
    )
    ax.text(1.71, -0.25, "time", fontsize=11, color="#3A3A3A")

    ax.text(2.14, upper + 0.035, "upper threshold", fontsize=11, style="italic", color="#3A3A3A")
    ax.text(2.14, lower - 0.085, "lower threshold", fontsize=11, style="italic", color="#3A3A3A")
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, fontsize=16, weight="bold", color="#1F2D3D")

    ax.set_xlim(-0.02, 2.95)
    ax.set_ylim(-0.92, 0.92)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "paper_figure_0_ddm_schematic"
    fig.savefig(f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
