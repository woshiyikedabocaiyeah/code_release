#!/usr/bin/env python
"""Plot Physical_Z effects on RT_onset, RT_critical, and ACC."""

import argparse
import csv
import subprocess
import tempfile
import warnings
from pathlib import Path

import matplotlib


# --- figure output redirected to _organized/figures/ -----------------------
# Models, tables, and prepared data still go to --output-dir;
# only figures are redirected.
_ORGANIZED_FIG_ROOT = Path(__file__).resolve().parents[3] / "figures" / "lmm"


def _organized_fig_dir(output_dir) -> Path:
    d = _ORGANIZED_FIG_ROOT / Path(output_dir).name
    d.mkdir(parents=True, exist_ok=True)
    return d

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import statsmodels.formula.api as smf


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
DV_LABELS = {
    "RT_onset": r"RT$_{onset}$",
    "RT_critical": r"RT$_{critical}$",
}
LOG_DV_LABELS = {
    "RT_onset": r"log(RT$_{onset}$)",
    "RT_critical": r"log(RT$_{critical}$)",
}
PANEL_LABELS = ["A", "B", "C"]


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(colors="#111111", width=1.0, labelsize=13)


def add_panel_label(ax, label, x=-0.09, y=1.10):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )


def load_empirical_rt(input_csv, dv):
    data = pd.read_csv(input_csv)
    data = data[(data[dv] > 0) & (data["ACC"] == 1)].copy()
    data["log_rt"] = np.log(data[dv])
    data["group"] = data["condition"].map(GROUP_MAP)

    out = {}
    for group in PLOT_ORDER:
        subset = (
            data[data["group"] == group]
            .groupby("Physical_Z", as_index=False)
            .agg(value=("log_rt", "mean"), n=("log_rt", "size"))
            .sort_values("Physical_Z")
        )
        out[group] = {
            "x": subset["Physical_Z"].to_numpy(dtype=float),
            "value": subset["value"].to_numpy(dtype=float),
            "n": subset["n"].to_numpy(dtype=int),
        }
    return out


def load_empirical_acc(input_csv):
    data = pd.read_csv(input_csv)
    data["group"] = data["condition"].map(GROUP_MAP)

    out = {}
    for group in PLOT_ORDER:
        subset = (
            data[data["group"] == group]
            .groupby("Physical_Z", as_index=False)
            .agg(success=("ACC", "sum"), n=("ACC", "size"))
            .sort_values("Physical_Z")
        )
        value = np.log((subset["success"].to_numpy(dtype=float) + 0.5) / (subset["n"].to_numpy(dtype=float) - subset["success"].to_numpy(dtype=float) + 0.5))
        out[group] = {
            "x": subset["Physical_Z"].to_numpy(dtype=float),
            "value": value,
            "n": subset["n"].to_numpy(dtype=int),
        }
    return out


def fit_lmer_physical_rt_with_r(input_csv, dv):
    r_code = r'''
suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[[1]]
dv <- args[[2]]
lines_out <- args[[3]]
summary_out <- args[[4]]

dat <- read.csv(input_csv, check.names = FALSE)
dat$Subject <- factor(dat$Subject)
dat$Video <- factor(dat$Video)
dat$condition <- factor(dat$condition, levels = c("categorization", "sensorimotor", "Voe"))
dat$ACC <- as.integer(dat$ACC)

keep <- dat$ACC == 1 &
  !is.na(dat[[dv]]) &
  dat[[dv]] > 0 &
  !is.na(dat$Visual_Z) &
  !is.na(dat$Physical_Z)
dat_rt <- dat[keep, ]
dat_rt$log_rt <- log(dat_rt[[dv]])

m_rt <- lmer(
  log_rt ~ condition * Visual_Z + condition * Physical_Z + (1 | Subject) + (1 | Video),
  data = dat_rt,
  REML = FALSE
)

beta <- fixef(m_rt)
vc <- as.matrix(vcov(m_rt))
x_grid <- seq(min(dat_rt$Physical_Z), max(dat_rt$Physical_Z), length.out = 240)
conditions <- c("categorization", "Voe", "sensorimotor")
groups <- c("Semantic", "Voe", "Action")

line_rows <- list()
for (i in seq_along(conditions)) {
  new_dat <- data.frame(
    condition = factor(conditions[[i]], levels = levels(dat_rt$condition)),
    Visual_Z = 0,
    Physical_Z = x_grid
  )
  design <- model.matrix(~ condition * Visual_Z + condition * Physical_Z, new_dat)
  design <- design[, names(beta), drop = FALSE]
  mean <- as.vector(design %*% beta)
  se <- sqrt(rowSums((design %*% vc) * design))
  line_rows[[i]] <- data.frame(
    group = groups[[i]],
    x = x_grid,
    mean = mean,
    lo = mean - 1.96 * se,
    hi = mean + 1.96 * se
  )
}
write.csv(do.call(rbind, line_rows), lines_out, row.names = FALSE)

slope_vec <- function(condition_name) {
  vec <- rep(0, length(beta))
  names(vec) <- names(beta)
  vec["Physical_Z"] <- 1
  if (condition_name == "sensorimotor") {
    vec["conditionsensorimotor:Physical_Z"] <- 1
  } else if (condition_name == "Voe") {
    vec["conditionVoe:Physical_Z"] <- 1
  }
  vec
}

summary_rows <- list()
for (i in seq_along(conditions)) {
  vec <- slope_vec(conditions[[i]])
  estimate <- as.numeric(sum(vec * beta))
  se <- as.numeric(sqrt(t(vec) %*% vc %*% vec))
  summary_rows[[i]] <- data.frame(
    group = groups[[i]],
    estimate = estimate,
    ci_low = estimate - 1.96 * se,
    ci_high = estimate + 1.96 * se
  )
}
write.csv(do.call(rbind, summary_rows), summary_out, row.names = FALSE)
'''
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        script_path = temp_path / "fit_physical_rt_lmer.R"
        lines_path = temp_path / "physical_rt_lines.csv"
        summary_path = temp_path / "physical_rt_summary.csv"
        script_path.write_text(r_code, encoding="utf-8")

        subprocess.run(
            ["Rscript", str(script_path), str(input_csv), dv, str(lines_path), str(summary_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        lines_df = pd.read_csv(lines_path)
        summary_df = pd.read_csv(summary_path)

    fixed_lines = {}
    for group in PLOT_ORDER:
        subset = lines_df[lines_df["group"] == group]
        fixed_lines[group] = {
            "x": subset["x"].to_numpy(dtype=float),
            "mean": subset["mean"].to_numpy(dtype=float),
            "lo": subset["lo"].to_numpy(dtype=float),
            "hi": subset["hi"].to_numpy(dtype=float),
        }

    summaries = {}
    for _, row in summary_df.iterrows():
        summaries[row["group"]] = {
            "estimate": float(row["estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        }

    return fixed_lines, summaries


def fit_lmm_physical_rt(input_csv, dv):
    try:
        return fit_lmer_physical_rt_with_r(input_csv, dv)
    except Exception as exc:
        warnings.warn(f"Falling back to statsmodels MixedLM for {dv}: {exc}")

    data = pd.read_csv(input_csv)
    data = data[(data[dv] > 0) & (data["ACC"] == 1)].copy()
    data["log_rt"] = np.log(data[dv])
    data["condition"] = pd.Categorical(
        data["condition"], categories=["categorization", "sensorimotor", "Voe"]
    )
    data["crossed_group"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "log_rt ~ C(condition) * Visual_Z + C(condition) * Physical_Z",
            data=data,
            groups=data["crossed_group"],
            re_formula="0",
            vc_formula={"Subject": "0 + C(Subject)", "Video": "0 + C(Video)"},
        )
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)

    fe_params = result.fe_params.copy()
    cov_fe = result.cov_params().loc[fe_params.index, fe_params.index]
    x_grid = np.linspace(float(data["Physical_Z"].min()), float(data["Physical_Z"].max()), 240)

    def design_row(group, physical_z):
        row = {name: 0.0 for name in fe_params.index}
        row["Intercept"] = 1.0
        row["Physical_Z"] = physical_z
        if group == "Action":
            row["C(condition)[T.sensorimotor]"] = 1.0
            row["C(condition)[T.sensorimotor]:Physical_Z"] = physical_z
        elif group == "Voe":
            row["C(condition)[T.Voe]"] = 1.0
            row["C(condition)[T.Voe]:Physical_Z"] = physical_z
        return row

    summaries = {}
    fixed_lines = {}
    slope_vectors = {
        "Semantic": {"Physical_Z": 1.0},
        "Action": {"Physical_Z": 1.0, "C(condition)[T.sensorimotor]:Physical_Z": 1.0},
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
            "mean": np.array(mean_vals),
            "lo": np.array(lo_vals),
            "hi": np.array(hi_vals),
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


def fit_glmer_physical_acc_with_r(input_csv):
    r_code = r'''
suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[[1]]
lines_out <- args[[2]]
summary_out <- args[[3]]

dat <- read.csv(input_csv, check.names = FALSE)
dat$Subject <- factor(dat$Subject)
dat$Video <- factor(dat$Video)
dat$condition <- factor(dat$condition, levels = c("categorization", "sensorimotor", "Voe"))
dat$ACC <- as.integer(dat$ACC)

dat_acc <- subset(dat, !is.na(ACC) & !is.na(Visual_Z) & !is.na(Physical_Z))

m_acc <- glmer(
  ACC ~ condition * Visual_Z + condition * Physical_Z + (1 | Subject) + (1 | Video),
  data = dat_acc,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

beta <- fixef(m_acc)
vc <- as.matrix(vcov(m_acc))
x_grid <- seq(min(dat_acc$Physical_Z), max(dat_acc$Physical_Z), length.out = 240)
conditions <- c("categorization", "Voe", "sensorimotor")
groups <- c("Semantic", "Voe", "Action")

line_rows <- list()
for (i in seq_along(conditions)) {
  new_dat <- data.frame(
    condition = factor(conditions[[i]], levels = levels(dat_acc$condition)),
    Visual_Z = 0,
    Physical_Z = x_grid
  )
  design <- model.matrix(~ condition * Visual_Z + condition * Physical_Z, new_dat)
  design <- design[, names(beta), drop = FALSE]
  mean <- as.vector(design %*% beta)
  se <- sqrt(rowSums((design %*% vc) * design))
  line_rows[[i]] <- data.frame(
    group = groups[[i]],
    x = x_grid,
    mean = mean,
    lo = mean - 1.96 * se,
    hi = mean + 1.96 * se
  )
}
write.csv(do.call(rbind, line_rows), lines_out, row.names = FALSE)

slope_vec <- function(condition_name) {
  vec <- rep(0, length(beta))
  names(vec) <- names(beta)
  vec["Physical_Z"] <- 1
  if (condition_name == "sensorimotor") {
    vec["conditionsensorimotor:Physical_Z"] <- 1
  } else if (condition_name == "Voe") {
    vec["conditionVoe:Physical_Z"] <- 1
  }
  vec
}

summary_rows <- list()
for (i in seq_along(conditions)) {
  vec <- slope_vec(conditions[[i]])
  estimate <- as.numeric(sum(vec * beta))
  se <- as.numeric(sqrt(t(vec) %*% vc %*% vec))
  summary_rows[[i]] <- data.frame(
    group = groups[[i]],
    estimate = estimate,
    ci_low = estimate - 1.96 * se,
    ci_high = estimate + 1.96 * se
  )
}
write.csv(do.call(rbind, summary_rows), summary_out, row.names = FALSE)
'''
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        script_path = temp_path / "fit_physical_acc_glmer.R"
        lines_path = temp_path / "physical_acc_lines.csv"
        summary_path = temp_path / "physical_acc_summary.csv"
        script_path.write_text(r_code, encoding="utf-8")

        subprocess.run(
            ["Rscript", str(script_path), str(input_csv), str(lines_path), str(summary_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        lines_df = pd.read_csv(lines_path)
        summary_df = pd.read_csv(summary_path)

    fixed_lines = {}
    for group in PLOT_ORDER:
        subset = lines_df[lines_df["group"] == group]
        fixed_lines[group] = {
            "x": subset["x"].to_numpy(dtype=float),
            "mean": subset["mean"].to_numpy(dtype=float),
            "lo": subset["lo"].to_numpy(dtype=float),
            "hi": subset["hi"].to_numpy(dtype=float),
        }

    summaries = {}
    for _, row in summary_df.iterrows():
        summaries[row["group"]] = {
            "estimate": float(row["estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        }

    return fixed_lines, summaries


def fit_logit_physical_acc(input_csv):
    try:
        return fit_glmer_physical_acc_with_r(input_csv)
    except Exception as exc:
        warnings.warn(f"Falling back to statsmodels BinomialBayesMixedGLM for ACC: {exc}")

    data = pd.read_csv(input_csv)
    data["condition"] = pd.Categorical(
        data["condition"], categories=["categorization", "sensorimotor", "Voe"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BinomialBayesMixedGLM.from_formula(
            "ACC ~ C(condition) * Visual_Z + C(condition) * Physical_Z",
            {"Subject": "0 + C(Subject)", "Video": "0 + C(Video)"},
            data=data,
        )
        result = model.fit_vb(verbose=False)

    fe_params = pd.Series(result.fe_mean, index=model.exog_names)
    fe_sd = pd.Series(result.fe_sd, index=model.exog_names)
    x_grid = np.linspace(float(data["Physical_Z"].min()), float(data["Physical_Z"].max()), 240)

    def design_row(group, physical_z):
        row = {name: 0.0 for name in fe_params.index}
        row["Intercept"] = 1.0
        row["Physical_Z"] = physical_z
        if group == "Action":
            row["C(condition)[T.sensorimotor]"] = 1.0
            row["C(condition)[T.sensorimotor]:Physical_Z"] = physical_z
        elif group == "Voe":
            row["C(condition)[T.Voe]"] = 1.0
            row["C(condition)[T.Voe]:Physical_Z"] = physical_z
        return row

    summaries = {}
    fixed_lines = {}
    slope_vectors = {
        "Semantic": {"Physical_Z": 1.0},
        "Action": {"Physical_Z": 1.0, "C(condition)[T.sensorimotor]:Physical_Z": 1.0},
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
            se = float(np.sqrt(np.sum((vec * fe_sd.values) ** 2)))
            mean_vals.append(mean)
            lo_vals.append(mean - 1.96 * se)
            hi_vals.append(mean + 1.96 * se)

        fixed_lines[group] = {
            "x": x_grid,
            "mean": np.array(mean_vals),
            "lo": np.array(lo_vals),
            "hi": np.array(hi_vals),
        }

        slope_row = {name: 0.0 for name in fe_params.index}
        for name, weight in slope_vectors[group].items():
            slope_row[name] = weight
        vec = np.array([slope_row[name] for name in fe_params.index], dtype=float)
        est = float(vec @ fe_params.values)
        se = float(np.sqrt(np.sum((vec * fe_sd.values) ** 2)))
        summaries[group] = {
            "estimate": est,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
        }

    return fixed_lines, summaries


def plot_rt_panel(ax, empirical, fixed_lines, slope_summaries, dv):
    note_x = 0.10
    note_y = {"Action": 0.92, "Voe": 0.84, "Semantic": 0.76}
    for group in PLOT_ORDER:
        row = fixed_lines[group]
        color = COLORS[group]
        ax.fill_between(row["x"], row["lo"], row["hi"], color=color, alpha=0.13, linewidth=0)
        ax.plot(row["x"], row["mean"], color=color, linewidth=2.8, zorder=2.3)
        ax.scatter(
            empirical[group]["x"],
            empirical[group]["value"],
            s=10,
            alpha=0.92,
            color=color,
            edgecolor="none",
            linewidth=0,
            zorder=3,
        )

    for group in ["Action", "Voe", "Semantic"]:
        slope = slope_summaries[group]
        ax.text(
            note_x,
            note_y[group],
            f"b = {slope['estimate']:.3f}",
            transform=ax.transAxes,
            fontsize=10.5,
            color=COLORS[group],
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=0.18),
        )

    all_means = np.concatenate([fixed_lines[group]["mean"] for group in PLOT_ORDER])
    all_points = np.concatenate([empirical[group]["value"] for group in PLOT_ORDER])
    ymin = min(float(all_means.min()), float(all_points.min()))
    ymax = max(float(all_means.max()), float(all_points.max()))
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.5

    ax.set_title(f"{PHYSICAL_LABEL} effect on {LOG_DV_LABELS[dv]}", fontsize=16, pad=12)
    ax.set_xlabel(PHYSICAL_LABEL, fontsize=14)
    ax.set_ylabel(LOG_DV_LABELS[dv], fontsize=14)
    ax.set_ylim(ymin - pad * 0.45, ymax + pad)
    ax.set_xlim(
        float(fixed_lines["Semantic"]["x"].min()) - 0.12,
        float(fixed_lines["Semantic"]["x"].max()) + 0.22,
    )
    style_axis(ax)


def plot_acc_panel(ax, empirical, fixed_lines, slope_summaries):
    label_offsets = {
        "Action": (0.30, 0.45),
        "Voe": (0.30, 0.0),
        "Semantic": (0.30, -0.45),
    }
    for group in PLOT_ORDER:
        row = fixed_lines[group]
        color = COLORS[group]
        ax.fill_between(row["x"], row["lo"], row["hi"], color=color, alpha=0.14, linewidth=0)
        ax.plot(row["x"], row["mean"], color=color, linewidth=2.6, zorder=3)
        ax.scatter(
            empirical[group]["x"],
            empirical[group]["value"],
            s=9,
            alpha=0.92,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            zorder=4,
        )

        slope = slope_summaries[group]
        label = f"b = {slope['estimate']:.3f}"
        x_anchor = float(row["x"][-20])
        y_anchor = float(row["mean"][-20])
        dx, dy = label_offsets[group]
        ax.annotate(
            label,
            xy=(x_anchor, y_anchor),
            xytext=(x_anchor + dx, y_anchor + dy),
            fontsize=10.5,
            color=color,
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.2),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=0, shrinkB=0),
        )

    ax.set_title(f"{PHYSICAL_LABEL} effect on logit(ACC)", fontsize=16, pad=12)
    ax.set_xlabel(PHYSICAL_LABEL, fontsize=14)
    ax.set_ylabel("logit(ACC)", fontsize=14)
    all_vals = np.concatenate(
        [fixed_lines[group]["mean"] for group in PLOT_ORDER]
        + [empirical[group]["value"] for group in PLOT_ORDER]
    )
    pad = (float(all_vals.max()) - float(all_vals.min())) * 0.14
    ax.set_ylim(float(all_vals.min()) - pad, float(all_vals.max()) + pad)
    ax.set_xlim(float(fixed_lines["Semantic"]["x"].min()) - 0.15, float(fixed_lines["Semantic"]["x"].max()) + 0.45)
    style_axis(ax)


def save_summary(output_dir, onset_summaries, critical_summaries, acc_summaries):
    path = output_dir / "tables" / "physical_z_rt_effects_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "group", "estimate", "ci_low", "ci_high"])
        for group in PLOT_ORDER:
            row = onset_summaries[group]
            writer.writerow(["log_rt_onset_slope", DISPLAY_LABELS[group], row["estimate"], row["ci_low"], row["ci_high"]])
        for group in PLOT_ORDER:
            row = critical_summaries[group]
            writer.writerow(["log_rt_critical_slope", DISPLAY_LABELS[group], row["estimate"], row["ci_low"], row["ci_high"]])
        for group in PLOT_ORDER:
            row = acc_summaries[group]
            writer.writerow(["logit_acc_slope", DISPLAY_LABELS[group], row["estimate"], row["ci_low"], row["ci_high"]])
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="all_dat_merged.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    root = Path.cwd()
    input_csv = root / args.input
    output_dir = root / args.output_dir

    empirical_onset = load_empirical_rt(input_csv, "RT_onset")
    onset_lines, onset_summaries = fit_lmm_physical_rt(input_csv, "RT_onset")

    empirical_critical = load_empirical_rt(input_csv, "RT_critical")
    critical_lines, critical_summaries = fit_lmm_physical_rt(input_csv, "RT_critical")

    empirical_acc = load_empirical_acc(input_csv)
    acc_lines, acc_summaries = fit_logit_physical_acc(input_csv)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), gridspec_kw={"width_ratios": [1.02, 1.02, 1.05]})
    plot_rt_panel(axes[0], empirical_onset, onset_lines, onset_summaries, "RT_onset")
    plot_rt_panel(axes[1], empirical_critical, critical_lines, critical_summaries, "RT_critical")
    plot_acc_panel(axes[2], empirical_acc, acc_lines, acc_summaries)
    label_positions = {
        "A": (-0.09, 1.10),
        "B": (-0.09, 1.10),
        "C": (-0.16, 1.14),
    }
    for ax, label in zip(axes, PANEL_LABELS):
        add_panel_label(ax, label, *label_positions[label])

    fig.suptitle(
        r"$\mathrm{Physical}_{z}$ effects across response times and observed accuracy",
        fontsize=17,
        y=0.98,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[group],
            lw=3.0,
            marker="o",
            markersize=5.6,
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
        fontsize=12,
        handlelength=1.6,
        handleheight=1.2,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    fig_dir = _organized_fig_dir(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "physical_z_rt_effects.png"
    pdf_path = fig_dir / "physical_z_rt_effects.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = save_summary(output_dir, onset_summaries, critical_summaries, acc_summaries)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
