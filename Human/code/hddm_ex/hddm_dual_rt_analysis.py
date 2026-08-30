#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hddm
from kabuki.analyze import gelman_rubin


CONDITION_MAP = {
    "categorization": "Semantic",
    "sensorimotor": "Action",
    "Voe": "Intuitive",
}

GROUP_ORDER = ["Semantic", "Action", "Intuitive"]

MODEL_ORDER = ["null", "group", "regression"]

REGRESSION_FORMULA = (
    "v ~ C(group, Treatment('Semantic'))"
    " + C(group, Treatment('Semantic')):visual_z"
    " + C(group, Treatment('Semantic')):physical_z"
)

RETRY_ATTEMPTS = [
    {"width_multiplier": 1.0, "maxiter": 5000},
    {"width_multiplier": 5.0, "maxiter": 50000},
    {"width_multiplier": 10.0, "maxiter": 200000},
    {"width_multiplier": 20.0, "maxiter": 500000},
    {"width_multiplier": 50.0, "maxiter": 1000000},
]


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HDDM analyses for both rt_onset and rt_critical."
    )
    parser.add_argument(
        "--csv",
        default="all_dat_merged.csv",
        help="Path to the merged CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="hddm_results",
        help="Directory for cleaned data, fitted models, figures, and summaries.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Posterior samples per chain for each model.",
    )
    parser.add_argument(
        "--burn",
        type=int,
        default=50,
        help="Burn-in samples per chain.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Thinning factor passed to HDDM sampling.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of MCMC chains per model.",
    )
    parser.add_argument(
        "--ppc-samples",
        type=int,
        default=20,
        help="Posterior predictive samples for the first fitted chain of each model.",
    )
    parser.add_argument(
        "--rt-kinds",
        nargs="+",
        default=["rt_onset", "rt_critical"],
        choices=["rt_onset", "rt_critical"],
        help="Which RT pipelines to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        choices=MODEL_ORDER,
        help="Which model families to run.",
    )
    parser.add_argument(
        "--subject-limit",
        type=int,
        default=None,
        help="Optional pilot mode: keep only the first N subjects after sorting.",
    )
    parser.add_argument(
        "--trials-per-subject",
        type=int,
        default=None,
        help="Optional pilot mode: keep only the first N trials per subject.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260423,
        help="Base random seed.",
    )
    return parser.parse_args()


def make_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="index")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [make_jsonable(v) for v in value.tolist()]
    if value is None:
        return None
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, ensure_ascii=False, indent=2)


def percentile_interval(trace: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(trace)),
        "std": float(np.std(trace, ddof=1)) if len(trace) > 1 else 0.0,
        "median": float(np.median(trace)),
        "hdi_2.5": float(np.percentile(trace, 2.5)),
        "hdi_97.5": float(np.percentile(trace, 97.5)),
    }


def trace_probability(trace_a: np.ndarray, trace_b: np.ndarray, op: str) -> float:
    n = min(len(trace_a), len(trace_b))
    if n == 0:
        return float("nan")
    if op == ">":
        return float(np.mean(trace_a[:n] > trace_b[:n]))
    if op == "<":
        return float(np.mean(trace_a[:n] < trace_b[:n]))
    raise ValueError(f"Unsupported op: {op}")


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("_") or "trace"


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["group"] = df["condition"].map(CONDITION_MAP)
    if df["group"].isna().any():
        missing = sorted(df.loc[df["group"].isna(), "condition"].unique())
        raise ValueError(f"Unmapped condition labels: {missing}")
    return df


def optional_pilot_slice(
    data: pd.DataFrame,
    subject_limit: int | None,
    trials_per_subject: int | None,
) -> pd.DataFrame:
    out = data.copy()
    if subject_limit is not None:
        keep_subjs = sorted(out["subj_idx"].astype(str).unique())[:subject_limit]
        out = out[out["subj_idx"].astype(str).isin(keep_subjs)].copy()
    if trials_per_subject is not None:
        out = (
            out.sort_values(["subj_idx"])
            .groupby("subj_idx", group_keys=False)
            .head(trials_per_subject)
            .copy()
        )
    return out.reset_index(drop=True)


def prepare_hddm_dataset(
    raw_df: pd.DataFrame,
    rt_kind: str,
    subject_limit: int | None = None,
    trials_per_subject: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    if rt_kind not in {"rt_onset", "rt_critical"}:
        raise ValueError(f"Unsupported rt_kind: {rt_kind}")

    rt_col = "RT_onset" if rt_kind == "rt_onset" else "RT_critical"
    working = raw_df.copy()
    working["subj_idx"] = working["Subject"].astype(str)
    working["response"] = working["ACC"].astype(int)
    working["rt"] = pd.to_numeric(working[rt_col], errors="coerce")
    working["visual_z"] = pd.to_numeric(working["Visual_Z"], errors="coerce")
    working["physical_z"] = pd.to_numeric(working["Physical_Z"], errors="coerce")
    working["group"] = pd.Categorical(
        working["group"], categories=GROUP_ORDER, ordered=True
    )

    removal = {
        "rt_kind": rt_kind,
        "rt_source_column": rt_col,
        "total_trials_before_filter": int(len(working)),
    }

    if rt_kind == "rt_critical":
        neg_mask = working["rt"] < 0
        removed = working.loc[neg_mask].copy()
        working = working.loc[~neg_mask].copy()
        removal["negative_trials_removed"] = int(neg_mask.sum())
        removal["negative_trials_removed_pct"] = float(neg_mask.mean() * 100.0)
        removal["negative_trials_removed_by_group"] = {}
        for group_name, group_df in raw_df.groupby("group"):
            group_neg = int((group_df["RT_critical"] < 0).sum())
            removal["negative_trials_removed_by_group"][str(group_name)] = {
                "removed_trials": group_neg,
                "pct_within_group": float(group_neg / len(group_df) * 100.0),
            }
        removal["removed_examples_preview"] = (
            removed[
                ["subj_idx", "condition", "group", "RT_critical", "ACC", "Video"]
            ]
            .head(10)
            .to_dict(orient="records")
        )
    else:
        removal["negative_trials_removed"] = 0
        removal["negative_trials_removed_pct"] = 0.0
        removal["negative_trials_removed_by_group"] = {
            group_name: {"removed_trials": 0, "pct_within_group": 0.0}
            for group_name in GROUP_ORDER
        }

    working = working.dropna(
        subset=["subj_idx", "rt", "response", "group", "visual_z", "physical_z"]
    ).copy()
    working = optional_pilot_slice(working, subject_limit, trials_per_subject)
    working = working[
        ["subj_idx", "rt", "response", "group", "visual_z", "physical_z", "condition"]
    ].reset_index(drop=True)

    dataset_summary = {
        "rows_after_filter": int(len(working)),
        "subjects_after_filter": int(working["subj_idx"].nunique()),
        "accuracy_overall": float(working["response"].mean()),
        "rt_summary": {
            "min": float(working["rt"].min()),
            "median": float(working["rt"].median()),
            "mean": float(working["rt"].mean()),
            "max": float(working["rt"].max()),
        },
        "group_counts": {
            group_name: int((working["group"] == group_name).sum())
            for group_name in GROUP_ORDER
        },
        "group_accuracy": {
            group_name: float(
                working.loc[working["group"] == group_name, "response"].mean()
            )
            for group_name in GROUP_ORDER
        },
        "group_rt_mean": {
            group_name: float(working.loc[working["group"] == group_name, "rt"].mean())
            for group_name in GROUP_ORDER
        },
    }

    return working, {"removal": removal, "dataset": dataset_summary}


def build_null_model(data: pd.DataFrame):
    return hddm.HDDM(data, include=["a", "v", "t"], p_outlier=0.05)


def build_group_model(data: pd.DataFrame):
    return hddm.HDDM(
        data,
        depends_on={"a": "group", "v": "group"},
        include=["a", "v", "t"],
        p_outlier=0.05,
    )


def build_regression_model(data: pd.DataFrame):
    return hddm.HDDMRegressor(
        data,
        REGRESSION_FORMULA,
        include=["a", "v", "t"],
        p_outlier=0.05,
        keep_regressor_trace=True,
    )


MODEL_BUILDERS = {
    "null": build_null_model,
    "group": build_group_model,
    "regression": build_regression_model,
}


def flatten_rhat_value(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.max())


def node_trace(model, node_name: str) -> np.ndarray:
    node = model.nodes_db.loc[node_name, "node"]
    return np.asarray(node.trace()[:], dtype=float).reshape(-1)


def combined_trace(models: Sequence, node_name: str) -> np.ndarray:
    traces = [node_trace(model, node_name) for model in models if node_name in model.nodes_db.index]
    if not traces:
        raise KeyError(f"Node {node_name} was not found in any fitted chain.")
    return np.concatenate(traces)


def selected_trace_nodes(model_name: str) -> List[str]:
    if model_name == "null":
        return ["a", "v", "t"]
    if model_name == "group":
        return [
            "a(Semantic)",
            "a(Action)",
            "a(Intuitive)",
            "v(Semantic)",
            "v(Action)",
            "v(Intuitive)",
            "t",
        ]
    if model_name == "regression":
        return [
            "a",
            "t",
            "v_Intercept",
            "v_C(group, Treatment('Semantic'))[T.Action]",
            "v_C(group, Treatment('Semantic'))[T.Intuitive]",
            "v_C(group, Treatment('Semantic'))[Semantic]:visual_z",
            "v_C(group, Treatment('Semantic'))[Action]:visual_z",
            "v_C(group, Treatment('Semantic'))[Intuitive]:visual_z",
            "v_C(group, Treatment('Semantic'))[Semantic]:physical_z",
            "v_C(group, Treatment('Semantic'))[Action]:physical_z",
            "v_C(group, Treatment('Semantic'))[Intuitive]:physical_z",
        ]
    raise ValueError(model_name)


def plot_trace_grid(models: Sequence, node_names: Iterable[str], out_path: Path) -> None:
    usable_nodes = [name for name in node_names if name in models[0].nodes_db.index]
    if not usable_nodes:
        return
    fig, axes = plt.subplots(
        len(usable_nodes),
        1,
        figsize=(12, 2.6 * len(usable_nodes)),
        squeeze=False,
    )
    for ax, node_name in zip(axes.ravel(), usable_nodes):
        for chain_idx, model in enumerate(models, start=1):
            trace = node_trace(model, node_name)
            ax.plot(trace, linewidth=0.8, alpha=0.8, label=f"chain_{chain_idx}")
        ax.set_title(node_name)
        ax.set_xlabel("sample")
        ax.set_ylabel("value")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def configure_slice_steps(
    model,
    width_multiplier: float = 1.0,
    maxiter: int | None = None,
) -> None:
    if getattr(model, "mc", None) is None:
        return
    for step_method in getattr(model.mc, "step_methods", []):
        if step_method.__class__.__name__ != "SliceStep":
            continue
        if maxiter is not None:
            step_method.maxiter = maxiter
        step_method.width = float(step_method.width) * width_multiplier


def sample_with_retries(
    builder,
    data: pd.DataFrame,
    model_name: str,
    chain_idx: int,
    chain_seed: int,
    samples: int,
    burn: int,
    thin: int,
    trace_db_path: Path,
    startval_error: str | None,
):
    last_error = None

    for attempt_idx, attempt in enumerate(RETRY_ATTEMPTS, start=1):
        if trace_db_path.exists():
            trace_db_path.unlink()

        attempt_seed = chain_seed + (attempt_idx - 1) * 1000
        seed_everything(attempt_seed)
        model = builder(data.copy())
        log(
            f"Prepared sampling state for {model_name} chain {chain_idx} "
            f"attempt {attempt_idx} with seed={attempt_seed}"
        )

        if attempt_idx > 1:
            try:
                retry_start = time.time()
                model.find_starting_values()
                log(
                    f"Retry find_starting_values finished for {model_name} "
                    f"chain {chain_idx} attempt {attempt_idx} in "
                    f"{time.time() - retry_start:.1f}s"
                )
            except Exception as exc:  # pragma: no cover
                startval_error = str(exc)
                log(
                    f"Retry find_starting_values failed for {model_name} "
                    f"chain {chain_idx} attempt {attempt_idx}: {startval_error}"
                )

        model.mcmc(db="pickle", dbname=str(trace_db_path))
        configure_slice_steps(
            model,
            width_multiplier=attempt["width_multiplier"],
            maxiter=attempt["maxiter"],
        )

        try:
            sample_start = time.time()
            log(
                f"Sampling start for {model_name} chain {chain_idx} "
                f"attempt {attempt_idx} "
                f"(width x{attempt['width_multiplier']}, maxiter={attempt['maxiter']})"
            )
            model.mc.sample(samples, burn=burn, thin=thin, progress_bar=False)
            model.sampled = True
            model.gen_stats()
            log(
                f"Sampling finished for {model_name} chain {chain_idx} "
                f"attempt {attempt_idx} in {time.time() - sample_start:.1f}s"
            )
            return model, startval_error
        except AssertionError as exc:
            last_error = exc
            log(
                f"Sampling retry needed for {model_name} chain {chain_idx} "
                f"attempt {attempt_idx}: {exc}"
            )
            try:
                model.mc.db.close()
            except Exception:
                pass
            del model
            gc.collect()

    raise RuntimeError(
        f"Sampling failed for {model_name} chain {chain_idx} after retries: {last_error}"
    )


def save_chain_stats(models: Sequence, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, model in enumerate(models, start=1):
        stats = model.gen_stats()
        stats.to_csv(out_dir / f"chain_{idx}_stats.csv")


def run_ppc(model, observed_data: pd.DataFrame, out_dir: Path, samples: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ppc = hddm.utils.post_pred_gen(model, samples=samples, progress_bar=False)
    ppc_rt = ppc["rt"]
    ppc_stats = hddm.utils.post_pred_stats(
        observed_data[["rt", "response"]], ppc[["rt", "response"]]
    )
    ppc_stats.to_csv(out_dir / "posterior_predictive_stats.csv")

    observed_rt = observed_data["rt"].to_numpy()
    try:
        sim_rt = ppc_rt.xs(0, level="sample").to_numpy()
    except Exception:
        sim_rt = ppc_rt.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        observed_rt,
        bins=50,
        density=True,
        histtype="step",
        linewidth=1.6,
        label="Observed",
    )
    ax.hist(
        sim_rt,
        bins=50,
        density=True,
        histtype="step",
        linewidth=1.6,
        label="Posterior predictive sample 0",
    )
    ax.set_xlabel("RT (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "posterior_predictive_rt_hist.png", dpi=160)
    plt.close(fig)

    return {
        "observed_rt_mean": float(np.mean(observed_rt)),
        "simulated_rt_mean_sample0": float(np.mean(sim_rt)),
        "stats_path": str(out_dir / "posterior_predictive_stats.csv"),
        "hist_path": str(out_dir / "posterior_predictive_rt_hist.png"),
    }


def summarize_group_hypotheses(models: Sequence) -> dict:
    a_sem = combined_trace(models, "a(Semantic)")
    a_act = combined_trace(models, "a(Action)")
    v_sem = combined_trace(models, "v(Semantic)")
    v_act = combined_trace(models, "v(Action)")
    v_int = combined_trace(models, "v(Intuitive)")

    return {
        "posterior_probability_a_action_gt_semantic": trace_probability(
            a_act, a_sem, ">"
        ),
        "posterior_probability_v_action_lt_semantic": trace_probability(
            v_act, v_sem, "<"
        ),
        "posterior_probability_v_action_lt_intuitive": trace_probability(
            v_act, v_int, "<"
        ),
        "a_semantic_summary": percentile_interval(a_sem),
        "a_action_summary": percentile_interval(a_act),
        "v_semantic_summary": percentile_interval(v_sem),
        "v_action_summary": percentile_interval(v_act),
    }


def summarize_regression_hypotheses(models: Sequence) -> dict:
    intercept = combined_trace(models, "v_Intercept")
    action_shift = combined_trace(
        models, "v_C(group, Treatment('Semantic'))[T.Action]"
    )
    intuitive_shift = combined_trace(
        models, "v_C(group, Treatment('Semantic'))[T.Intuitive]"
    )

    visual_sem = combined_trace(
        models, "v_C(group, Treatment('Semantic'))[Semantic]:visual_z"
    )
    visual_action = combined_trace(
        models, "v_C(group, Treatment('Semantic'))[Action]:visual_z"
    )
    visual_intuitive = combined_trace(
        models, "v_C(group, Treatment('Semantic'))[Intuitive]:visual_z"
    )

    physical_sem = combined_trace(
        models, "v_C(group, Treatment('Semantic'))[Semantic]:physical_z"
    )
    physical_action = combined_trace(
        models, "v_C(group, Treatment('Semantic'))[Action]:physical_z"
    )
    physical_intuitive = combined_trace(
        models, "v_C(group, Treatment('Semantic'))[Intuitive]:physical_z"
    )
    v_action = intercept + action_shift
    v_intuitive = intercept + intuitive_shift

    return {
        "drift_intercepts": {
            "semantic": percentile_interval(intercept),
            "action": percentile_interval(v_action),
            "intuitive": percentile_interval(v_intuitive),
        },
        "visual_slopes": {
            "semantic": percentile_interval(visual_sem),
            "action": percentile_interval(visual_action),
            "intuitive": percentile_interval(visual_intuitive),
        },
        "physical_slopes": {
            "semantic": percentile_interval(physical_sem),
            "action": percentile_interval(physical_action),
            "intuitive": percentile_interval(physical_intuitive),
        },
        "posterior_probability_action_physical_gt_0": float(
            np.mean(physical_action > 0)
        ),
        "posterior_probability_semantic_visual_gt_0": float(
            np.mean(visual_sem > 0)
        ),
        "posterior_probability_action_visual_lt_semantic_visual": trace_probability(
            visual_action, visual_sem, "<"
        ),
        "posterior_probability_action_physical_gt_semantic_physical": trace_probability(
            physical_action, physical_sem, ">"
        ),
    }


def fit_model_family(
    model_name: str,
    data: pd.DataFrame,
    rt_dir: Path,
    samples: int,
    burn: int,
    thin: int,
    chains: int,
    ppc_samples: int,
    seed: int,
) -> dict:
    model_dir = rt_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log(f"Start model family: {model_name} -> {model_dir}")

    chain_models = []
    chain_summaries = []
    builder = MODEL_BUILDERS[model_name]

    for chain_idx in range(chains):
        chain_seed = seed + chain_idx
        seed_everything(chain_seed)
        trace_db_path = model_dir / f"chain_{chain_idx + 1}_trace.db"
        model_path = model_dir / f"chain_{chain_idx + 1}.model"
        stats_path = model_dir / f"chain_{chain_idx + 1}_stats.csv"

        if model_path.exists():
            log(f"Loading existing model for {model_name} chain {chain_idx + 1}")
            model = hddm.load(str(model_path))
            chain_summary = {
                "chain": chain_idx + 1,
                "seed": chain_seed,
                "runtime_seconds": None,
                "model_path": str(model_path),
                "trace_db_path": str(trace_db_path),
                "dic": float(model.dic) if math.isfinite(model.dic) else float("nan"),
                "loaded_existing": True,
            }
            chain_models.append(model)
            chain_summaries.append(chain_summary)
            continue

        chain_start = time.time()
        model = builder(data.copy())
        log(
            f"Building chain {chain_idx + 1}/{chains} for {model_name} "
            f"(seed={chain_seed})"
        )

        startval_error = None
        try:
            startval_start = time.time()
            model.find_starting_values()
            log(
                f"find_starting_values finished for {model_name} "
                f"chain {chain_idx + 1} in {time.time() - startval_start:.1f}s"
            )
        except Exception as exc:  # pragma: no cover - robustness for HDDM runtime issues
            startval_error = str(exc)
            log(
                f"find_starting_values failed for {model_name} "
                f"chain {chain_idx + 1}: {startval_error}"
            )

        model, startval_error = sample_with_retries(
            builder=builder,
            data=data,
            model_name=model_name,
            chain_idx=chain_idx + 1,
            chain_seed=chain_seed,
            samples=samples,
            burn=burn,
            thin=thin,
            trace_db_path=trace_db_path,
            startval_error=startval_error,
        )
        model.save(str(model_path))
        log(f"Saved model for {model_name} chain {chain_idx + 1}: {model_path}")

        chain_summary = {
            "chain": chain_idx + 1,
            "seed": chain_seed,
            "runtime_seconds": round(time.time() - chain_start, 3),
            "model_path": str(model_path),
            "trace_db_path": str(trace_db_path),
            "dic": float(model.dic) if math.isfinite(model.dic) else float("nan"),
            "loaded_existing": False,
        }
        if startval_error is not None:
            chain_summary["find_starting_values_error"] = startval_error

        chain_models.append(model)
        chain_summaries.append(chain_summary)

    save_chain_stats(chain_models, model_dir)
    log(f"Saved chain stats for {model_name}")

    trace_path = model_dir / "trace_plots.png"
    plot_trace_grid(chain_models, selected_trace_nodes(model_name), trace_path)
    log(f"Saved trace plots for {model_name}: {trace_path}")

    rhat = {}
    if len(chain_models) > 1:
        rhat = gelman_rubin(chain_models)
        save_json(
            model_dir / "gelman_rubin.json",
            {k: flatten_rhat_value(v) for k, v in rhat.items()},
        )
        log(f"Saved Gelman-Rubin summary for {model_name}")

    ppc_summary = run_ppc(chain_models[0], data, model_dir, ppc_samples)
    log(f"Finished PPC for {model_name}")

    result = {
        "chains": chain_summaries,
        "trace_plot": str(trace_path),
        "ppc": ppc_summary,
        "rhat_max": float(
            np.nanmax([flatten_rhat_value(v) for v in rhat.values()])
        )
        if rhat
        else None,
    }

    if model_name == "group":
        result["hypotheses"] = summarize_group_hypotheses(chain_models)
    elif model_name == "regression":
        result["hypotheses"] = summarize_regression_hypotheses(chain_models)

    save_json(model_dir / "summary.json", result)
    log(f"Saved summary for {model_name}")

    for model in chain_models:
        if getattr(model, "mc", None) is not None:
            try:
                model.mc.db.close()
            except Exception:
                pass
        del model
    gc.collect()

    return result


def fit_for_rt_kind(
    raw_df: pd.DataFrame,
    rt_kind: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    rt_dir = output_dir / rt_kind
    rt_dir.mkdir(parents=True, exist_ok=True)
    log(f"Start RT pipeline: {rt_kind}")

    data, prep_summary = prepare_hddm_dataset(
        raw_df,
        rt_kind=rt_kind,
        subject_limit=args.subject_limit,
        trials_per_subject=args.trials_per_subject,
    )
    cleaned_csv = rt_dir / f"{rt_kind}_cleaned.csv"
    data.to_csv(cleaned_csv, index=False)
    log(f"Saved cleaned data for {rt_kind}: {cleaned_csv}")

    result = {
        "rt_kind": rt_kind,
        "condition_mapping": CONDITION_MAP,
        "group_order": GROUP_ORDER,
        "regression_formula": REGRESSION_FORMULA,
        "cleaned_csv": str(cleaned_csv),
        "preprocessing": prep_summary,
        "models": {},
    }

    for model_name in args.models:
        result["models"][model_name] = fit_model_family(
            model_name=model_name,
            data=data,
            rt_dir=rt_dir,
            samples=args.samples,
            burn=args.burn,
            thin=args.thin,
            chains=args.chains,
            ppc_samples=args.ppc_samples,
            seed=args.seed + (1000 * MODEL_ORDER.index(model_name)),
        )

    save_json(rt_dir / "rt_summary.json", result)
    log(f"Saved RT summary for {rt_kind}")
    return result


def write_markdown_summary(results: Dict[str, dict], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# HDDM Dual-RT Summary")
    lines.append("")
    lines.append(
        "Condition mapping assumed in this run: "
        "`categorization -> Semantic`, `sensorimotor -> Action`, `Voe -> Intuitive`."
    )
    lines.append("")

    for rt_kind, rt_result in results.items():
        lines.append(f"## {rt_kind}")
        prep = rt_result["preprocessing"]
        removal = prep["removal"]
        lines.append(
            f"- Rows after preprocessing: {prep['dataset']['rows_after_filter']}"
        )
        lines.append(
            f"- Subjects after preprocessing: {prep['dataset']['subjects_after_filter']}"
        )
        lines.append(
            f"- Negative `rt_critical` trials removed: "
            f"{removal['negative_trials_removed']} "
            f"({removal['negative_trials_removed_pct']:.3f}%)"
        )
        for model_name, model_result in rt_result["models"].items():
            dics = [chain["dic"] for chain in model_result["chains"]]
            lines.append(
                f"- {model_name}: mean DIC = {np.nanmean(dics):.3f}, "
                f"max R-hat = {model_result['rhat_max']}"
            )
        if "group" in rt_result["models"]:
            hypo = rt_result["models"]["group"]["hypotheses"]
            lines.append(
                "- Group model: "
                f"P(a_Action > a_Semantic) = "
                f"{hypo['posterior_probability_a_action_gt_semantic']:.3f}; "
                f"P(v_Action < v_Semantic) = "
                f"{hypo['posterior_probability_v_action_lt_semantic']:.3f}"
            )
        if "regression" in rt_result["models"]:
            hypo = rt_result["models"]["regression"]["hypotheses"]
            lines.append(
                "- Regression model: "
                f"P(action physical slope > 0) = "
                f"{hypo['posterior_probability_action_physical_gt_0']:.3f}; "
                f"P(action physical slope > semantic physical slope) = "
                f"{hypo['posterior_probability_action_physical_gt_semantic_physical']:.3f}"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Run started. Output dir: {output_dir}")

    raw_df = load_raw_data(csv_path)
    log(f"Loaded raw data from {csv_path} with {len(raw_df)} rows")

    all_results = {}
    for rt_kind in args.rt_kinds:
        all_results[rt_kind] = fit_for_rt_kind(
            raw_df=raw_df,
            rt_kind=rt_kind,
            args=args,
            output_dir=output_dir,
        )

    summary = {
        "runtime_seconds_total": round(time.time() - start, 3),
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "samples": args.samples,
        "burn": args.burn,
        "thin": args.thin,
        "chains": args.chains,
        "ppc_samples": args.ppc_samples,
        "subject_limit": args.subject_limit,
        "trials_per_subject": args.trials_per_subject,
        "results": all_results,
    }

    save_json(output_dir / "run_summary.json", summary)
    write_markdown_summary(all_results, output_dir / "summary.md")
    log(f"Run finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
