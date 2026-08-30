#!/usr/bin/env python
"""Post-process saved HDDM chain outputs without re-sampling.

This script pools posterior traces across saved chains for hypothesis tests and
summarizes DIC across chains. It is intentionally separate from
run_hddm_models.py so a completed long MCMC run can be summarized again without
touching the expensive sampling step.
"""

import argparse
import json
import math
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
import plot_physical_z_posterior_distributions as physical_z_posteriors
import plot_visual_z_posterior_distributions as visual_z_posteriors
import run_hddm_models as runner


MODEL_NAMES = ["null", "group", "regression"]


def load_models(output_dir, model_name, chains):
    models = []
    for chain in range(1, chains + 1):
        path = output_dir / "models" / model_name / f"{model_name}_chain{chain}.pkl"
        models.append(hddm.load(str(path)))
    return models


def node_names(model):
    return [str(name) for name in model.nodes_db.index]


def get_trace(model, name):
    try:
        return np.asarray(model.nodes_db.node[name].trace(), dtype=float)
    except Exception:
        return None


def find_trace(model, substrings):
    lowered = [item.lower() for item in substrings]
    for name in node_names(model):
        text = name.lower()
        if all(part in text for part in lowered):
            trace = get_trace(model, name)
            if trace is not None:
                return name, trace
    return None, None


def pooled_trace(models, substrings):
    names = []
    traces = []
    for model in models:
        name, trace = find_trace(model, substrings)
        if trace is None:
            return None, None
        names.append(name)
        traces.append(trace)
    return names[0], np.concatenate(traces)


def summarize_dic(all_models, output_dir):
    rows = []
    for model_name, models in all_models.items():
        dics = []
        for model in models:
            try:
                dics.append(float(model.dic))
            except Exception:
                dics.append(math.nan)
        row = {
            "model": model_name,
            "dic": float(np.nanmean(dics)),
            "dic_mean": float(np.nanmean(dics)),
            "dic_sd": float(np.nanstd(dics, ddof=1)) if len(dics) > 1 else 0.0,
        }
        for idx, dic in enumerate(dics, start=1):
            row[f"dic_chain{idx}"] = dic
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("dic")
    frame.to_csv(output_dir / "tables" / "model_comparison_dic.csv", index=False)
    return frame


def boundary_test(group_models, output_dir):
    action_name, action = pooled_trace(group_models, ["a", "action"])
    semantic_name, semantic = pooled_trace(group_models, ["a", "semantic"])
    if action is None or semantic is None:
        return {
            "test": "P(a_Action > a_Semantic)",
            "available": False,
            "reason": "Could not find both Action and Semantic boundary traces.",
            "source": "pooled_chains",
        }

    n = min(len(action), len(semantic))
    diff = action[:n] - semantic[:n]
    result = {
        "test": "P(a_Action > a_Semantic)",
        "available": True,
        "source": "pooled_4_chains",
        "action_node": action_name,
        "semantic_node": semantic_name,
        "probability": float(np.mean(diff > 0)),
        "diff_mean": float(np.mean(diff)),
        "diff_hdi_2.5": float(np.quantile(diff, 0.025)),
        "diff_hdi_97.5": float(np.quantile(diff, 0.975)),
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(action, bins=60, alpha=0.55, density=True, label="Action")
    ax.hist(semantic, bins=60, alpha=0.55, density=True, label="Semantic")
    ax.set_title("Boundary posterior: Action vs Semantic")
    ax.set_xlabel("a")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_organized_fig_dir(output_dir) / "boundary_action_vs_semantic.png", dpi=180)
    plt.close(fig)
    return result


def slope_result(models, label, substrings):
    name, trace = pooled_trace(models, substrings)
    if trace is None:
        return {
            "slope": label,
            "available": False,
            "node": None,
            "source": "pooled_4_chains",
            "reason": f"No matching node found for substrings {substrings}",
        }
    return {
        "slope": label,
        "available": True,
        "source": "pooled_4_chains",
        "node": name,
        "mean": float(np.mean(trace)),
        "hdi_2.5": float(np.quantile(trace, 0.025)),
        "hdi_97.5": float(np.quantile(trace, 0.975)),
        "p_gt_0": float(np.mean(trace > 0)),
        "p_lt_0": float(np.mean(trace < 0)),
    }


def regression_tests(regression_models, output_dir):
    checks = [
        ("visual_z slope in Semantic group", ["v", "visual", "semantic"]),
        ("physical_z slope in Action group", ["v", "physical", "action"]),
    ]
    results = [slope_result(regression_models, label, pieces) for label, pieces in checks]

    fig, axes = plt.subplots(len(results), 1, figsize=(7, 3.2 * len(results)))
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        if not result.get("available"):
            ax.text(0.5, 0.5, result.get("reason", "not available"), ha="center", va="center")
            ax.axis("off")
            continue
        _, trace = pooled_trace(regression_models, [part for part in result["node"].split("_") if part])
        if trace is None:
            _, trace = pooled_trace(regression_models, [result["node"]])
        ax.hist(trace, bins=60, density=True, alpha=0.75)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(result["slope"])
        ax.set_xlabel(result["node"])
        ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(_organized_fig_dir(output_dir) / "regression_slopes.png", dpi=180)
    plt.close(fig)
    return results


def save_hypotheses(all_models, output_dir):
    results = {
        "boundary": boundary_test(all_models["group"], output_dir),
        "regression_slopes": regression_tests(all_models["regression"], output_dir),
    }
    path = output_dir / "tables" / "hypothesis_tests.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def save_visual_z_posterior_distributions(all_models, output_dir):
    slopes = visual_z_posteriors.summarize_visual_slopes(all_models["regression"])
    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    visual_z_posteriors.plot_posterior_distributions(ax, slopes)
    fig.tight_layout()
    fig.savefig(_organized_fig_dir(output_dir) / "visual_z_posterior_distributions.png", dpi=240, bbox_inches="tight")
    fig.savefig(_organized_fig_dir(output_dir) / "visual_z_posterior_distributions.pdf", bbox_inches="tight")
    plt.close(fig)
    return visual_z_posteriors.save_summary(output_dir, slopes)


def save_physical_z_posterior_distributions(all_models, output_dir):
    slopes = physical_z_posteriors.summarize_physical_slopes(all_models["regression"])
    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    physical_z_posteriors.plot_posterior_distributions(ax, slopes)
    fig.tight_layout()
    fig.savefig(_organized_fig_dir(output_dir) / "physical_z_posterior_distributions.png", dpi=240, bbox_inches="tight")
    fig.savefig(_organized_fig_dir(output_dir) / "physical_z_posterior_distributions.pdf", bbox_inches="tight")
    plt.close(fig)
    return physical_z_posteriors.save_summary(output_dir, slopes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data", default="outputs/prepared_hddm_data.csv")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--ppc-samples", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_models = {name: load_models(output_dir, name, args.chains) for name in MODEL_NAMES}
    data = runner.load_data(Path(args.data))

    for model_name, models in all_models.items():
        rhat = runner.save_rhat(models, model_name, output_dir)
        runner.plot_traces(models, model_name, output_dir)
        print(f"Max R-hat for {model_name}: {rhat['rhat'].max():.4f}")

        if args.ppc_samples > 0:
            print(f"Running PPC for {model_name}...")
            ppc = runner.run_ppc(models[0], model_name, output_dir, args.ppc_samples)
            runner.summarize_ppc(data, ppc, model_name, output_dir)

    dic = summarize_dic(all_models, output_dir)
    hypotheses = save_hypotheses(all_models, output_dir)
    visual_z_summary = save_visual_z_posterior_distributions(all_models, output_dir)
    physical_z_summary = save_physical_z_posterior_distributions(all_models, output_dir)
    print(dic.to_string(index=False))
    print(json.dumps(hypotheses, indent=2, ensure_ascii=False))
    print(f"Saved visual_z posterior summary: {visual_z_summary}")
    print(f"Saved physical_z posterior summary: {physical_z_summary}")


if __name__ == "__main__":
    main()
