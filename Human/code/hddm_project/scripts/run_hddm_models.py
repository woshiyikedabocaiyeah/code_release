#!/usr/bin/env python
"""Run the HDDM models requested in the assignment.

Models
------
1. Null model:          a ~ 1, v ~ 1, t ~ 1
2. Group model:         a ~ C(group), v ~ C(group), t ~ 1
3. Regression model:    v ~ C(group) + C(group):visual_z + C(group):physical_z

The script runs multiple chains per model so Gelman-Rubin R-hat can be checked.
Use small sample counts only for environment tests; the assignment defaults are
2000 samples with 500 burn-in.
"""

import argparse
import json
import math
import pickle
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


MODEL_SPECS = {
    "null": {
        "kind": "hddm",
        "depends_on": None,
        "description": "Null model: a ~ 1, v ~ 1, t ~ 1",
    },
    "group": {
        "kind": "hddm",
        "depends_on": {"v": "group", "a": "group"},
        "description": "Group model: a ~ C(group), v ~ C(group), t ~ 1",
    },
    "regression": {
        "kind": "regressor",
        "formula": "v ~ C(group) + C(group):visual_z + C(group):physical_z",
        "description": "Regression model: v ~ C(group) + C(group):visual_z + C(group):physical_z",
    },
}


KEY_NODE_CANDIDATES = [
    "a",
    "v",
    "t",
    "a(Action)",
    "a(Semantic)",
    "a(Voe)",
    "v(Action)",
    "v(Semantic)",
    "v(Voe)",
]


def load_data(path):
    data = hddm.load_csv(str(path))
    data["subj_idx"] = data["subj_idx"].astype(str)
    data["group"] = pd.Categorical(
        data["group"], categories=["Action", "Semantic", "Voe"], ordered=False
    )
    return data


def build_model(data, spec):
    if spec["kind"] == "hddm":
        if spec["depends_on"] is None:
            return hddm.HDDM(data, include=["v", "a", "t"])
        return hddm.HDDM(data, depends_on=spec["depends_on"], include=["v", "a", "t"])
    if spec["kind"] == "regressor":
        return hddm.HDDMRegressor(data, spec["formula"], include=["v", "a", "t"])
    raise ValueError("Unknown model kind: {}".format(spec["kind"]))


def sample_model(model, samples, burn, dbname):
    Path(dbname).parent.mkdir(parents=True, exist_ok=True)
    model.find_starting_values()
    model.sample(samples, burn=burn, dbname=str(dbname), db="pickle")
    return model


def node_names(model):
    return [str(name) for name in model.nodes_db.index]


def get_node(model, name):
    try:
        return model.nodes_db.node[name]
    except Exception:
        return None


def get_trace(model, name):
    node = get_node(model, name)
    if node is None:
        return None
    try:
        return np.asarray(node.trace(), dtype=float)
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


def save_model_outputs(model, model_name, chain, output_dir):
    chain_dir = output_dir / "models" / model_name
    chain_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(chain_dir / f"{model_name}_chain{chain}.pkl"))

    stats = model.gen_stats()
    stats.to_csv(chain_dir / f"{model_name}_chain{chain}_stats.csv")

    with (chain_dir / f"{model_name}_chain{chain}_nodes.txt").open("w", encoding="utf-8") as f:
        for name in node_names(model):
            f.write(f"{name}\n")


def save_rhat(models, model_name, output_dir):
    rhat = hddm.analyze.gelman_rubin(models)
    rows = []
    for key, value in sorted(rhat.items()):
        try:
            numeric = float(value)
        except Exception:
            numeric = math.nan
        rows.append({"parameter": key, "rhat": numeric})
    frame = pd.DataFrame(rows)
    path = output_dir / "tables" / f"{model_name}_rhat.csv"
    frame.to_csv(path, index=False)
    return frame


def plot_traces(models, model_name, output_dir):
    fig_dir = _organized_fig_dir(output_dir) / model_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for candidate in KEY_NODE_CANDIDATES:
        if any(get_trace(model, candidate) is not None for model in models):
            names.append(candidate)

    if not names:
        names = node_names(models[0])[:12]

    for name in names:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        plotted = False
        for i, model in enumerate(models, start=1):
            trace = get_trace(model, name)
            if trace is None:
                continue
            ax.plot(trace, alpha=0.8, linewidth=0.8, label=f"chain {i}")
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{model_name}: trace for {name}")
        ax.set_xlabel("post-burn sample")
        ax.set_ylabel(name)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        safe_name = (
            name.replace("/", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("[", "_")
            .replace("]", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )
        fig.savefig(fig_dir / f"trace_{safe_name}.png", dpi=160)
        plt.close(fig)


def run_ppc(model, model_name, output_dir, samples):
    ppc_dir = output_dir / "ppc" / model_name
    ppc_dir.mkdir(parents=True, exist_ok=True)
    ppc = hddm.utils.post_pred_gen(model, samples=samples)
    with (ppc_dir / f"{model_name}_ppc.pkl").open("wb") as f:
        pickle.dump(ppc, f, protocol=pickle.HIGHEST_PROTOCOL)

    if hasattr(ppc, "to_csv"):
        ppc.to_csv(ppc_dir / f"{model_name}_ppc.csv")

    return ppc


def summarize_ppc(data, ppc, model_name, output_dir):
    rows = []
    observed = (
        data.groupby("group")
        .agg(
            obs_n=("rt", "size"),
            obs_acc=("response", "mean"),
            obs_rt_mean=("rt", "mean"),
            obs_rt_q10=("rt", lambda x: x.quantile(0.10)),
            obs_rt_q50=("rt", lambda x: x.quantile(0.50)),
            obs_rt_q90=("rt", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )

    sim = None
    if isinstance(ppc, pd.DataFrame):
        sim_frame = ppc.reset_index(drop=True)
        if "group" in sim_frame.columns and "rt" in sim_frame.columns:
            sim = (
                sim_frame.groupby("group")
                .agg(
                    sim_n=("rt", "size"),
                    sim_acc=("response", "mean"),
                    sim_rt_mean=("rt", "mean"),
                    sim_rt_q10=("rt", lambda x: x.quantile(0.10)),
                    sim_rt_q50=("rt", lambda x: x.quantile(0.50)),
                    sim_rt_q90=("rt", lambda x: x.quantile(0.90)),
                )
                .reset_index()
            )

    if sim is not None:
        merged = observed.merge(sim, on="group", how="left")
    else:
        merged = observed

    path = output_dir / "tables" / f"{model_name}_ppc_summary.csv"
    merged.to_csv(path, index=False)
    return merged


def hypothesis_group_boundary(model, output_dir):
    action_name, action = find_trace(model, ["a", "action"])
    semantic_name, semantic = find_trace(model, ["a", "semantic"])
    if action is None or semantic is None:
        return {
            "test": "P(a_Action > a_Semantic)",
            "available": False,
            "reason": "Could not find both Action and Semantic boundary traces.",
        }

    n = min(len(action), len(semantic))
    diff = action[:n] - semantic[:n]
    result = {
        "test": "P(a_Action > a_Semantic)",
        "available": True,
        "action_node": action_name,
        "semantic_node": semantic_name,
        "probability": float(np.mean(diff > 0)),
        "diff_mean": float(np.mean(diff)),
        "diff_hdi_2.5": float(np.quantile(diff, 0.025)),
        "diff_hdi_97.5": float(np.quantile(diff, 0.975)),
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(action, bins=40, alpha=0.55, density=True, label="Action")
    ax.hist(semantic, bins=40, alpha=0.55, density=True, label="Semantic")
    ax.set_title("Boundary posterior: Action vs Semantic")
    ax.set_xlabel("a")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_organized_fig_dir(output_dir) / "boundary_action_vs_semantic.png", dpi=180)
    plt.close(fig)

    return result


def regression_slope_result(model, label, substrings):
    name, trace = find_trace(model, substrings)
    if trace is None:
        return {
            "slope": label,
            "available": False,
            "node": None,
            "reason": "No matching node found for substrings {}".format(substrings),
        }
    return {
        "slope": label,
        "available": True,
        "node": name,
        "mean": float(np.mean(trace)),
        "hdi_2.5": float(np.quantile(trace, 0.025)),
        "hdi_97.5": float(np.quantile(trace, 0.975)),
        "p_gt_0": float(np.mean(trace > 0)),
        "p_lt_0": float(np.mean(trace < 0)),
    }


def hypothesis_regression(model, output_dir):
    checks = [
        ("visual_z slope in Semantic group", ["v", "visual", "semantic"]),
        ("physical_z slope in Action group", ["v", "physical", "action"]),
    ]
    results = [regression_slope_result(model, label, pieces) for label, pieces in checks]

    fig, axes = plt.subplots(len(results), 1, figsize=(7, 3.2 * len(results)))
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        if not result.get("available"):
            ax.text(0.5, 0.5, result.get("reason", "not available"), ha="center", va="center")
            ax.axis("off")
            continue
        _, trace = find_trace(model, [part for part in result["node"].split("_") if part])
        if trace is None:
            trace = get_trace(model, result["node"])
        ax.hist(trace, bins=40, density=True, alpha=0.75)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(result["slope"])
        ax.set_xlabel(result["node"])
        ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(_organized_fig_dir(output_dir) / "regression_slopes.png", dpi=180)
    plt.close(fig)

    return results


def save_hypotheses(group_model, regression_model, output_dir):
    results = {
        "boundary": hypothesis_group_boundary(group_model, output_dir),
        "regression_slopes": hypothesis_regression(regression_model, output_dir),
    }
    path = output_dir / "tables" / "hypothesis_tests.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def write_model_comparison(stats_by_model, output_dir):
    rows = []
    for model_name, model in stats_by_model.items():
        try:
            dic = float(model.dic)
        except Exception:
            dic = math.nan
        rows.append({"model": model_name, "dic": dic})
    frame = pd.DataFrame(rows).sort_values("dic")
    frame.to_csv(output_dir / "tables" / "model_comparison_dic.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/prepared_hddm_data.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--burn", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--chain-start", type=int, default=1)
    parser.add_argument("--ppc-samples", type=int, default=100)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["null", "group", "regression"],
        choices=sorted(MODEL_SPECS),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for folder in ["figures", "models", "ppc", "tables"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    data = load_data(Path(args.data))
    print("Loaded {} rows from {}".format(len(data), args.data))

    first_chain_by_model = {}
    all_chains_by_model = {}

    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        print("\n=== {} ===".format(spec["description"]))
        chains = []
        for idx, chain in enumerate(range(args.chain_start, args.chain_start + args.chains), start=1):
            if args.chain_start == 1 and args.chains > 1:
                label = "{}/{}".format(chain, args.chains)
            elif args.chains > 1:
                label = "{} (job chain {}/{})".format(chain, idx, args.chains)
            else:
                label = str(chain)
            print("Sampling {} chain {}...".format(model_name, label))
            model = build_model(data, spec)
            dbname = output_dir / "models" / model_name / f"{model_name}_chain{chain}_trace.db"
            sample_model(model, args.samples, args.burn, dbname)
            save_model_outputs(model, model_name, chain, output_dir)
            chains.append(model)

        all_chains_by_model[model_name] = chains
        first_chain_by_model[model_name] = chains[0]
        if not args.no_postprocess:
            rhat = save_rhat(chains, model_name, output_dir)
            plot_traces(chains, model_name, output_dir)
            print("Max R-hat for {}: {:.4f}".format(model_name, rhat["rhat"].max()))

            if args.ppc_samples > 0:
                print("Running PPC for {}...".format(model_name))
                ppc = run_ppc(chains[0], model_name, output_dir, args.ppc_samples)
                summarize_ppc(data, ppc, model_name, output_dir)

    if not args.no_postprocess:
        write_model_comparison(first_chain_by_model, output_dir)

        if "group" in first_chain_by_model and "regression" in first_chain_by_model:
            save_hypotheses(first_chain_by_model["group"], first_chain_by_model["regression"], output_dir)

        print("\nDone. Outputs are in {}".format(output_dir))
    else:
        print("\nSampling-only run complete. Outputs are in {}".format(output_dir))


if __name__ == "__main__":
    main()
