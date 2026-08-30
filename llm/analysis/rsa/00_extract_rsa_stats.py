#!/usr/bin/env python3
"""
Extract every RSA statistic that the manuscript section reports.

Single source of truth: everything the text and the figure quote is written to
derived/rsa_stats.json. No number is ever typed by hand into the prose or the
plotting script.

Three human measures are covered:
  rt           RT_onset RDM        (package)
  rt_critical  RT_critical RDM     (shipped in the deposit; rebuilt by
                                    optional/05_build_rt_critical.py)
  corr         correctness RDM     (package)

Run scripts/05 and scripts/01 first.

Usage:  python 00_extract_rsa_stats.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import paths as P  # noqa: E402
DERIVED = P.derived_dir(__file__)

from model_naming import (  # noqa: E402
    HUMAN_MEASURE_LABEL_TEXT,
    MODELS,
    STAGE_LABEL as LAYER_LABEL,
    STAGE_ORDER as LAYER_ORDER,
    TASK_LABEL,
    TASK_ORDER,
    label as model_label,
)

MEASURE_ORDER = ["rt", "rt_critical", "corr"]
TASK_FILE = {"Category": "cat", "VoE": "voe", "SM": "sensor"}


def r4(x):
    return None if pd.isna(x) else float(np.round(float(x), 6))


def main() -> None:
    pkg_pairs = pd.read_csv(P.RSA_DIR / "04_rsa_pairwise_results.csv")
    crit_pairs = pd.read_csv(P.rt_critical_dir(DERIVED) / "rsa_pairwise_rt_critical.csv")
    pair = pd.concat([pkg_pairs, crit_pairs], ignore_index=True)

    nc_pkg = pd.read_csv(P.NOISE_CEILING / "02_noise_ceiling_by_human.csv")
    nc_crit = pd.read_csv(P.rt_critical_dir(DERIVED) / "noise_ceiling_rt_critical.csv")
    perm = pd.read_csv(DERIVED / "permutation_tests.csv")
    crit_val = json.loads((P.rt_critical_dir(DERIVED) / "validation.json").read_text())

    stats: dict = {"provenance": {}, "design": {}, "global": {}}
    stats["provenance"] = {
        "package": "02_llm",
        "rsa_pairwise_rt_and_corr": "RSA/Data/04_rsa_pairwise_results.csv",
        "rsa_pairwise_rt_critical": "derived/rt_critical/rsa_pairwise_rt_critical.csv",
        "noise_ceiling_rt_and_corr": "Noise_Ceiling/Data/02_noise_ceiling_by_human.csv",
        "noise_ceiling_rt_critical": "derived/rt_critical/noise_ceiling_rt_critical.csv",
        "permutation": "derived/permutation_tests.csv",
        "rt_critical_source": crit_val["source"],
    }

    pair["model_label"] = [model_label(n, m) for n, m in
                           zip(pair.mllm_model_name, pair.mllm_model_mode)]

    # --------------------------------------------------------- design counts
    n_families = len({m.release_family for m in MODELS})
    n_variants = pair.groupby(["mllm_model_name", "mllm_model_mode"]).ngroups
    n_prompts = pair["mllm_condition_label"].nunique()
    n_layers = pair["mllm_embedding_type"].nunique()
    n_tasks = pair["question_type"].nunique()
    n_measures = pair["human_metric"].nunique()
    assert n_variants == len(MODELS)

    stats["design"] = {
        "n_model_variants": int(n_variants),
        "n_model_families": int(n_families),
        "n_prompt_conditions": int(n_prompts),
        "n_layers": int(n_layers),
        "n_tasks": int(n_tasks),
        "n_human_measures": int(n_measures),
        "n_model_configurations": int(n_variants * n_prompts * n_layers),
        "n_model_rdms": int(n_variants * n_prompts * n_layers * n_tasks),
        "n_human_rdms": int(pair["human_matrix_id"].nunique()),
        "n_pairings": int(len(pair)),
        "n_pairings_per_cell": int(n_variants * n_prompts),
        "n_pairings_per_human_matrix": int(n_variants * n_prompts * n_layers),
        "n_videos": 48,
        "n_video_pairs": int(pair["vector_length"].unique()[0]),
    }
    assert stats["design"]["n_pairings"] == 1188
    assert stats["design"]["n_human_rdms"] == 9

    # ----------------------------------------------------------- global stats
    for m in MEASURE_ORDER:
        s = pair[pair.human_metric == m].spearman_rho
        stats["global"][m] = {
            "label": HUMAN_MEASURE_LABEL_TEXT[m],
            "n": int(len(s)),
            "mean_rho": r4(s.mean()), "median_rho": r4(s.median()),
            "min_rho": r4(s.min()), "max_rho": r4(s.max()),
            "n_positive": int((s > 0).sum()),
            "pct_positive": r4(100 * (s > 0).mean()),
        }

    # ------------------------------------------------- measure x task x layer
    rows = []
    for m in MEASURE_ORDER:
        for t in TASK_ORDER:
            for l in LAYER_ORDER:
                s = pair[(pair.human_metric == m) & (pair.question_type == t)
                         & (pair.mllm_embedding_type == l)]
                rows.append({
                    "human_metric": m, "measure_label": HUMAN_MEASURE_LABEL_TEXT[m],
                    "question_type": t, "task_label": TASK_LABEL[t],
                    "embedding_type": l, "layer_label": LAYER_LABEL[l],
                    "n": len(s), "n_distinct": int(s.mllm_group_slug.nunique()),
                    "mean_rho": s.spearman_rho.mean(),
                    "sd_rho": s.spearman_rho.std(ddof=1),
                    "median_rho": s.spearman_rho.median(),
                    "min_rho": s.spearman_rho.min(), "max_rho": s.spearman_rho.max(),
                    "pct_positive": 100 * (s.spearman_rho > 0).mean(),
                })
    layer_tbl = pd.DataFrame(rows)

    # distinct RDM count per cell (prompt and task do not reach the visual stages)
    rdm_hash = {f.name[: -len("__mean_pool.npy")]:
                hashlib.md5(np.ascontiguousarray(np.load(f))).hexdigest()
                for f in P.RDM_MEAN_POOL.glob("*.npy")}
    pair["rdm_hash"] = pair.mllm_group_slug.str.replace(
        "__mean_pool", "", regex=False).map(rdm_hash)
    assert pair.rdm_hash.notna().all()
    for i, r in layer_tbl.iterrows():
        s = pair[(pair.human_metric == r.human_metric)
                 & (pair.question_type == r.question_type)
                 & (pair.mllm_embedding_type == r.embedding_type)]
        layer_tbl.at[i, "n_distinct"] = int(s.rdm_hash.nunique())
    layer_tbl.to_csv(DERIVED / "table_layer_means.csv", index=False)

    stats["by_task_layer"] = {}
    for _, r in layer_tbl.iterrows():
        stats["by_task_layer"][
            f"{r.human_metric}|{r.question_type}|{r.embedding_type}"] = {
            "mean_rho": r4(r.mean_rho), "sd_rho": r4(r.sd_rho),
            "n": int(r.n), "n_distinct": int(r.n_distinct),
            "pct_positive": r4(r.pct_positive),
        }

    stats["by_layer"] = {}
    for m in MEASURE_ORDER:
        for l in LAYER_ORDER:
            s = pair[(pair.human_metric == m) & (pair.mllm_embedding_type == l)]
            sub = layer_tbl[(layer_tbl.human_metric == m)
                            & (layer_tbl.embedding_type == l)]
            stats["by_layer"][f"{m}|{l}"] = {
                "n": int(len(s)), "mean_rho": r4(s.spearman_rho.mean()),
                "pct_positive": r4(100 * (s.spearman_rho > 0).mean()),
                "mean_sd_across_tasks": r4(sub.sd_rho.mean()),
            }

    # ------------------------------------------ vision encoder vs language model
    key = ["question_type", "mllm_model_name", "mllm_model_mode",
           "mllm_condition_label"]
    stats["ve_minus_lm"] = {}
    for m in MEASURE_ORDER:
        w = pair[pair.human_metric == m].pivot_table(
            index=key, columns="mllm_embedding_type", values="spearman_rho")
        d = w["vision_encoder_last"] - w["language_model_last"]
        stats["ve_minus_lm"][m] = {
            "n": int(len(d)), "mean_delta": r4(d.mean()),
            "n_positive": int((d > 0).sum()),
            "pct_positive": r4(100 * (d > 0).mean()),
        }
        mono = 0
        for t in TASK_ORDER:
            v = (layer_tbl[(layer_tbl.human_metric == m)
                           & (layer_tbl.question_type == t)]
                 .set_index("embedding_type").loc[LAYER_ORDER, "mean_rho"])
            mono += int(v.is_monotonic_decreasing)
        stats["ve_minus_lm"][m]["monotone_tasks"] = mono

    # ---------------------------------------------------------- noise ceiling
    nc_rows = []
    for m in MEASURE_ORDER:
        for t in TASK_ORDER:
            if m == "rt_critical":
                r = nc_crit[nc_crit.question_type == t].iloc[0]
                lo, up = r.noise_lower_bound_spearman, r.noise_upper_bound_spearman
                npart = int(r.participant_count_used_full)
                best_label = model_label(r.best_model_name, r.best_model_mode)
                best_layer = r.best_embedding_type
            else:
                r = nc_pkg[(nc_pkg.question_type == t)
                           & (nc_pkg.human_metric == m)].iloc[0]
                lo, up = r.noise_lower_bound_spearman, r.noise_upper_bound_spearman
                npart = int(r.participant_count_used_full)
                best_label = model_label(r.best_model_name, r.best_model_mode)
                best_layer = r.best_embedding_type
            s = pair[(pair.human_metric == m) & (pair.question_type == t)]
            inside = (s.spearman_rho >= lo) & (s.spearman_rho <= up)
            by_layer = {LAYER_LABEL[l]: int(inside[
                s.mllm_embedding_type == l].sum()) for l in LAYER_ORDER}
            nc_rows.append({
                "human_metric": m, "measure_label": HUMAN_MEASURE_LABEL_TEXT[m],
                "question_type": t, "task_label": TASK_LABEL[t],
                "n_participants": npart, "lower": lo, "upper": up,
                "n_configurations": int(len(s)),
                "n_within_band": int(inside.sum()),
                "n_outside_band": int(len(s) - inside.sum()),
                "best_rho": float(s.spearman_rho.max()),
                "best_model": best_label, "best_layer": LAYER_LABEL[best_layer],
                "best_rho_as_pct_of_lower_bound": 100 * s.spearman_rho.max() / lo,
                **{f"n_within_band__{k}": v for k, v in by_layer.items()},
            })
    nc_tbl = pd.DataFrame(nc_rows)
    nc_tbl.to_csv(DERIVED / "table_noise_ceiling.csv", index=False)
    stats["noise_ceiling"] = {}
    for _, r in nc_tbl.iterrows():
        stats["noise_ceiling"][f"{r.human_metric}|{r.question_type}"] = {
            "task_label": r.task_label, "measure_label": r.measure_label,
            "n_participants": int(r.n_participants),
            "lower": r4(r.lower), "upper": r4(r.upper),
            "n_configurations": int(r.n_configurations),
            "n_within_band": int(r.n_within_band),
            "n_outside_band": int(r.n_outside_band),
            "best_rho": r4(r.best_rho), "best_model": r.best_model,
            "best_layer": r.best_layer,
            "best_rho_as_pct_of_lower_bound": r4(r.best_rho_as_pct_of_lower_bound),
            "n_within_band_by_layer": {
                l: int(r[f"n_within_band__{l}"]) for l in LAYER_LABEL.values()},
        }

    # ------------------------------------------------------------ permutation
    stats["permutation"] = {}
    for _, r in perm.iterrows():
        stats["permutation"][f"{r.human_metric}|{r.question_type}"] = {
            "task_label": r.task_label,
            "measure_label": HUMAN_MEASURE_LABEL_TEXT[r.human_metric],
            "observed_rho": r4(r.observed_spearman_rho),
            "n_permutations": int(r.n_permutations),
            "null_q025": r4(r.null_q025_spearman),
            "null_q975": r4(r.null_q975_spearman),
            "p_two_sided": r4(r.p_value_two_sided),
            "q_fdr": r4(r.q_fdr_over_9),
            "significant": bool(r.significant_fdr_005),
            "best_model": r.best_model_label,
            "best_layer": LAYER_LABEL[r.best_embedding_type],
        }
    stats["permutation_summary"] = {
        "n_tests": int(len(perm)),
        "n_significant": int(perm.significant_fdr_005.sum()),
        "n_significant_rt": int(
            perm[perm.human_metric == "rt"].significant_fdr_005.sum()),
        "n_significant_rt_critical": int(
            perm[perm.human_metric == "rt_critical"].significant_fdr_005.sum()),
        "n_significant_corr": int(
            perm[perm.human_metric == "corr"].significant_fdr_005.sum()),
        "n_best_from_vision_encoder": int(
            (perm.best_embedding_type == "vision_encoder_last").sum()),
    }

    # -------------------------------------------------------- RT_critical facts
    tv = crit_val["tasks"]["Concept Verification"]
    stats["rt_critical"] = {
        "t_mean_s": r4(tv["t_critical_mean_s"]),
        "t_sd_s": r4(tv["t_critical_sd_s"]),
        "t_min_s": r4(tv["t_critical_min_s"]),
        "t_max_s": r4(tv["t_critical_max_s"]),
        "rdm_corr_with_rt_onset": {
            crit_val["tasks"][TASK_LABEL[t]]["n_participants"] and TASK_LABEL[t]:
            r4(crit_val["tasks"][TASK_LABEL[t]]["rdm_corr_with_rt_onset"])
            for t in TASK_ORDER},
        "rdm_corr_min": r4(min(v["rdm_corr_with_rt_onset"]
                               for v in crit_val["tasks"].values())),
        "rdm_corr_max": r4(max(v["rdm_corr_with_rt_onset"]
                               for v in crit_val["tasks"].values())),
        "gain_over_rt_onset": {},
    }
    gains = []
    for t in TASK_ORDER:
        for l in LAYER_ORDER:
            a = stats["by_task_layer"][f"rt|{t}|{l}"]["mean_rho"]
            b = stats["by_task_layer"][f"rt_critical|{t}|{l}"]["mean_rho"]
            gains.append(b - a)
    stats["rt_critical"]["gain_over_rt_onset"] = {
        "n_cells": len(gains), "n_positive": int(sum(g > 0 for g in gains)),
        "min": r4(min(gains)), "max": r4(max(gains)), "mean": r4(np.mean(gains)),
    }

    # ------------------------------------------------------- prompt effects
    stats["prompt_effect_language_model"] = {}
    for m in MEASURE_ORDER:
        s = pair[(pair.human_metric == m)
                 & (pair.mllm_embedding_type == "language_model_last")]
        bp = s.groupby("mllm_condition_label")["spearman_rho"].mean()
        n_per = s.groupby("mllm_condition_label").size()
        assert n_per.nunique() == 1
        stats["prompt_effect_language_model"][m] = {
            "by_condition": {k: r4(v) for k, v in bp.items()},
            "n_per_condition": int(n_per.iat[0]),
            "min_mean_rho": r4(bp.min()), "max_mean_rho": r4(bp.max()),
            "range": r4(bp.max() - bp.min()),
        }
    for layer in ["vision_encoder_last", "vision_projection"]:
        s = pair[pair.mllm_embedding_type == layer]
        w = s.pivot_table(index=["human_matrix_id", "mllm_model_name",
                                 "mllm_model_mode"],
                          columns="mllm_condition_label", values="spearman_rho")
        assert np.allclose(w.max(axis=1), w.min(axis=1))

    # ---------------------------------------------------- replication structure
    rep = {"n_alignments_nominal": int(len(pair)),
           "n_alignments_distinct": int(
               pair.drop_duplicates(["human_matrix_id", "rdm_hash"]).shape[0]),
           "by_stage": {}}
    for l in LAYER_ORDER:
        s = pair[pair.mllm_embedding_type == l]
        rep["by_stage"][LAYER_LABEL[l]] = {
            "n_rdm_files": int(s.mllm_group_slug.nunique()),
            "n_distinct_rdms": int(s.rdm_hash.nunique()),
            "n_alignments_nominal": int(len(s)),
            "n_alignments_distinct": int(
                s.drop_duplicates(["human_matrix_id", "rdm_hash"]).shape[0]),
        }
    ded = []
    for m in MEASURE_ORDER:
        for t in TASK_ORDER:
            for l in LAYER_ORDER:
                s = pair[(pair.human_metric == m) & (pair.question_type == t)
                         & (pair.mllm_embedding_type == l)]
                u = s.drop_duplicates("rdm_hash")
                ded.append({"human_metric": m, "question_type": t,
                            "embedding_type": l, "n_nominal": len(s),
                            "mean_rho_nominal": s.spearman_rho.mean(),
                            "n_distinct": len(u),
                            "mean_rho_distinct": u.spearman_rho.mean()})
    ded = pd.DataFrame(ded)
    ded["mean_shift"] = ded.mean_rho_distinct - ded.mean_rho_nominal
    ded.to_csv(DERIVED / "table_replication_structure.csv", index=False)
    rep["max_abs_mean_shift_after_dedup"] = r4(ded.mean_shift.abs().max())
    rep["sign_preserved_after_dedup"] = bool(
        (np.sign(ded.mean_rho_nominal) == np.sign(ded.mean_rho_distinct)).all())
    stats["replication"] = rep

    # -------------------------------------------------------- cross-checks
    checks = {}
    s06 = pd.read_csv(P.RSA_DIR / "06_rsa_summary_by_human_and_layer.csv")
    mine = (pkg_pairs.groupby(["human_matrix_id", "mllm_embedding_type"])
            .spearman_rho.mean().reset_index())
    m06 = s06.merge(mine, on=["human_matrix_id", "mllm_embedding_type"])
    checks["matches_package_summary_06"] = bool(
        np.allclose(m06.mean_spearman_rho, m06.spearman_rho, atol=1e-12)
        and len(m06) == 18)
    s09 = pd.read_csv(P.RSA_DIR / "09_rsa_overall_metric_stats.csv").set_index(
        "human_metric")
    checks["matches_package_summary_09"] = all(
        np.isclose(pkg_pairs[pkg_pairs.human_metric == m].spearman_rho.mean(),
                   s09.loc[m, "mean_spearman_rho"]) for m in ["rt", "corr"])
    checks["pairings_equals_product"] = (
        len(pair) == stats["design"]["n_model_configurations"] * n_tasks * n_measures)
    checks["rt_critical_validated"] = all(
        v["max_abs_diff_rt_onset_vs_package"] == 0.0
        and v["t_critical_max_within_video_sd"] < 1e-9
        for v in crit_val["tasks"].values())
    stats["checks"] = checks
    assert all(checks.values()), f"consistency check failed: {checks}"

    with open(DERIVED / "rsa_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 220)
    print("design:", json.dumps(stats["design"], indent=2))
    print("\nmean Spearman rho by measure x task x stage:")
    print(layer_tbl.pivot_table(index=["measure_label", "task_label"],
                                columns="layer_label", values="mean_rho")
          [list(LAYER_LABEL.values())].round(4).to_string())
    print("\nnoise ceiling:")
    print(nc_tbl[["measure_label", "task_label", "lower", "upper", "best_rho",
                  "n_within_band", "n_configurations"]].round(4).to_string(index=False))
    print("\nchecks:", checks)
    print(f"\nwrote {DERIVED/'rsa_stats.json'}")


if __name__ == "__main__":
    main()
