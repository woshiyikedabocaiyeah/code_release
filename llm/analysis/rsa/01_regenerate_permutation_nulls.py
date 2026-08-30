#!/usr/bin/env python3
"""
Permutation tests for the best-matching configuration of every human RDM.

The package ships permutation results for its 6 human matrices (RT_onset and
correctness x 3 tasks) but stores only the summary of each null, not the draws.
RT_critical adds 3 more matrices, which the package never tested. This script
therefore runs all 9 from scratch with the package's protocol -- hold the human
RDM fixed, permute the model's video labels, 10,000 times -- and:

  * asserts that the 6 it shares with the package reproduce the published
    observed rho exactly and the published null summary to Monte-Carlo
    precision (an independent replication of the original test);
  * applies Benjamini-Hochberg across all 9 tests, which is the correct family
    once RT_critical is reported alongside the other two measures.

Outputs derived/permutation_nulls.npz and derived/permutation_tests.csv

Usage:  python 01_regenerate_permutation_nulls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import paths as P  # noqa: E402
DERIVED = P.derived_dir(__file__)

from model_naming import TASK_LABEL, TASK_ORDER, label as model_label  # noqa: E402

N_PERM = 10_000
SEED = 20260419  # rng_seed recorded in the package's permutation results
TRI = np.triu_indices(48, k=1)
TASK_FILE = {"Category": "cat", "VoE": "voe", "SM": "sensor"}
MEASURES = ["rt", "rt_critical", "corr"]


def upper(m: np.ndarray) -> np.ndarray:
    return m[TRI]


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, monotone-enforced."""
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return q


def main() -> None:
    pkg_perm = pd.read_csv(
        P.PERMUTATION_TEST / "01_permutation_tests_best_model_per_human.csv"
    ).set_index(["human_metric", "question_type"])
    pkg_pairs = pd.read_csv(P.RSA_DIR / "04_rsa_pairwise_results.csv")
    crit_pairs = pd.read_csv(
        P.rt_critical_dir(DERIVED) / "rsa_pairwise_rt_critical.csv")
    pairs = pd.concat([pkg_pairs, crit_pairs], ignore_index=True)

    rng_master = np.random.default_rng(SEED)
    nulls, rows = {}, []

    for measure in MEASURES:
        for task in TASK_ORDER:
            f = TASK_FILE[task]
            hid = (f"human__{f}__rt_critical" if measure == "rt_critical"
                   else f"human__{f}__{measure}")
            H = (np.load(P.rt_critical_dir(DERIVED) / "matrices" / f"{hid}.npy")
                 if measure == "rt_critical"
                 else np.load(P.HUMAN_ALIGNED / "matrices" / f"{hid}.npy")
                 ).astype(float)

            sel = pairs[(pairs.human_metric == measure)
                        & (pairs.question_type == task)]
            best = sel.loc[sel.spearman_rho.idxmax()]
            M = np.load(
                P.RDM_MEAN_POOL
                / f"{best.mllm_group_slug.replace('__mean_pool', '')}__mean_pool.npy"
            ).astype(float)

            h_vec = upper(H)
            rho_obs = spearmanr(h_vec, upper(M)).statistic
            assert np.isclose(rho_obs, best.spearman_rho, atol=1e-6)

            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            null = np.empty(N_PERM)
            idx = np.arange(48)
            for k in range(N_PERM):
                p = rng.permutation(idx)
                null[k] = spearmanr(h_vec, upper(M[np.ix_(p, p)])).statistic
            nulls[hid] = null.astype(np.float32)

            row = {
                "human_matrix_id": hid,
                "question_type": task,
                "task_label": TASK_LABEL[task],
                "human_metric": measure,
                "best_model_label": model_label(best.mllm_model_name,
                                                best.mllm_model_mode),
                "best_setting": best.mllm_setting,
                "best_prompt_family": best.mllm_prompt_family,
                "best_embedding_type": best.mllm_embedding_type,
                "observed_spearman_rho": rho_obs,
                "n_permutations": N_PERM,
                "null_mean_spearman": null.mean(),
                "null_sd_spearman": null.std(ddof=1),
                "null_q025_spearman": np.quantile(null, 0.025),
                "null_q975_spearman": np.quantile(null, 0.975),
                # (b + 1) / (m + 1): never reports p = 0
                "p_value_two_sided": (np.sum(np.abs(null) >= abs(rho_obs)) + 1)
                / (N_PERM + 1),
            }
            if (measure, task) in pkg_perm.index:
                pk = pkg_perm.loc[(measure, task)]
                row["pkg_observed_rho"] = pk.observed_spearman_rho
                row["pkg_null_sd"] = pk.null_sd_spearman
                row["pkg_q_fdr_over_6"] = pk.p_value_two_sided_fdr_bh
            rows.append(row)

    res = pd.DataFrame(rows)
    res["q_fdr_over_9"] = bh(res.p_value_two_sided.to_numpy())
    res["significant_fdr_005"] = res.q_fdr_over_9 < 0.05
    res.to_csv(DERIVED / "permutation_tests.csv", index=False)
    np.savez_compressed(DERIVED / "permutation_nulls.npz", **nulls)

    # ------------------------------------------- replication of the package
    shared = res[res.pkg_observed_rho.notna()]
    assert np.allclose(shared.observed_spearman_rho, shared.pkg_observed_rho,
                       atol=1e-6), "observed rho disagrees with the package"
    assert np.allclose(shared.null_sd_spearman, shared.pkg_null_sd, rtol=0.05), \
        "regenerated null sd disagrees with the package"
    same_call = ((shared.q_fdr_over_9 < 0.05)
                 == (shared.pkg_q_fdr_over_6 < 0.05)).all()
    assert same_call, "correcting over 9 flips a significance call"

    pd.set_option("display.width", 220)
    print(res[["human_metric", "task_label", "best_model_label",
               "best_embedding_type", "observed_spearman_rho",
               "p_value_two_sided", "q_fdr_over_9", "significant_fdr_005"]]
          .round(5).to_string(index=False))
    print(f"\nsignificant: {int(res.significant_fdr_005.sum())} / {len(res)}")
    print("package's 6 tests reproduced; correcting over 9 changes no call")
    print(f"wrote {DERIVED / 'permutation_tests.csv'}")


if __name__ == "__main__":
    main()
