#!/usr/bin/env python3
"""
Bring RT_critical into the RSA as a third human measure.

The analysis package only ships RT_onset and correctness RDMs. The trial-level
RT_critical data arrived separately, so this script builds the missing layer of
the analysis with exactly the procedures the package used, and writes it in the
package's own schema so everything downstream is uniform.

Validation performed before anything is computed:
  * RT_onset reconstructed from the trial-level file must equal the package's
    wide matrices exactly (this proves it is the same source data);
  * the rebuilt RT_onset RDM must equal the package's stored RDM;
  * RT_onset - RT_critical must be constant within a video (the Methods define
    the critical event as a property of the video).

Outputs, all under derived/rt_critical/:
  matrices/human__{cat,voe,sensor}__rt_critical.npy
  rsa_pairwise_rt_critical.csv      396 alignments, package schema
  noise_ceiling_rt_critical.csv     3 ceilings, split-half + bootstrap
  validation.json

Usage:  python 05_build_rt_critical.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, zscore

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
import paths as P  # noqa: E402
OUT = P.derived_dir(__file__) / "rt_critical"
(OUT / "matrices").mkdir(parents=True, exist_ok=True)

from model_naming import TASK_LABEL, TASK_ORDER  # noqa: E402

P.require_human()
TRIAL_CSV = P.HUMAN_BEHAVIOUR / "inclusion_flags.csv"
TASK_FILE = {"Category": "cat", "VoE": "voe", "SM": "sensor"}
TASK_FROM_NAME = {v: k for k, v in TASK_LABEL.items()}
TRI = np.triu_indices(48, k=1)
N_SPLIT = N_BOOT = 2000
SEED = 20260419


def rt_rdm(X: np.ndarray) -> np.ndarray:
    """Package recipe: within-participant z-score, then correlation distance."""
    M = np.apply_along_axis(lambda c: zscore(c, nan_policy="omit"), 0, X)
    D = 1.0 - np.corrcoef(M)
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2.0


def main() -> None:
    trials = pd.read_csv(TRIAL_CSV)
    trials["video_id"] = trials.Video.str.extract(r"([0-9a-f]{64})")[0]
    trials["question_type"] = trials.task_name.map(TASK_FROM_NAME)
    assert trials.question_type.notna().all(), "unmapped task_name"

    val: dict = {"source": str(TRIAL_CSV.relative_to(P.DATA)), "tasks": {}}
    rdms: dict[str, np.ndarray] = {}

    for task in TASK_ORDER:
        f = TASK_FILE[task]
        order = pd.read_csv(
            P.HUMAN_ALIGNED / "video_orders" / f"{f}.csv"
        ).video_id.tolist()
        sub = trials[trials.question_type == task]

        Xo = sub.pivot_table(index="video_id", columns="Subject",
                             values="RT_onset").loc[order]
        Xc = sub.pivot_table(index="video_id", columns="Subject",
                             values="RT_critical").loc[order]
        assert list(Xo.columns) == list(Xc.columns)

        # (1) same source data as the package's RT_onset matrices?
        wide = pd.read_csv(P.HUMAN_RDM / f"rt_wide_{f}.csv")
        vid = wide[wide.columns[0]].str.extract(r"([0-9a-f]{64})")[0]
        wide = (wide.assign(video_id=vid).set_index("video_id")
                    .drop(columns=[wide.columns[0]]).loc[order])
        cols = [str(c) for c in Xo.columns]
        assert set(cols) == set(wide.columns), f"{task}: participant sets differ"
        max_diff_onset = float(np.nanmax(np.abs(
            wide[cols].to_numpy(float) - Xo.to_numpy(float))))
        assert max_diff_onset == 0.0, f"{task}: RT_onset differs from the package"

        # (2) rebuilt RT_onset RDM equals the stored one?
        ref = np.load(P.HUMAN_ALIGNED / "matrices" / f"human__{f}__rt.npy")
        max_diff_rdm = float(np.abs(rt_rdm(Xo.to_numpy(float)) - ref).max())
        assert max_diff_rdm < 1e-6, f"{task}: RDM builder mismatch"

        # (3) is the critical-event time a per-video constant?
        t = (Xo.to_numpy(float) - Xc.to_numpy(float))
        within_video_sd = float(np.nanmax(t.std(axis=1)))
        assert within_video_sd < 1e-9, f"{task}: t_critical varies within a video"
        t_video = t[:, 0]

        Xc_np = Xc.to_numpy(float)
        R = rt_rdm(Xc_np)
        np.save(OUT / "matrices" / f"human__{f}__rt_critical.npy", R)
        rdms[task] = R

        val["tasks"][TASK_LABEL[task]] = {
            "n_participants": int(Xc.shape[1]),
            "n_videos": int(Xc.shape[0]),
            "max_abs_diff_rt_onset_vs_package": max_diff_onset,
            "max_abs_diff_rdm_vs_package": max_diff_rdm,
            "t_critical_max_within_video_sd": within_video_sd,
            "t_critical_mean_s": float(t_video.mean()),
            "t_critical_sd_s": float(t_video.std()),
            "t_critical_min_s": float(t_video.min()),
            "t_critical_max_s": float(t_video.max()),
            "n_negative_rt_critical": int((Xc_np < 0).sum()),
            "frac_negative_rt_critical": float((Xc_np < 0).mean()),
            "rdm_corr_with_rt_onset": float(
                np.corrcoef(R[TRI], ref[TRI])[0, 1]),
            "spearman_t_critical_vs_video_mean_rt_onset": float(
                spearmanr(t_video, Xo.to_numpy(float).mean(axis=1)).statistic),
        }

    # ---------------------------------------------------------- alignments
    pair = pd.read_csv(P.RSA_DIR / "04_rsa_pairwise_results.csv")
    template = pair[pair.human_metric == "rt"].copy()
    rows = []
    for _, r in template.iterrows():
        task = r.question_type
        slug = r.mllm_group_slug.replace("__mean_pool", "")
        M = np.load(P.RDM_MEAN_POOL / f"{slug}__mean_pool.npy")
        h, m = rdms[task][TRI], M[TRI]
        out = r.to_dict()
        out.update({
            "human_matrix_id": f"human__{TASK_FILE[task]}__rt_critical",
            "human_metric": "rt_critical",
            "pair_id": r.pair_id.replace("__rt__", "__rt_critical__"),
            "spearman_rho": spearmanr(h, m).statistic,
            "pearson_r": pearsonr(h, m).statistic,
        })
        rows.append(out)
    align = pd.DataFrame(rows)
    assert len(align) == 396
    align.to_csv(OUT / "rsa_pairwise_rt_critical.csv", index=False)

    # -------------------------------------------------------- noise ceiling
    nc_rows = []
    for task in TASK_ORDER:
        f = TASK_FILE[task]
        sub = trials[trials.question_type == task]
        order = pd.read_csv(
            P.HUMAN_ALIGNED / "video_orders" / f"{f}.csv"
        ).video_id.tolist()
        X = sub.pivot_table(index="video_id", columns="Subject",
                            values="RT_critical").loc[order].to_numpy(float)
        full = rdms[task][TRI]
        rng = np.random.default_rng(SEED)
        n = X.shape[1]
        lower, upper = [], []
        for _ in range(N_SPLIT):
            p = rng.permutation(n)
            a, b = p[: n // 2], p[n // 2:]
            r = spearmanr(rt_rdm(X[:, a])[TRI], rt_rdm(X[:, b])[TRI]).statistic
            lower.append(2 * r / (1 + r))           # Spearman-Brown
            bs = rng.integers(0, n, n)
            upper.append(spearmanr(rt_rdm(X[:, bs])[TRI], full).statistic)
        a_task = align[align.question_type == task]
        lo, up = float(np.mean(lower)), float(np.mean(upper))
        nc_rows.append({
            "human_matrix_id": f"human__{f}__rt_critical",
            "question_type": task,
            "human_metric": "rt_critical",
            "participant_count_used_full": n,
            "noise_lower_bound_spearman": lo,
            "noise_upper_bound_spearman": up,
            "n_configurations": len(a_task),
            "n_within_ceiling_band": int(
                ((a_task.spearman_rho >= lo) & (a_task.spearman_rho <= up)).sum()),
            "best_model_spearman": float(a_task.spearman_rho.max()),
            **a_task.loc[a_task.spearman_rho.idxmax(),
                         ["mllm_group_slug", "mllm_model_name", "mllm_model_mode",
                          "mllm_setting", "mllm_prompt_family",
                          "mllm_embedding_type"]]
            .rename(lambda c: "best_" + c.replace("mllm_", "")).to_dict(),
        })
    nc = pd.DataFrame(nc_rows)
    nc["best_model_within_spearman_ceiling"] = (
        (nc.best_model_spearman >= nc.noise_lower_bound_spearman)
        & (nc.best_model_spearman <= nc.noise_upper_bound_spearman))
    nc.to_csv(OUT / "noise_ceiling_rt_critical.csv", index=False)

    with open(OUT / "validation.json", "w") as fh:
        json.dump(val, fh, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 220)
    print("validation:\n", json.dumps(val, indent=2, ensure_ascii=False))
    print("\nmean rho by task x stage:")
    print((align.groupby(["question_type", "mllm_embedding_type"])
           .spearman_rho.mean().unstack()
           [["vision_encoder_last", "vision_projection", "language_model_last"]]
           ).round(4).to_string())
    print("\nnoise ceiling:")
    print(nc[["question_type", "noise_lower_bound_spearman",
              "noise_upper_bound_spearman", "best_model_spearman",
              "n_within_ceiling_band", "n_configurations"]].round(4)
          .to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
