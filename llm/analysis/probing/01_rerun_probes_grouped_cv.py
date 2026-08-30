#!/usr/bin/env python3
"""
Re-run the probes with a design that can actually estimate decodability.

The released runs cannot: the overall probe leaks whole videos between train and
test, and the task-conditioned probe scores on 6 held-out items after selecting
on 4. The pooled embeddings the probes were trained on are in the package
(RDM/Data/matrices_ready/*.npy, 48 videos x d, with row_metadata/*.csv giving
the labels), so the analysis can be redone here.

What changes:
  * every video is held out exactly once, via stratified k-fold over the 48
    videos, so the estimate uses all 48 predictions rather than 6;
  * for task identity, where each video contributes three rows (one per
    question), folds are formed over videos, so no video appears on both sides;
  * a regularised linear readout replaces the 512-unit MLP. With 40 training
    rows and up to 4096 dimensions the MLP has orders of magnitude more
    parameters than samples; "linear readout" is what the claim needs anyway.
    Regularisation strength is chosen by an inner split on the training folds
    only.

Outputs derived/rerun_probe_results.csv and derived/rerun_probe_summary.json

Usage:  python 01_rerun_probes_grouped_cv.py [--jobs N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Every fit here is tiny (<=120x4096). Multi-threaded BLAS spends more time
# waking its pool than doing the arithmetic -- a single fold took 145 ms for a
# 26x4096 @ 4096x25 matmul. Pin to one thread and parallelise across runs.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import paths as P  # noqa: E402
MAT = P.MAT
META = P.META
DERIVED = P.derived_dir(__file__)

from model_naming import STAGE_LABEL, STAGE_ORDER, label as model_label  # noqa: E402

N_FOLDS = 6
C_GRID = [0.01, 0.1, 1.0, 10.0]
# PCA keeps every estimable direction (n_train - 1), so it is a lossless
# rotation of the centred training data rather than a feature-selection step:
# the probe is equivalent to one on the raw standardised features, but is
# better conditioned and far cheaper. Truncating it is not neutral -- at 25
# components the task-identity signal at the language model reads 35.6%, and at
# 60+ it reads 86.1%, because that signal lives off the high-variance axes.
SEEDS = [0, 1, 2]
CHANCE = {"physical_concept": 1 / 3, "plausibility": 1 / 2, "task_identity": 1 / 3}
SLUG_MODEL = {
    "glm_4_1v_base": ("GLM-4.1V-9B-Base", "base"),
    "glm_4_1v_thinking": ("GLM-4.1V-9B-Thinking", "think"),
    "qwen": ("Qwen3-VL-8B-Instruct", "base"),
    "qwen_thinking": ("Qwen3-VL-8B-Thinking", "think"),
    "internvl3_5_base": ("InternVL3.5-8B", "base"),
    "internvl3_5_think": ("InternVL3.5-8B", "think"),
    "mimo_embodied_base": ("MiMo-Embodied-7B", "base"),
    "mimo_embodied_think": ("MiMo-Embodied-7B", "think"),
    "robobrain2_5": ("RoboBrain2.5-8B-NV", "base"),
    "rynnbrain_8b": ("RynnBrain-8B", "base"),
    "rynnbrain_cop": ("RynnBrain-CoP-8B", "base"),
}
TASK_SLUG = {"category": "Category", "voe": "VoE", "sm": "SM"}


def parse_slug(slug: str):
    """<model>__<setting>__<prompt>__<task>__<stage>"""
    for m in sorted(SLUG_MODEL, key=len, reverse=True):
        if slug.startswith(m + "__"):
            rest = slug[len(m) + 2:]
            setting, prompt, task, stage = rest.split("__", 3)
            return m, setting, prompt, task, stage
    raise ValueError(slug)


def split(X, y, groups, k, seed):
    """k folds; group-aware when a video contributes more than one row."""
    if groups is None:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y))
    # GroupKFold is deterministic, so permute the group ids to vary folds by seed
    rng = np.random.default_rng(seed)
    perm = {g: i for i, g in enumerate(rng.permutation(np.unique(groups)))}
    gidx = np.array([perm[g] for g in groups])
    return list(GroupKFold(k).split(X, y, groups=gidx))


def reduce_(Xtr, Xte):
    """Standardise and PCA-reduce, fitted on the training rows only.

    svd_solver='full' is an economy SVD; the 'auto' default picks the
    randomized solver here and is ~75x slower for identical output.
    """
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    p = PCA(n_components=min(len(Xtr) - 1, Xtr.shape[1]), svd_solver="full",
            random_state=0).fit(A)
    return p.transform(A), p.transform(B)


def cv_accuracy(X, y, groups, seed):
    """Held-out accuracy with every sample predicted exactly once."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in split(X, y, groups, N_FOLDS, seed):
        Xtr, ytr = X[tr], y[tr]
        gtr = None if groups is None else groups[tr]
        # choose C on an inner split of the training folds only
        score = np.zeros(len(C_GRID))
        for itr, ite in split(Xtr, ytr, gtr, 3, seed):
            A, B = reduce_(Xtr[itr], Xtr[ite])
            for ci, c in enumerate(C_GRID):
                score[ci] += LogisticRegression(C=c, max_iter=5000) \
                    .fit(A, ytr[itr]).score(B, ytr[ite])
        best_c = C_GRID[int(np.argmax(score))]
        A, B = reduce_(Xtr, X[te])
        pred[te] = LogisticRegression(C=best_c, max_iter=5000) \
            .fit(A, ytr).predict(B)
    return float((pred == y).mean())


def _stat(accs, n, chance):
    """Mean over seeds, plus a one-sided exact binomial test against chance.

    Every item is predicted exactly once per seed, so the seed-mean accuracy
    corresponds to a count out of n. Folds share training data, so the count is
    not a sum of independent Bernoulli trials and the p-value is indicative
    rather than exact; it is used only to screen which runs are above chance.
    """
    m = float(np.mean(accs))
    k = int(round(m * n))
    return {"acc_mean": m, "acc_sd": float(np.std(accs, ddof=1)),
            "n_correct": k, "chance": chance,
            "p_binom": float(binomtest(k, n, chance, "greater").pvalue)}


def run_video_target(slug, target):
    md = pd.read_csv(META / f"{slug}.csv")
    X = np.load(MAT / f"{slug}.npy").astype(np.float64)
    y = md[target].to_numpy()
    assert len(X) == len(y) == 48
    m, setting, prompt, task, stage = parse_slug(slug)
    accs = [cv_accuracy(X, y, None, s) for s in SEEDS]
    return {**_stat(accs, 48, CHANCE[target]),
            "analysis": "grouped_cv", "target": target,
            "model": model_label(*SLUG_MODEL[m]), "setting": setting,
            "prompt": prompt, "task": TASK_SLUG[task], "stage_key": stage,
            "stage": STAGE_LABEL[stage], "n_samples": 48}


def run_task_identity(model_slug, setting, prompt, stage):
    Xs, ys, gs = [], [], []
    for t in TASK_SLUG:
        slug = f"{model_slug}__{setting}__{prompt}__{t}__{stage}"
        md = pd.read_csv(META / f"{slug}.csv")
        Xs.append(np.load(MAT / f"{slug}.npy").astype(np.float64))
        ys.append(md.question_type.to_numpy())
        gs.append(md.video_id.to_numpy())
    X, y, g = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
    assert len(X) == 144 and len(np.unique(g)) == 48
    accs = [cv_accuracy(X, y, g, s) for s in SEEDS]
    return {**_stat(accs, 144, CHANCE["task_identity"]),
            "analysis": "grouped_cv", "target": "task_identity",
            "model": model_label(*SLUG_MODEL[model_slug]), "setting": setting,
            "prompt": prompt, "task": "all", "stage_key": stage,
            "stage": STAGE_LABEL[stage], "n_samples": 144}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    slugs = sorted(p.stem for p in MAT.glob("*.npy"))
    assert len(slugs) == 396, len(slugs)

    jobs = [delayed(run_video_target)(s, t) for s in slugs
            for t in ("physical_concept", "plausibility")]
    combos = {(parse_slug(s)[0], parse_slug(s)[1], parse_slug(s)[2],
               parse_slug(s)[4]) for s in slugs}
    jobs += [delayed(run_task_identity)(*c) for c in sorted(combos)]
    print(f"{len(jobs)} probe runs x {len(SEEDS)} seeds x {N_FOLDS} folds")

    rows = Parallel(n_jobs=args.jobs, verbose=5)(jobs)
    res = pd.DataFrame(rows)
    res.to_csv(DERIVED / "rerun_probe_results.csv", index=False)

    summary = {}
    for target in ["physical_concept", "plausibility", "task_identity"]:
        s = res[res.target == target]
        summary[target] = {"chance": CHANCE[target], "n_runs": int(len(s))}
        for st in STAGE_ORDER:
            v = s[s.stage_key == st].acc_mean
            summary[target][STAGE_LABEL[st]] = {
                "mean": round(float(v.mean()), 6),
                "sd": round(float(v.std(ddof=1)), 6),
                "min": round(float(v.min()), 6), "max": round(float(v.max()), 6),
                "n_groups": int(len(v)),
                "n_above_chance": int((v > CHANCE[target]).sum()),
            }
    with open(DERIVED / "rerun_probe_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 200)
    print("\nheld-out accuracy, every video predicted exactly once (%):")
    print((res.pivot_table(index="target", columns="stage", values="acc_mean")
           [list(STAGE_LABEL.values())] * 100).round(1).to_string())
    print("\ngroups above chance:")
    for t, d in summary.items():
        print(f"  {t:18s} chance {d['chance']*100:.1f}%: " +
              ", ".join(f"{STAGE_LABEL[st]} {d[STAGE_LABEL[st]]['n_above_chance']}"
                        f"/{d[STAGE_LABEL[st]]['n_groups']}" for st in STAGE_ORDER))
    print(f"\nwrote {DERIVED/'rerun_probe_results.csv'}")


if __name__ == "__main__":
    main()
