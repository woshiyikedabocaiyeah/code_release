#!/usr/bin/env python3
"""
Which probe family should carry the claim? Measure, don't argue.

Our headline result for plausibility is negative, and a negative result is only
as good as the probe that failed to find anything: a reviewer's first move is
"you did not look hard enough". So the question is not which probe is nicer in
principle but whether the answer survives a probe with far more capacity, and
whether that capacity is real rather than nominal.

Three numbers per run, for both readouts:

  train     accuracy on the training folds. Establishes that the MLP really can
            fit these data -- if it memorises the training set and still fails
            on held-out rows, the negative result is not a capacity artefact.
  test      held-out accuracy under the same grouped folds.
  control   held-out accuracy after permuting the labels (Hewitt & Liang's
            control task). A probe that scores above chance here is fitting the
            split, not the representation.

selectivity = test - control is the quantity that makes a probe's number mean
something; it is reported per run.

Outputs derived/probe_capacity_diagnostics.csv

Usage:  python 09_probe_capacity_diagnostics.py [--jobs N]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "rerun", HERE / "01_rerun_probes_grouped_cv.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

PROMPT = ("non_embodied", "simple")
TASK = "category"


def make(kind, seed):
    if kind == "linear":
        return LogisticRegression(C=1.0, max_iter=5000)
    return MLPClassifier(hidden_layer_sizes=(512,), activation="relu",
                         max_iter=500, random_state=seed)


def fit_folds(X, y, groups, seed, kind):
    """Train and held-out accuracy over the same folds the main analysis uses."""
    pred = np.empty(len(y), dtype=object)
    train_hits = train_n = 0
    for tr, te in R.split(X, y, groups, R.N_FOLDS, seed):
        A, B = R.reduce_(X[tr], X[te])
        clf = make(kind, seed).fit(A, y[tr])
        pred[te] = clf.predict(B)
        train_hits += (clf.predict(A) == y[tr]).sum()
        train_n += len(tr)
    return float((pred == y).mean()), train_hits / train_n


def one(model_slug, stage, target):
    setting, prompt = PROMPT
    if target == "task_identity":
        Xs, ys, gs = [], [], []
        for t in R.TASK_SLUG:
            slug = f"{model_slug}__{setting}__{prompt}__{t}__{stage}"
            md = pd.read_csv(R.META / f"{slug}.csv")
            Xs.append(np.load(R.MAT / f"{slug}.npy").astype(np.float64))
            ys.append(md.question_type.to_numpy())
            gs.append(md.video_id.to_numpy())
        X, y, g = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
    else:
        slug = f"{model_slug}__{setting}__{prompt}__{TASK}__{stage}"
        md = pd.read_csv(R.META / f"{slug}.csv")
        X = np.load(R.MAT / f"{slug}.npy").astype(np.float64)
        y, g = md[target].to_numpy(), None

    out = {"model": R.model_label(*R.SLUG_MODEL[model_slug]),
           "stage": R.STAGE_LABEL[stage], "target": target,
           "chance_pct": 100 * R.CHANCE[target]}
    for kind in ("linear", "mlp_512_relu"):
        te, trn, ctl = [], [], []
        for s in R.SEEDS:
            a, b = fit_folds(X, y, g, s, kind)
            te.append(a); trn.append(b)
            # control task: same folds, labels permuted (by group where grouped)
            rng = np.random.default_rng(1000 + s)
            if g is None:
                # one label per video: permute across videos
                yp = rng.permutation(y)
            else:
                # the label varies within a video (question type), so permute
                # within each video. Permuting at video level would give every
                # row of a video the same label and leave folds single-class.
                yp = y.copy()
                for k in np.unique(g):
                    i = np.where(g == k)[0]
                    yp[i] = rng.permutation(y[i])
            ctl.append(fit_folds(X, yp, g, s, kind)[0])
        out[f"{kind}_train_pct"] = 100 * float(np.mean(trn))
        out[f"{kind}_test_pct"] = 100 * float(np.mean(te))
        out[f"{kind}_test_sd_pp"] = 100 * float(np.std(te, ddof=1))
        out[f"{kind}_control_pct"] = 100 * float(np.mean(ctl))
        out[f"{kind}_selectivity_pp"] = out[f"{kind}_test_pct"] - out[f"{kind}_control_pct"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    jobs = [delayed(one)(m, st, tg) for m in R.SLUG_MODEL
            for st in R.STAGE_ORDER
            for tg in ("physical_concept", "plausibility", "task_identity")]
    print(f"  {len(jobs)} runs x 2 families x {len(R.SEEDS)} seeds x (real + control)")
    df = pd.DataFrame(Parallel(n_jobs=a.jobs, verbose=1)(jobs))
    out = R.DERIVED / "probe_capacity_diagnostics.csv"
    df.to_csv(out, index=False)
    print(f"\n  wrote {out}\n")

    cols = ["chance_pct",
            "linear_train_pct", "linear_test_pct", "linear_test_sd_pp",
            "linear_control_pct", "linear_selectivity_pp",
            "mlp_512_relu_train_pct", "mlp_512_relu_test_pct",
            "mlp_512_relu_test_sd_pp", "mlp_512_relu_control_pct",
            "mlp_512_relu_selectivity_pp"]
    g = df.groupby(["target", "stage"], sort=False)[cols].mean().round(1)
    pd.set_option("display.width", 200)
    print(g.to_string())


if __name__ == "__main__":
    main()
