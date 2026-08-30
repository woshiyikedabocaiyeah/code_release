#!/usr/bin/env python3
"""
Does the probe family change the answer? Re-run under the source paper's MLP.

Ballout, Jassim & Bruni ("Pixels to Principles") describe their probes as
"linear" in the text but specify a one-hidden-layer MLP (hidden size 512, ReLU,
no dropout, 500 epochs, cross-entropy) in their setup. Our re-analysis uses a
regularised linear readout, so the two differ -- and the question is whether
that difference is what produces our result.

This runs the paper's readout through our folds: same embeddings, same grouped
cross-validation, same seeds, only the classifier swapped. Anything that changes
is attributable to the probe family; anything that does not, is not.

Scope is the non-embodied simple prompt under Concept Verification, all eleven
model variants at all three stages, which is the cell the main text quotes.

The probe is fitted on the same lossless PCA rotation the linear probe uses
(n_train - 1 components, so no information is discarded); with 4,096 raw
dimensions and 40 training rows the rotation only conditions the problem.

Outputs derived/architecture_control.csv

Usage:  python 08_architecture_control.py [--jobs N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rerun", HERE / "01_rerun_probes_grouped_cv.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

PROMPT = ("non_embodied", "simple")
TASK = "category"


def mlp_cv_accuracy(X, y, groups, seed):
    """R.cv_accuracy with the paper's readout in place of the linear one."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in R.split(X, y, groups, R.N_FOLDS, seed):
        A, B = R.reduce_(X[tr], X[te])
        clf = MLPClassifier(hidden_layer_sizes=(512,), activation="relu",
                            max_iter=500, random_state=seed)
        with np.errstate(all="ignore"):
            import warnings
            warnings.filterwarnings("ignore")
            clf.fit(A, y[tr])
        pred[te] = clf.predict(B)
    return float((pred == y).mean())


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
        X, y, g, n = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), 144
    else:
        slug = f"{model_slug}__{setting}__{prompt}__{TASK}__{stage}"
        md = pd.read_csv(R.META / f"{slug}.csv")
        X = np.load(R.MAT / f"{slug}.npy").astype(np.float64)
        y, g, n = md[target].to_numpy(), None, 48

    lin = [R.cv_accuracy(X, y, g, s) for s in R.SEEDS]
    mlp = [mlp_cv_accuracy(X, y, g, s) for s in R.SEEDS]
    return {"model": R.model_label(*R.SLUG_MODEL[model_slug]),
            "stage": R.STAGE_LABEL[stage], "stage_key": stage,
            "target": target, "n": n,
            "chance_pct": 100 * R.CHANCE[target],
            "linear_pct": 100 * float(np.mean(lin)),
            "mlp_512_relu_pct": 100 * float(np.mean(mlp))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()

    jobs = [delayed(one)(m, st, tg)
            for m in R.SLUG_MODEL for st in R.STAGE_ORDER
            for tg in ("physical_concept", "plausibility", "task_identity")]
    print(f"  {len(jobs)} runs x 2 probe families x {len(R.SEEDS)} seeds")
    rows = Parallel(n_jobs=a.jobs, verbose=1)(jobs)
    df = pd.DataFrame(rows)
    out = R.DERIVED / "architecture_control.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n  wrote {out}\n")
    g = (df.groupby(["target", "stage"], sort=False)
         [["chance_pct", "linear_pct", "mlp_512_relu_pct"]].mean().round(1))
    print(g.to_string())
    df["diff"] = df.mlp_512_relu_pct - df.linear_pct
    print(f"\n  MLP − linear: mean {df['diff'].mean():+.1f} pp, "
          f"range {df['diff'].min():+.1f} to {df['diff'].max():+.1f} pp")


if __name__ == "__main__":
    main()
