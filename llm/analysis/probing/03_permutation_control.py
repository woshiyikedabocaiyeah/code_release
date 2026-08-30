#!/usr/bin/env python3
"""
Label-permutation null for the grouped-CV probes.

The re-run returns plausibility accuracies far below 50%. Two things could
produce that: a genuinely anti-clustered representation, in which a held-out
video's nearest training neighbours carry the opposite label (which is what
02_plausibility_geometry.py measures directly), or some property of the
cross-validation itself that biases a balanced binary target downwards. This
script separates them.

For each selected run the plausibility (or physical-concept) labels are shuffled
across the 48 videos and the entire pipeline -- fold construction, standardising,
PCA, inner C selection, fitting -- is repeated on the shuffled labels. If the
procedure were biased, the null would sit below chance too. If the null sits at
chance and only the real labels fall below it, the effect is in the data.

The same null is run for physical concept as a positive comparison: there the
observed value should sit far above its null.

Outputs derived/permutation_control.csv and derived/permutation_control.json

Usage:  python 03_permutation_control.py [--perms N] [--jobs N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import paths as P  # noqa: E402
MAT = P.MAT
META = P.META
DERIVED = P.derived_dir(__file__)

from model_naming import STAGE_LABEL, STAGE_ORDER, label as model_label  # noqa: E402

# the re-run module's name begins with a digit, so load it by path
_spec = importlib.util.spec_from_file_location(
    "rerun", HERE / "01_rerun_probes_grouped_cv.py")
rerun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rerun)

# one representative condition per model: the prompt cannot reach the visual
# stages anyway, and the language-model conditions differ by ~2 points
CONDITION = ("non_embodied", "simple", "category")
TARGETS = ["plausibility", "physical_concept"]


def one(model_slug: str, stage: str, target: str, n_perm: int) -> dict:
    setting, prompt, task = CONDITION
    slug = f"{model_slug}__{setting}__{prompt}__{task}__{stage}"
    X = np.load(MAT / f"{slug}.npy").astype(np.float64)
    y = pd.read_csv(META / f"{slug}.csv")[target].to_numpy()

    observed = rerun.cv_accuracy(X, y, None, 0)
    rng = np.random.default_rng(0)
    null = np.array([rerun.cv_accuracy(X, rng.permutation(y), None, 0)
                     for _ in range(n_perm)])
    # +1 corrections keep the p-values from ever being exactly 0
    return {
        "model": model_label(*rerun.SLUG_MODEL[model_slug]),
        "stage_key": stage, "stage": STAGE_LABEL[stage], "target": target,
        "chance": rerun.CHANCE[target], "observed": observed,
        "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
        "null_p02": float(np.percentile(null, 2.5)),
        "null_p97": float(np.percentile(null, 97.5)),
        "p_below": float(((null <= observed).sum() + 1) / (n_perm + 1)),
        "p_above": float(((null >= observed).sum() + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    jobs = [delayed(one)(m, st, t, args.perms)
            for m in rerun.SLUG_MODEL for st in STAGE_ORDER for t in TARGETS]
    print(f"{len(jobs)} cells x {args.perms} permutations")
    res = pd.DataFrame(Parallel(n_jobs=args.jobs, verbose=5)(jobs))
    res.to_csv(DERIVED / "permutation_control.csv", index=False)

    stats = {"n_perm": args.perms, "condition": "/".join(CONDITION), "by_target": {}}
    for t in TARGETS:
        s = res[res.target == t]
        stats["by_target"][t] = {"chance": rerun.CHANCE[t], "by_stage": {}}
        for st in STAGE_ORDER:
            v = s[s.stage_key == st]
            stats["by_target"][t]["by_stage"][STAGE_LABEL[st]] = {
                "observed_mean": round(float(v.observed.mean()), 6),
                "null_mean": round(float(v.null_mean.mean()), 6),
                "null_sd": round(float(v.null_sd.mean()), 6),
                "n_models": int(len(v)),
                "n_observed_below_null_p025": int((v.p_below <= 0.025).sum()),
                "n_observed_above_null_p975": int((v.p_above <= 0.025).sum()),
            }
    with open(DERIVED / "permutation_control.json", "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 220)
    print("\nobserved vs label-permuted null (%), mean over the 11 model variants:")
    for t in TARGETS:
        print(f"\n  {t}  (chance {rerun.CHANCE[t]*100:.1f}%)")
        for st in STAGE_ORDER:
            d = stats["by_target"][t]["by_stage"][STAGE_LABEL[st]]
            print(f"    {STAGE_LABEL[st]:26s} observed {d['observed_mean']*100:5.1f}%"
                  f"   null {d['null_mean']*100:5.1f}% +/- {d['null_sd']*100:4.1f}"
                  f"   below null in {d['n_observed_below_null_p025']}/11"
                  f"   above null in {d['n_observed_above_null_p975']}/11")
    print(f"\nwrote {DERIVED/'permutation_control.csv'}")


if __name__ == "__main__":
    main()
