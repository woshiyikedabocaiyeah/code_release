#!/usr/bin/env python3
"""
Why plausibility decodes below chance: the geometry of the stimulus set.

A linear probe with no information scores at chance, not far below it. The
grouped-CV re-run returns plausibility accuracies well under 50% at every
stage, which is a statement about the arrangement of the embeddings, not about
missing information, and this script measures that arrangement directly.

The design is fully crossed: 4 scene templates x 3 physical concepts x
(2 plausible + 2 implausible) = 48 videos. Within one (scene, concept) cell the
four videos depict the same event; two of them contain a physical violation and
two do not. So for every video there is a same-cell counterpart carrying the
opposite plausibility label and differing only in the violated moment.

Two nearest-neighbour statistics follow, with their exact chance levels under a
random arrangement of the 48 videos:

  * whether a video's nearest neighbour shares its scene template (11/47),
    physical concept (15/47) or plausibility label (23/47);
  * within a video's own (scene, concept) cell -- which holds one same-label and
    two opposite-label mates -- whether the closest of the three carries the
    opposite label (2/3);
  * whether a video's nearest neighbour among all 47 others is its own matched
    counterpart (1/47), the one video that depicts the same event and differs
    only in whether the physical violation occurs.

possibility_label_from_dataset names the pairing explicitly: each cell contains
1_Possible, 1_Impossible, 2_Possible and 2_Impossible, giving 24 matched pairs.

Outputs derived/plausibility_geometry.csv and derived/plausibility_geometry.json

Usage:  python 02_plausibility_geometry.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import paths as P  # noqa: E402
MAT = P.MAT
META = P.META
DERIVED = P.derived_dir(__file__)
VIDEOS = P.VIDEOS

from model_naming import STAGE_LABEL, STAGE_ORDER, label as model_label  # noqa: E402

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

# exact chance levels for the 48-video design, computed once below
CHANCE_NN = {"scene_template": 11 / 47, "physical_concept": 15 / 47,
             "plausibility": 23 / 47}
CHANCE_CELLMATE = 2 / 3
CHANCE_COUNTERPART = 1 / 47        # nearest neighbour is the matched counterpart
CHANCE_CLOSEST_CELLMATE = 1 / 3    # counterpart is the closest of the 3 cell-mates


def parse_slug(slug: str):
    for m in sorted(SLUG_MODEL, key=len, reverse=True):
        if slug.startswith(m + "__"):
            setting, prompt, task, stage = slug[len(m) + 2:].split("__", 3)
            return m, setting, prompt, task, stage
    raise ValueError(slug)


def add_pair_id(md: pd.DataFrame) -> pd.DataFrame:
    """Each (scene, concept) cell holds two matched pairs, indexed 1 and 2."""
    md = md.copy()
    md["pair_id"] = (md.scene_template + "/" + md.physical_concept + "/"
                     + md.possibility_label_from_dataset.str.split("_").str[0])
    return md


def verify_design(md: pd.DataFrame) -> None:
    """The chance levels above are only correct for the fully crossed design."""
    assert len(md) == 48 and md.video_id.nunique() == 48
    assert md.scene_template.value_counts().eq(12).all()
    assert md.physical_concept.value_counts().eq(16).all()
    assert md.plausibility.value_counts().eq(24).all()
    cells = md.groupby(["scene_template", "physical_concept"])
    assert cells.size().eq(4).all(), "cells are not 4 videos each"
    for _, c in cells:
        assert c.plausibility.value_counts().eq(2).all(), "cell is not 2 + 2"
        assert set(c.possibility_label_from_dataset) == {
            "1_Possible", "1_Impossible", "2_Possible", "2_Impossible"}
    assert md.pair_id.nunique() == 24
    assert md.groupby("pair_id").plausibility.nunique().eq(2).all()


def geometry(X: np.ndarray, md: pd.DataFrame) -> dict:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(1)

    out = {f"nn_same_{c}": float((md[c].to_numpy() == md[c].to_numpy()[nn]).mean())
           for c in ("scene_template", "physical_concept", "plausibility")}

    cell = (md.scene_template + "|" + md.physical_concept).to_numpy()
    pair = md.pair_id.to_numpy()
    pl = md.plausibility.to_numpy()
    idx = np.arange(48)

    # the matched counterpart: same event, opposite plausibility
    mate_of = np.array([idx[(pair == pair[i]) & (idx != i)][0] for i in range(48)])
    out["nn_is_matched_counterpart"] = float((nn == mate_of).mean())
    # rank of the counterpart among the other 47 videos (1 = nearest)
    order = np.argsort(-S, axis=1)
    rank = np.array([int(np.where(order[i] == mate_of[i])[0][0]) + 1 for i in range(48)])
    out["counterpart_rank_mean"] = float(rank.mean())

    opp, closest_is_mate, same_sim, opp_sim = 0, 0, [], []
    for i in range(48):
        mates = idx[(cell == cell[i]) & (idx != i)]
        j = mates[S[i, mates].argmax()]
        opp += int(pl[j] != pl[i])
        closest_is_mate += int(j == mate_of[i])
        same_sim.append(S[i, mates[pl[mates] == pl[i]]].mean())
        opp_sim.append(S[i, mates[pl[mates] != pl[i]]].mean())
    out["cellmate_opposite_label"] = opp / 48
    out["closest_cellmate_is_counterpart"] = closest_is_mate / 48
    # positive = the violated counterpart sits closer than the same-label mate
    out["cell_sim_opposite_minus_same"] = float(np.mean(opp_sim) - np.mean(same_sim))
    return out


def pixel_baseline(md: pd.DataFrame, k: int = 16) -> dict | None:
    """The same geometry computed on the videos themselves.

    Mean-pooled greyscale frames (plus their per-pixel s.d.) stand in for a
    model with no learned representation at all. It is the reference the visual
    stages have to be compared against: a stage that merely preserves the pixel
    arrangement is not organising anything.
    """
    try:
        import cv2
    except ImportError:
        return None
    if not VIDEOS.is_dir():
        return None

    rows = []
    for vid in md.video_id:
        cap = cv2.VideoCapture(str(VIDEOS / f"{vid}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fs = []
        for i in range(k):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (i + 0.5) / k))
            ok, im = cap.read()
            if ok:
                fs.append(cv2.cvtColor(cv2.resize(im, (80, 45)),
                                       cv2.COLOR_BGR2GRAY).astype(np.float64).ravel())
        cap.release()
        rows.append(np.concatenate([np.mean(fs, 0), np.std(fs, 0)]))
    return geometry(np.array(rows), md)


def main() -> None:
    slugs = sorted(p.stem for p in MAT.glob("*.npy"))
    assert len(slugs) == 396, len(slugs)
    verify_design(add_pair_id(pd.read_csv(META / f"{slugs[0]}.csv")))

    rows = []
    for slug in slugs:
        m, setting, prompt, task, stage = parse_slug(slug)
        md = add_pair_id(pd.read_csv(META / f"{slug}.csv"))
        X = np.load(MAT / f"{slug}.npy").astype(np.float64)
        rows.append({"model": model_label(*SLUG_MODEL[m]), "setting": setting,
                     "prompt": prompt, "task": TASK_SLUG[task], "stage_key": stage,
                     "stage": STAGE_LABEL[stage], **geometry(X, md)})
    g = pd.DataFrame(rows)
    g.to_csv(DERIVED / "plausibility_geometry.csv", index=False)

    stats: dict = {"chance": {**CHANCE_NN, "cellmate_opposite_label": CHANCE_CELLMATE},
                   "n_matrices": len(g), "by_stage": {}}
    for st in STAGE_ORDER:
        s = g[g.stage_key == st]
        d = {}
        for c, ch in [("nn_same_scene_template", CHANCE_NN["scene_template"]),
                      ("nn_same_physical_concept", CHANCE_NN["physical_concept"]),
                      ("nn_same_plausibility", CHANCE_NN["plausibility"]),
                      ("cellmate_opposite_label", CHANCE_CELLMATE),
                      ("nn_is_matched_counterpart", CHANCE_COUNTERPART),
                      ("closest_cellmate_is_counterpart", CHANCE_CLOSEST_CELLMATE)]:
            v = s[c].to_numpy()
            d[c] = {"mean": round(float(v.mean()), 6),
                    "sd": round(float(v.std(ddof=1)), 6),
                    "min": round(float(v.min()), 6), "max": round(float(v.max()), 6),
                    "chance": round(float(ch), 6),
                    "n_above_chance": int((v > ch).sum()), "n": int(len(v)),
                    "p_wilcoxon": float(wilcoxon(v - ch).pvalue)
                    if np.ptp(v - ch) > 0 else float("nan")}
        d["counterpart_rank_mean"] = {
            "mean": round(float(s.counterpart_rank_mean.mean()), 6),
            "chance": 24.0, "n": int(len(s))}
        d["cell_sim_opposite_minus_same"] = {
            "mean": round(float(s.cell_sim_opposite_minus_same.mean()), 6),
            "n_positive": int((s.cell_sim_opposite_minus_same > 0).sum()),
            "n": int(len(s))}
        stats["by_stage"][STAGE_LABEL[st]] = d

    base = pixel_baseline(add_pair_id(pd.read_csv(META / f"{slugs[0]}.csv")))
    stats["pixel_baseline"] = base

    with open(DERIVED / "plausibility_geometry.json", "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 220)
    print("nearest-neighbour structure, mean over matrices at each stage (%):")
    cols = ["nn_same_scene_template", "nn_same_physical_concept",
            "nn_same_plausibility", "cellmate_opposite_label",
            "nn_is_matched_counterpart", "closest_cellmate_is_counterpart"]
    tbl = (g.groupby("stage")[cols].mean() * 100).loc[
        [STAGE_LABEL[s] for s in STAGE_ORDER]].round(1)
    if base is not None:
        tbl.loc["raw video pixels"] = [base[c] * 100 for c in cols]
    tbl.loc["chance"] = [CHANCE_NN["scene_template"] * 100,
                         CHANCE_NN["physical_concept"] * 100,
                         CHANCE_NN["plausibility"] * 100, CHANCE_CELLMATE * 100,
                         CHANCE_COUNTERPART * 100, CHANCE_CLOSEST_CELLMATE * 100]
    print(tbl.round(1).to_string())
    print("\nmean rank of the matched counterpart among the other 47 videos "
          "(1 = nearest, 24 = chance):")
    for st in STAGE_ORDER:
        print(f"  {STAGE_LABEL[st]:26s} "
              f"{stats['by_stage'][STAGE_LABEL[st]]['counterpart_rank_mean']['mean']:.2f}")
    print("\nmatrices above chance, out of 132 per stage:")
    for st in STAGE_ORDER:
        d = stats["by_stage"][STAGE_LABEL[st]]
        print(f"  {STAGE_LABEL[st]:24s} " + "  ".join(
            f"{c.replace('nn_same_','')}: {d[c]['n_above_chance']}/{d[c]['n']}"
            for c in cols))
    print(f"\nwrote {DERIVED/'plausibility_geometry.csv'}")


if __name__ == "__main__":
    main()
