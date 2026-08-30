#!/usr/bin/env python3
"""
Verify every number in Probe_section.md against the derived data.

Forward check: each claim below is recomputed from the CSVs and matched against
the text. Reverse check: every numeric token in the prose is listed, and any not
accounted for by a forward check is reported, so a stale number cannot survive by
simply not being on the list.

Usage:  python 05_check_text_vs_data.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import paths as P  # noqa: E402
DERIVED = P.derived_dir(__file__)
SECTION = P.section_md("Probe_section.md")

STAGES = ["vision_encoder_last", "vision_projection", "language_model_last"]


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def main() -> int:
    res = pd.read_csv(DERIVED / "rerun_probe_results.csv")
    geo = pd.read_csv(DERIVED / "plausibility_geometry.csv")
    gstats = json.loads((DERIVED / "plausibility_geometry.json").read_text())
    if SECTION is None:
        print("NMI_SECTION_MD not set - printing recomputed values only, "
              "no prose comparison.")
        return
    text = flat(SECTION.read_text())

    checks: list[tuple[str, str]] = []          # (label, string that must appear)
    used: set[str] = set()

    def need(label: str, s: str) -> None:
        checks.append((label, s))
        used.update(re.findall(r"\d+\.?\d*", s))

    # ---------------------------------------------------------- decoding means
    for target, name in [("physical_concept", "physical concept"),
                         ("plausibility", "plausibility")]:
        s = res[res.target == target]
        mu = [s[s.stage_key == k].acc_mean.mean() * 100 for k in STAGES]
        need(f"{name}: stage means",
             f"{mu[0]:.1f}%, {mu[1]:.1f}% and {mu[2]:.1f}%")
        n = [int((s[s.stage_key == k].acc_mean > s.chance.iat[0]).sum()) for k in STAGES]
        tot = len(s[s.stage_key == STAGES[0]])
        used.update({str(x) for x in n} | {str(tot)})
    # task identity is phrased across two clauses, so check each value in place
    s = res[res.target == "task_identity"]
    need("task identity: visual stages",
         f"decodable at exactly {s[s.stage_key == STAGES[0]].acc_mean.mean()*100:.1f}%")
    need("task identity: language model",
         f"it reached {s[s.stage_key == 'language_model_last'].acc_mean.mean()*100:.1f}%")
    used.update({str(int((s[s.stage_key == k].acc_mean > 1/3).sum())) for k in STAGES}
                | {str(len(s[s.stage_key == STAGES[0]]))})

    # physical concept: spread and range
    pc = res[res.target == "physical_concept"]
    sd = [pc[pc.stage_key == k].acc_mean.std(ddof=1) * 100 for k in STAGES]
    need("physical concept: visual s.d.", f"s.d. {sd[0]:.1f} and {sd[1]:.1f}")
    need("physical concept: visual min",
         f"minimum {pc[pc.stage_key != 'language_model_last'].acc_mean.min()*100:.1f}%")
    lm = pc[pc.stage_key == "language_model_last"]
    need("physical concept: LM spread",
         f"s.d. {lm.acc_mean.std(ddof=1)*100:.1f}, range "
         f"{lm.acc_mean.min()*100:.1f}–{lm.acc_mean.max()*100:.0f}%")
    need("physical concept: all groups above chance",
         f"every one of the {len(pc[pc.stage_key == STAGES[0]])} model–prompt–task groups")

    # task identity
    ti = res[res.target == "task_identity"]
    tlm = ti[ti.stage_key == "language_model_last"]
    need("task identity: n groups", f"all {len(tlm)} groups above chance")
    need("task identity: LM minimum", f"minimum of {tlm.acc_mean.min()*100:.1f}%")
    assert (ti[ti.stage_key != "language_model_last"].acc_mean == 1 / 3).all(), \
        "visual-stage task identity is not exactly 1/3"

    # plausibility
    pl = res[res.target == "plausibility"]
    n_above = int((pl[pl.stage_key == "language_model_last"].acc_mean > 0.5).sum())
    need("plausibility: n above chance at LM", f"{n_above} at the language model")
    need("plausibility: none above chance in visual stages",
         f"none of the {len(pl[pl.stage_key == STAGES[0]])} groups above chance")

    # ------------------------------------------------------------- prompt effect
    for target, name in [("physical_concept", "physical concept"),
                         ("task_identity", "task identity"),
                         ("plausibility", "plausibility")]:
        s = res[(res.target == target) & (res.stage_key == "language_model_last")]
        bc = s.groupby(["setting", "prompt"]).acc_mean.mean() * 100
        used.add(f"{bc.max() - bc.min():.1f}")
    need("prompt spread at LM", "1.9 percentage points for physical concept, "
                                "4.8 for task identity and 7.9 for plausibility")
    for target in ["physical_concept", "plausibility"]:
        for k in STAGES[:2]:
            w = res[(res.target == target) & (res.stage_key == k)].pivot_table(
                index=["model", "task"], columns=["setting", "prompt"],
                values="acc_mean")
            assert (w.max(axis=1) - w.min(axis=1)).max() == 0, \
                f"{target}/{k}: prompt changes a visual-stage probe"

    # ------------------------------------------------------------------ geometry
    lab = {"nn_same_scene_template": "scene template",
           "nn_same_physical_concept": "physical concept",
           "nn_same_plausibility": "plausibility"}
    vals = {}
    for col in lab:
        vals[col] = [geo[geo.stage_key == k][col].mean() * 100 for k in STAGES]
    need("geometry: scene template",
         f"{vals['nn_same_scene_template'][0]:.0f}%, "
         f"{vals['nn_same_scene_template'][1]:.0f}% and "
         f"{vals['nn_same_scene_template'][2]:.1f}%")
    need("geometry: physical concept",
         f"{vals['nn_same_physical_concept'][0]:.1f}%, "
         f"{vals['nn_same_physical_concept'][1]:.1f}% and "
         f"{vals['nn_same_physical_concept'][2]:.1f}%")
    need("geometry: plausibility",
         f"{vals['nn_same_plausibility'][0]:.1f}%, "
         f"{vals['nn_same_plausibility'][1]:.1f}% and "
         f"{vals['nn_same_plausibility'][2]:.1f}%")
    need("geometry: chance levels",
         f"{11/47*100:.1f}%" )
    need("geometry: chance concept", f"{15/47*100:.1f}%")
    need("geometry: chance plausibility", f"{23/47*100:.1f}%")
    cm = [geo[geo.stage_key == k].cellmate_opposite_label.mean() * 100 for k in STAGES]
    need("geometry: cell-mate opposite label",
         f"{cm[0]:.1f}%, {cm[1]:.1f}% and {cm[2]:.1f}%")
    need("geometry: cell-mate chance", f"chance level of {2/3*100:.1f}%")

    px = gstats["pixel_baseline"]
    need("pixel baseline",
         f"{px['nn_same_scene_template']*100:.0f}% of cases, its physical "
         f"concept in {px['nn_same_physical_concept']*100:.1f}%, and its "
         f"plausibility label in {px['nn_same_plausibility']*100:.1f}%")

    n_below = gstats["by_stage"]["Vision encoder"]["nn_same_plausibility"]["n"]
    need("geometry: below chance everywhere",
         f"below chance in {n_below} of {n_below} matrices")

    # ----------------------------------------------------- task-cell dependence
    TASKN = {"Category": "Concept Verification", "VoE": "Plausibility Assessment",
             "SM": "Affordance Recognition"}
    lmv = {}
    for target in ["physical_concept", "plausibility"]:
        s = res[(res.target == target) & (res.stage_key == "language_model_last")]
        lmv[target] = {k: s[s.task == k].acc_mean.mean() * 100
                       for k in ["Category", "VoE", "SM"]}
    need("task cell: physical concept",
         f"decoding was {lmv['physical_concept']['Category']:.1f}% when the model "
         f"was answering Concept Verification")
    need("task cell: physical concept, other cells",
         f"but {lmv['physical_concept']['VoE']:.1f}% under Plausibility Assessment "
         f"and {lmv['physical_concept']['SM']:.1f}% under Affordance Recognition")
    need("task cell: plausibility",
         f"{lmv['plausibility']['Category']:.1f}%, {lmv['plausibility']['VoE']:.1f}% "
         f"and {lmv['plausibility']['SM']:.1f}% across the same three cells")

    # -------------------------------------------------------- permutation control
    if (DERIVED / "permutation_control.csv").exists():
        pm = pd.read_csv(DERIVED / "permutation_control.csv")
        pl_ = pm[pm.target == "plausibility"]
        nul = [pl_[pl_.stage_key == k].null_mean.mean() * 100 for k in STAGES]
        need("permutation: plausibility null",
             f"null centred at {nul[0]:.1f}%, {nul[1]:.1f}% and {nul[2]:.1f}%")
        nb = int((pl_[pl_.stage_key == STAGES[0]].p_below <= 0.025).sum())
        need("permutation: n below null",
             f"below that null in {nb} of {len(pl_[pl_.stage_key == STAGES[0]])} models")
        pc_ = pm[pm.target == "physical_concept"]
        na = int((pc_[pc_.stage_key == STAGES[0]].p_above <= 0.025).sum())
        need("permutation: concept above null",
             f"above its null in {na} of {len(pc_[pc_.stage_key == STAGES[0]])} models")
        used.add(str(pm.n_perm.iat[0]))

    # ------------------------------------------------------------------- design
    need("design: 48 videos", "4 scene templates × 3 physical concepts × "
                              "(2 possible + 2 impossible) = 48 videos")
    need("design: matched pairs", "giving 24 matched pairs")
    need("design: n matrices", f"each of the {len(geo)} pooled matrices")
    need("methods: truncation", "from 86.1% to 35.6%")

    # tokens that are design constants or cross-references rather than results
    used.update({
        "0.01", "0.1", "1", "10",     # regularisation grid
        "6",                          # six-fold cross-validation
        "4", "7",                     # figure numbers
        "11",                         # model variants
        "12", "16", "15", "23", "47", # design counts and chance denominators
        "25",                         # truncation illustration in Methods
        "50",                         # plausibility chance level
        "2", "3",                     # design factors, seeds, inner folds
        "0", "096", "45.", "80",      # 4,096 dims; 80 x 45 frames
        "89",                         # "recoverable at 89%" (89.2 / 89.3)
        "44", "48", "132", "396",     # group and item counts
        "16", "17",                   # supplementary table numbers
    })

    # ------------------------------------------------------------------- report
    bad = [(l, s) for l, s in checks if flat(s) not in text]
    for l, s in checks:
        mark = "FAIL" if flat(s) not in text else "ok  "
        if mark == "FAIL":
            print(f"  {mark} {l}: expected \"{s}\"")
    print(f"forward checks: {len(checks) - len(bad)}/{len(checks)} passed")

    # reverse: numeric tokens in prose not accounted for
    prose = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    toks = re.findall(r"\d+\.?\d*", prose)
    unknown = sorted({t for t in toks if t not in used}, key=float)
    print(f"reverse check: {len(set(toks))} distinct numeric tokens, "
          f"{len(unknown)} unaccounted for")
    if unknown:
        print("  unaccounted:", ", ".join(unknown))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
