#!/usr/bin/env python3
"""
What does each principal component of each model representation encode?

The manuscript characterises PC1 and PC2 by eyeballing the two highest-scoring
videos. That is unstable: with 48 videos, the top two can share a property by
chance, and it ignores the other 46. This script instead asks, for every PC,
how much of the variance in PC score across all 48 videos each stimulus factor
explains (eta squared from a one-way ANOVA of PC score by that factor). The
factor with the largest eta squared is the PC's dominant axis, and the contrast
is read off as the highest- minus lowest-scoring level of that factor.

Stimulus factors, and one design fact that constrains their interpretation:
    physical concept   continuity / permanence / immutability   (16 videos each)
    plausibility       plausible / implausible                  (24 each)
    scene template     Box / MovingAroundOccluder / RotatingCup / HotAirBallon
                                                                (12 each)
    camera             Moving / Fixed                           (24 each)
Camera is perfectly confounded with scene template -- Box and
MovingAroundOccluder are always moving-camera, RotatingCup and HotAirBallon
always fixed -- so a "camera" axis and a "scene template" axis cannot be
separated in this stimulus set. Scene template is therefore reported as the
4-level factor and camera is flagged, never claimed independently.

Outputs
    derived/pca_stats.json              everything the text and figure quote
    derived/table_pc_meaning.csv        one row per stage x model x PC
    derived/table_pc_meaning_full.csv   the same, for all 12 task x prompt cells
    derived/video_metadata.csv          the 48-video design

Usage:  python 00_extract_pca_stats.py
"""

from __future__ import annotations

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
    MODELS,
    STAGE_LABEL,
    STAGE_ORDER,
    TASK_LABEL,
    TASK_ORDER,
    label as model_label,
    order_key as model_order_key,
)

FACTORS = {
    "physical_concept": "physical concept",
    "plausibility": "plausibility",
    "scene_template": "scene template",
}
PC_COLS = {"PC1": "principal_component_1_score",
           "PC2": "principal_component_2_score"}
PROMPT_ORDER = ["Non-embodied | Simple", "Non-embodied | Detailed",
                "Embodied | Simple", "Embodied | Detailed"]
# The prompt condition already encodes the embodiment setting, so setting is
# never carried as a separate column (revision comment).
SLUG_TASK = {"Category": "category", "VoE": "voe", "SM": "sm"}
SLUG_SETTING = {"Non-embodied": "non_embodied", "Embodied": "embodied"}


def eta_sq(y: np.ndarray, g: np.ndarray) -> float:
    """Share of variance in y explained by group membership g."""
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_between = sum(len(y[g == lv]) * (y[g == lv].mean() - grand) ** 2
                     for lv in np.unique(g))
    return float(ss_between / ss_tot)


def contrast(y: np.ndarray, g: np.ndarray) -> tuple[str, str]:
    """Highest- and lowest-scoring level of the factor."""
    means = {lv: y[g == lv].mean() for lv in np.unique(g)}
    hi = max(means, key=means.get)
    lo = min(means, key=means.get)
    return hi, lo


def main() -> None:
    meta = (pd.read_csv(
        P.PCA_DIR / "06_pc_top_bottom_videos_top3_centered_readable.csv")
        .drop_duplicates("video_id")
        [["video_id", "physical_concept", "plausibility", "scene_template",
          "camera_condition", "occlusion_configuration"]]
        .set_index("video_id"))
    assert len(meta) == 48
    meta.to_csv(DERIVED / "video_metadata.csv")

    # camera is nested inside scene template in this stimulus set
    cam = pd.crosstab(meta.scene_template, meta.camera_condition)
    assert (cam == 0).sum().sum() == len(cam) * 2 - len(cam), \
        "camera is no longer perfectly nested in scene template"

    run = pd.read_csv(
        P.PCA_DIR / "04_pca_run_summary_centered_main_analysis_readable.csv")

    rows = []
    for _, r in run.iterrows():
        scores = pd.read_csv(
            P.PCA_RESULTS / "centered" / "scores" / f"{r.group_slug}.csv"
        ).set_index("video_id")
        m = meta.loc[scores.index]
        variant = model_label(r.model_name, r.model_mode)
        for pc, col in PC_COLS.items():
            y = scores[col].to_numpy(float)
            etas = {k: eta_sq(y, m[k].to_numpy()) for k in FACTORS}
            dom = max(etas, key=etas.get)
            hi, lo = contrast(y, m[dom].to_numpy())
            rows.append({
                "stage": r.embedding_layer, "stage_key": r.group_slug.split("__")[-1],
                "model": variant, "model_order": model_order_key(r.model_name,
                                                                 r.model_mode),
                "task_raw": r.task_type,
                "prompt": r.condition_label,
                "pc": pc,
                "explained_variance": (r.pc1_explained_variance_ratio if pc == "PC1"
                                       else r.pc2_explained_variance_ratio),
                "dominant_factor": FACTORS[dom],
                "contrast_high": hi, "contrast_low": lo,
                **{f"eta2_{k}": v for k, v in etas.items()},
                "eta2_dominant": etas[dom],
                "top_video": scores[col].idxmax(),
                "group_slug": r.group_slug,
            })
    full = pd.DataFrame(rows)
    # the package labels tasks with the manuscript's own display names already
    full["task"] = full.task_raw.map(
        {"Categorization": "Concept Verification", "VoE": "Plausibility Assessment",
         "Sensorimotor": "Affordance Recognition"}).fillna(full.task_raw)
    full.to_csv(DERIVED / "table_pc_meaning_full.csv", index=False)

    # ------------------------------------------------ visual-stage invariance
    inv = {}
    for stage in ["vision_encoder_last", "vision_projection"]:
        s = full[full.stage_key == stage]
        n = s.groupby(["model", "pc"])[
            ["dominant_factor", "contrast_high", "contrast_low", "top_video"]
        ].nunique()
        inv[STAGE_LABEL[stage]] = {
            "cells_per_model_pc": int(s.groupby(["model", "pc"]).size().iat[0]),
            "all_invariant": bool((n == 1).all().all()),
        }
    lm = full[full.stage_key == "language_model_last"]
    n_lm = lm.groupby(["model", "pc"])["top_video"].nunique()
    inv[STAGE_LABEL["language_model_last"]] = {
        "cells_per_model_pc": int(lm.groupby(["model", "pc"]).size().iat[0]),
        "all_invariant": False,
        "median_distinct_top_videos": float(n_lm.median()),
    }

    # ---------------------------------------------- the reported summary table
    # Visual stages are prompt- and task-invariant, so one row represents all 12
    # cells. The language model is not, so the non-embodied simple condition is
    # reported as the baseline and the variability is quantified alongside.
    keep = full[(full.prompt == PROMPT_ORDER[0])
                & (full.task == "Concept Verification")].copy()
    keep = keep.sort_values(["stage_key", "model_order", "pc"])
    summary = keep[["stage", "model", "pc", "explained_variance",
                    "dominant_factor", "contrast_high", "contrast_low",
                    "eta2_dominant", "eta2_physical_concept",
                    "eta2_plausibility", "eta2_scene_template",
                    "top_video", "group_slug", "model_order", "stage_key"]]
    summary.to_csv(DERIVED / "table_pc_meaning.csv", index=False)

    # ------------------------------------------------------------- aggregates
    stats: dict = {"design": {}, "stage_invariance": inv}
    stats["design"] = {
        "n_videos": 48, "n_model_variants": len(MODELS),
        "n_prompt_conditions": len(PROMPT_ORDER), "n_tasks": len(TASK_ORDER),
        "n_stages": len(STAGE_ORDER),
        "n_pca_runs": int(len(run)),
        "n_task_prompt_cells": len(TASK_ORDER) * len(PROMPT_ORDER),
    }
    stats["explained_variance"] = {}
    for stage in STAGE_ORDER:
        s = run[run.embedding_layer == STAGE_LABEL[stage]] if False else \
            run[run.group_slug.str.endswith(stage)]
        stats["explained_variance"][STAGE_LABEL[stage]] = {
            "pc1_mean": float(s.pc1_explained_variance_ratio.mean()),
            "pc2_mean": float(s.pc2_explained_variance_ratio.mean()),
            "top3_mean": float(s.top3_cumulative_explained_variance_ratio.mean()),
            "k90_mean": float(s.principal_components_needed_for_90_percent_variance
                              .mean()),
        }

    stats["dominant_factor_counts"] = {}
    stats["mean_eta2"] = {}
    for stage in STAGE_ORDER:
        s = full[full.stage_key == stage]
        # one row per model x PC x task x prompt; for the visual stages these are
        # replicates, so count over the invariant representative instead
        rep = s[(s.prompt == PROMPT_ORDER[0]) & (s.task == "Concept Verification")] \
            if stage != "language_model_last" else s
        stats["dominant_factor_counts"][STAGE_LABEL[stage]] = {
            "n_components_all_cells": int(len(s)),
            **{k: int((s.dominant_factor == k).sum()) for k in FACTORS.values()},
            "representative_cell": {
                "n_components": int(len(rep)),
                **{k: int((rep.dominant_factor == k).sum())
                   for k in FACTORS.values()},
            },
        }
        stats["mean_eta2"][STAGE_LABEL[stage]] = {
            FACTORS[k]: float(rep[f"eta2_{k}"].mean()) for k in FACTORS}

    with open(DERIVED / "pca_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 240)
    print("PCA runs:", len(run), " rows in full table:", len(full))
    print("\nvisual-stage invariance:", json.dumps(inv, indent=1))
    print("\ndominant factor per PC, by stage (representative cell):")
    print(json.dumps(stats["dominant_factor_counts"], indent=1))
    print("\nmean eta^2 by stage:")
    print(pd.DataFrame(stats["mean_eta2"]).round(3).to_string())
    print("\nexplained variance:")
    print(pd.DataFrame(stats["explained_variance"]).round(3).to_string())
    print(f"\nwrote {DERIVED/'pca_stats.json'} and 3 tables")


if __name__ == "__main__":
    main()
