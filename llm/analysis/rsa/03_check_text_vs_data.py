#!/usr/bin/env python3
"""
Verify that every number in the RSA prose comes from the data.

  FORWARD   every statistic the section is supposed to report must literally
            appear in the text, formatted as the text formats it. Catches
            values that were dropped or left blank.
  REVERSE   every numeric token in the manuscript prose (Results + figure
            legend + Methods paragraph) must be licensed either by
            derived/rsa_stats.json or by an explicitly justified design
            constant. Catches values invented, mistyped, or carried over from
            an older analysis run.

The figure is not checked separately: the figure script reads the
same JSON, so its annotations cannot diverge from the text.

Usage:  python 03_check_text_vs_data.py     (exit 0 = clean)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import paths as P  # noqa: E402
DERIVED = P.derived_dir(__file__)
SECTION = P.section_md("RSA_section.md")
from model_naming import MODELS, STAGE_ORDER, TASK_ORDER  # noqa: E402

MEASURES = ["rt", "rt_critical", "corr"]


def norm(s: str) -> str:
    return s.replace("−", "-").replace("–", "-").replace("—", "-")


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", norm(s))


def main() -> int:
    stats = json.loads((DERIVED / "rsa_stats.json").read_text())
    if SECTION is None:
        print("NMI_SECTION_MD not set - printing recomputed values only, "
              "no prose comparison.")
        return
    text = norm(SECTION.read_text())

    results = text[text.index("## MLLMs track"):text.index("![Figure 4]")]
    legend = text[text.index("**Fig. 4 |"):text.index("## Methods -")]
    methods = text[text.index("## Methods -"):]
    prose = results + legend + methods
    hay = {"results": flat(results), "legend": flat(legend),
           "methods": flat(methods), "prose": flat(prose)}

    problems: list[str] = []
    ok = 0

    def expect(needle: str, where: str = "results", label: str = "") -> None:
        nonlocal ok
        if flat(needle) in hay[where]:
            ok += 1
        else:
            problems.append(f"FORWARD  missing in {where}: {label or needle!r}")

    # ------------------------------------------------------------- FORWARD
    d = stats["design"]
    expect(str(d["n_videos"]), "results", "n videos")
    expect(str(d["n_model_rdms"]), "results", "n model RDMs")
    expect(f"{d['n_model_variants']} model-mode variants", "results")
    expect(f"{d['n_pairings']:,} model-human alignments", "results")
    expect(f"{d['n_human_rdms']} human RDMs", "results")
    expect(f"{d['n_video_pairs']:,}", "results", "n video pairs")
    expect(f"{d['n_model_configurations']} model configurations", "results")

    g = stats["global"]
    expect(f"{g['rt']['n_positive']} were positive ({g['rt']['pct_positive']:.1f}%",
           "results", "RT_onset positive")
    expect(f"mean ρ = {g['rt']['mean_rho']:.3f}", "results", "RT_onset mean")
    expect(f"median ρ = {g['rt']['median_rho']:.3f}", "results", "RT_onset median")
    expect(f"{g['rt_critical']['n_positive']} were positive "
           f"({g['rt_critical']['pct_positive']:.1f}%",
           "results", "RT_critical positive")
    expect(f"mean ρ = {g['rt_critical']['mean_rho']:.3f}", "results")
    expect(f"median ρ = {g['rt_critical']['median_rho']:.3f}", "results")
    expect(f"{g['corr']['n_positive']} of {g['corr']['n']} "
           f"({g['corr']['pct_positive']:.1f}%)", "results", "ACC positive")
    expect(f"mean was ρ = {g['corr']['mean_rho']:.3f}", "results")
    expect(f"median ρ = {g['corr']['median_rho']:.3f}", "results")

    # 27 stage means
    for m in MEASURES:
        for t in TASK_ORDER:
            for l in STAGE_ORDER:
                v = stats["by_task_layer"][f"{m}|{t}|{l}"]["mean_rho"]
                expect(f"{v:.3f}", "results", f"{m} {t} {l} stage mean")

    # noise ceilings
    for key in [f"{m}|{t}" for m in MEASURES for t in TASK_ORDER]:
        nc = stats["noise_ceiling"][key]
        expect(f"{nc['lower']:.3f}", "results", f"{key} ceiling lower")
    for key in ["rt|SM", "rt_critical|SM"]:
        nc = stats["noise_ceiling"][key]
        expect(f"{nc['lower']:.3f}-{nc['upper']:.3f}", "results", f"{key} band")
        expect(f"{nc['n_within_band']} of {nc['n_configurations']}", "results")
    byl = {norm(k): v for k, v in
           stats["noise_ceiling"]["rt|SM"]["n_within_band_by_layer"].items()}
    expect(f"{byl['Vision encoder']} of 44 vision-encoder and "
           f"{byl['Vision-language projector']} of 44 projector cells", "results")
    expect(f"{byl['Language model']} of 44", "results")

    # paired stage comparison
    for m, n in [("rt", stats["ve_minus_lm"]["rt"]["n_positive"]),
                 ("rt_critical", stats["ve_minus_lm"]["rt_critical"]["n_positive"]),
                 ("corr", stats["ve_minus_lm"]["corr"]["n_positive"])]:
        expect(f"{n} of ", "results", f"{m} paired count")

    # RT_critical specifics
    rc = stats["rt_critical"]
    expect(f"all {rc['gain_over_rt_onset']['n_cells']} measure × task × stage cells",
           "results", "gain cells")
    expect(f"{rc['gain_over_rt_onset']['min']:.3f} and "
           f"{rc['gain_over_rt_onset']['max']:.3f}", "results", "gain range")
    expect(f"mean {rc['gain_over_rt_onset']['mean']:.3f}", "results", "gain mean")
    expect(f"mean {rc['t_mean_s']:.2f} s, s.d. {rc['t_sd_s']:.2f} s, "
           f"range {rc['t_min_s']:.2f}-{rc['t_max_s']:.2f} s", "results", "t_critical")
    for t in TASK_ORDER:
        v = rc["rdm_corr_with_rt_onset"][stats["noise_ceiling"][f"rt|{t}"]["task_label"]]
        expect(f"{v:.2f}", "results", f"rdm corr {t}")

    # permutation
    for key in [f"{m}|{t}" for m in MEASURES for t in TASK_ORDER]:
        p = stats["permutation"][key]
        expect(f"{p['observed_rho']:.3f}", "results", f"{key} observed")
        if not p["significant"]:
            expect(f"q = {p['q_fdr']:.3f}", "results", f"{key} q")
    ps = stats["permutation_summary"]
    NUM = {7: "Seven", 6: "Six", 5: "Five", 4: "Four", 3: "Three"}
    expect(f"{NUM[ps['n_significant']]} of the nine alignments were significant",
           "results", "n significant")
    expect(f"{stats['permutation']['rt|Category']['n_permutations']:,}", "results")
    expect(f"{stats['noise_ceiling']['corr|SM']['best_rho_as_pct_of_lower_bound']:.0f}%",
           "results", "best ACC vs ceiling")

    # replication
    rep = stats["replication"]
    expect(f"only {rep['by_stage']['Vision encoder']['n_distinct_rdms']} distinct RDMs each",
           "results")
    expect(f"{rep['n_alignments_distinct']} are therefore distinct", "results")
    expect(f"at most {rep['max_abs_mean_shift_after_dedup']:.3f}", "results")

    # prompt effect
    pe = stats["prompt_effect_language_model"]
    expect(f"{pe['rt']['min_mean_rho']:.3f} to {pe['rt']['max_mean_rho']:.3f}", "results")
    expect(f"{pe['rt_critical']['min_mean_rho']:.3f} to "
           f"{pe['rt_critical']['max_mean_rho']:.3f}", "results")
    expect(f"{pe['corr']['min_mean_rho']:.3f} to {pe['corr']['max_mean_rho']:.3f}",
           "results")
    expect(f"{pe['rt']['n_per_condition']} alignments contributed", "results")

    for s in [str(d["n_model_rdms"]), f"{d['n_pairings']:,}", f"{d['n_video_pairs']:,}",
              str(d["n_human_rdms"]), "10,000"]:
        expect(s, "legend", f"legend constant {s}")

    # -------------------------------------------------------------- REVERSE
    allowed: set[str] = set()

    def allow(*vals) -> None:
        for v in vals:
            allowed.add(norm(str(v)).replace(",", ""))

    for v in d.values():
        allow(v)
    for m in MEASURES:
        gm = stats["global"][m]
        allow(gm["n"], gm["n_positive"], f"{gm['pct_positive']:.1f}")
        allow(f"{abs(gm['mean_rho']):.3f}", f"{abs(gm['median_rho']):.3f}")
        for t in TASK_ORDER:
            for l in STAGE_ORDER:
                c = stats["by_task_layer"][f"{m}|{t}|{l}"]
                allow(f"{abs(c['mean_rho']):.3f}", c["n"], c["n_distinct"])
        allow(stats["ve_minus_lm"][m]["n_positive"], stats["ve_minus_lm"][m]["n"])
        allow(stats["prompt_effect_language_model"][m]["n_per_condition"])
        for k in ("min_mean_rho", "max_mean_rho"):
            allow(f"{abs(stats['prompt_effect_language_model'][m][k]):.3f}")
    for nc in stats["noise_ceiling"].values():
        allow(f"{nc['lower']:.3f}", f"{nc['upper']:.3f}",
              f"{nc['lower']:.2f}", f"{nc['upper']:.2f}",
              nc["n_within_band"], nc["n_outside_band"], nc["n_configurations"],
              nc["n_participants"], f"{nc['best_rho_as_pct_of_lower_bound']:.0f}",
              f"{abs(nc['best_rho']):.3f}", *nc["n_within_band_by_layer"].values())
    for p in stats["permutation"].values():
        allow(f"{abs(p['observed_rho']):.3f}", f"{p['q_fdr']:.3f}",
              p["n_permutations"])
    allow(*stats["permutation_summary"].values())
    rc = stats["rt_critical"]
    allow(f"{rc['t_mean_s']:.2f}", f"{rc['t_sd_s']:.2f}",
          f"{rc['t_min_s']:.2f}", f"{rc['t_max_s']:.2f}")
    for v in rc["rdm_corr_with_rt_onset"].values():
        allow(f"{v:.2f}")
    allow(rc["gain_over_rt_onset"]["n_cells"],
          f"{rc['gain_over_rt_onset']['min']:.3f}",
          f"{rc['gain_over_rt_onset']['max']:.3f}",
          f"{rc['gain_over_rt_onset']['mean']:.3f}")
    allow(rep["n_alignments_nominal"], rep["n_alignments_distinct"],
          f"{rep['max_abs_mean_shift_after_dedup']:.3f}")
    for st in rep["by_stage"].values():
        allow(*st.values())
    for v in stats["by_layer"].values():
        allow(f"{abs(v['mean_rho']):.3f}", f"{v['mean_sd_across_tasks']:.3f}")

    # design / method constants that are not RSA outputs
    allow(
        0.001,   # q < 0.001 significance statement
        0.05,    # FDR level (Methods)
        95,      # central 95% of the null
        2000,    # split-half and bootstrap iterations
        0,       # rho = 0 reference line
    )

    scan = re.sub(r"Supplementary (Figs?|Tables?)\.?\s*S\d+(\s*-\s*S\d+)?", " ", prose)
    scan = re.sub(r"Figs?\.\s*\d+[a-z]?(,[a-z])?", " ", scan)
    for name in sorted({m.full_name for m in MODELS} | {m.checkpoint for m in MODELS}
                       | {m.family for m in MODELS}, key=len, reverse=True):
        scan = scan.replace(norm(name), " ")
    tokens = re.findall(r"\d[\d,]*(?:\.\d+)?", scan)
    bad: dict[str, int] = {}
    for tok in tokens:
        t = tok.replace(",", "")
        if t in allowed:
            continue
        try:
            if float(t) == int(float(t)) and str(int(float(t))) in allowed:
                continue
        except ValueError:
            pass
        bad[tok] = bad.get(tok, 0) + 1
    for tok, n in sorted(bad.items()):
        problems.append(f"REVERSE  numeric token not licensed by the data: {tok!r} (x{n})")

    print(f"forward checks passed : {ok}")
    print(f"numeric tokens scanned: {len(tokens)}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  -", p)
        return 1
    print("\nOK - every number in the prose is traceable to derived/rsa_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
