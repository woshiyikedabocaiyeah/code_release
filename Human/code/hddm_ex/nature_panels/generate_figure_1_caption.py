"""Generate the Figure 1 caption and its numerical supplement from the fits.

Every number in the caption and in Supplementary Table 1 is read from the
posterior draws and the behavioural fits at run time, so re-running this after a
refit keeps the text and the figure in agreement.  Nothing is transcribed by
hand.

Three facts about the figure are stated in the caption because they are not
inferable from the plot and mislead readers if omitted:

* Panel a is a schematic.  Its paths are drawn to observed reaction-time
  quantiles with Brownian noise added, not simulated from the posterior, and its
  horizontal axis is scaled per task rather than shared.
* Panel b's two sub-panels come from different fits.  Boundary separation is
  per-task only in the condition model; baseline drift is per-task only in the
  regression model, as intercept plus group shift.  Neither fit supplies both.
* The brackets are posterior probabilities, not p-values, and pairs below
  P = 0.95 are left unmarked rather than annotated.

Outputs (written next to the figures):
    figure_1_caption_short.md   -- journal length, no numbers
    figure_1_caption.md         -- archival, numbers inline
    supplementary_table_1_figure_estimates.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- path repair after reorganisation into _organized/code/ ---------------
# BASE_DIR: original project directory, still the source of data and the
# destination of figures. CODE_DIR: where this script and its sibling
# modules now live.
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parents[1]
for sub in ("manuscript_common", "nature_panels"):
    p = str(CODE_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib as mpl  # noqa: E402
import figure_style as fs  # noqa: E402

fs.apply(mpl)

import panel_b_boundary_ndt as pb  # noqa: E402
import panel_ce_feature_modulation as pce  # noqa: E402
import panel_df_posterior_coefs as pdf_  # noqa: E402

RESULTS = BASE_DIR / "hddm_results_4chains_2000samples"
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT / "nature"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TASKS = list(fs.TASK_ORDER)


def summarise(draws) -> dict:
    a = np.asarray(draws)
    return dict(mean=float(a.mean()),
                lo=float(np.percentile(a, 2.5)),
                hi=float(np.percentile(a, 97.5)))


def sign_probabilities(draws_by_task: dict) -> dict:
    """P(difference has the sign shown), and the count on the minority side.

    The count matters: with a finite number of draws a probability of exactly
    1.0 is not attainable, so an empty minority side is reported as a lower
    bound rather than as certainty.
    """
    out = {}
    keys = list(draws_by_task)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = np.asarray(draws_by_task[keys[j]]) - np.asarray(draws_by_task[keys[i]])
            minority = int(min((d > 0).sum(), (d < 0).sum()))
            out[(keys[i], keys[j])] = dict(
                p=float(max((d > 0).mean(), (d < 0).mean())),
                minority=minority, n=int(d.size))
    return out


def collect() -> dict:
    boundary = pb.boundary_draws()
    drift = pb.drift_draws()
    beh = {}
    for feat in pce.FEATURES:
        beh[feat] = {}
        for outcome, *_ in pce.MEASURES:
            _, slopes = pce.fit(feat, outcome)
            beh[feat][outcome] = {t: dict(estimate=float(d["estimate"]),
                                          ci_low=float(d["ci_low"]),
                                          ci_high=float(d["ci_high"]))
                                  for t, d in slopes.items()}
    coef = {f: {pdf_.OLD2NEW[c]: summarise(pdf_.draws(f, c)) for c in pdf_.M.GROUPS}
            for f in pdf_.FEATURES}

    raw = pce.raw_behavior()
    ddm = pd.read_csv(RESULTS / "rt_critical" / "rt_critical_cleaned.csv")
    # Panel a annotates the mean of CORRECT responses on the DDM-cleaned set,
    # which differs from the behavioural set: a reader recomputing from the
    # behavioural file alone will not reproduce the annotated values.
    rt_correct = {fs.CODE_TO_TASK[c]: float(g.loc[g["response"] == 1, "rt"].mean())
                  for c, g in ddm.groupby("group")}
    accuracy = {fs.CODE_TO_TASK[c]: float(g["response"].mean()) * 100
                for c, g in ddm.groupby("group")}

    points = pce.scatter(list(pce.FEATURES)[0], "ACC")

    # Accuracy points that fall below the axis floor.  Computed from dyn_ylim
    # directly rather than read off a rendered figure: the count is no longer
    # annotated on the panel (no free position at composite width), so the
    # caption is the only place it is disclosed and must not depend on plotting.
    clipped = {}
    for feat in pce.FEATURES:
        sc = pce.scatter(feat, "ACC")
        curves, _ = pce.fit(feat, "ACC")
        _, n_clip = pce.dyn_ylim(sc, curves, floor_q=0.04)
        clipped[feat] = (int(n_clip), int(len(sc)))
    n_clip_set = {v for v in clipped.values()}
    if len(n_clip_set) == 1:
        nc, ntot = n_clip_set.pop()
        clipped_note = (
            f"with the axis floor excluding {nc} of {ntot} video means "
            f"(fits are never clipped)."
        ) if nc else ""
    else:
        parts = ", ".join(f"{v[0]} of {v[1]} in {k}" for k, v in clipped.items())
        # Same phrase form as the single-count branch: the caption text appends
        # this mid-sentence, so a standalone sentence here would break grammar.
        clipped_note = (f"with the axis floor excluding {parts} video means "
                        f"(fits are never clipped).")

    # Backticked: bare *** is bold-italic markup in Markdown, so an unescaped
    # key renders as emphasis instead of as the glyphs drawn on the panel.
    star_key = "; ".join(
        f"`{glyph}` $P$ > {thr}" for thr, glyph in pb.STAR_LEVELS)
    return dict(
        boundary={t: summarise(v) for t, v in boundary.items()},
        drift={t: summarise(v) for t, v in drift.items()},
        boundary_P=sign_probabilities(boundary),
        drift_P=sign_probabilities(drift),
        behaviour=beh, coef=coef,
        n_draws=int(len(next(iter(boundary.values())))),
        rt_correct=rt_correct, accuracy=accuracy,
        n_videos=int(points["x"].nunique()),
        n_points=int(len(points)),
        per_point=dict(median=int(points["n"].median()),
                       lo=int(points["n"].min()), hi=int(points["n"].max())),
        n_subjects=int(raw["Subject"].nunique()),
        # Stated exactly rather than as "each performed one task": the design is
        # between-participants but one participant appears in two conditions, so
        # the absolute claim would be false and a reviewer checking would find it.
        n_multi_task=int((raw.groupby("Subject")["condition"].nunique() > 1).sum()),
        n_trials=int(len(raw)),
        n_by_condition={fs.CODE_TO_TASK[k]: int(v)
                        for k, v in raw["condition"].value_counts().items()},
        ddm_trials=int(len(ddm)),
        clipped=clipped, clipped_note=clipped_note, star_key=star_key,
    )


def table(q: dict) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        rows.append(dict(panel="a", quantity="Mean RT_critical, correct responses",
                         model="observed (DDM-cleaned set)", task=task,
                         estimate=round(q["rt_correct"][task], 3),
                         ci_low=None, ci_high=None, unit="s"))
        rows.append(dict(panel="a", quantity="Accuracy", model="observed (DDM-cleaned set)",
                         task=task, estimate=round(q["accuracy"][task], 1),
                         ci_low=None, ci_high=None, unit="%"))
    for key, label, model in (("boundary", "Boundary separation a", "condition"),
                              ("drift", "Baseline drift rate v",
                               "regression (intercept + group shift)")):
        for task in TASKS:
            d = q[key][task]
            rows.append(dict(panel="b", quantity=label, model=model, task=task,
                             estimate=round(d["mean"], 4), ci_low=round(d["lo"], 4),
                             ci_high=round(d["hi"], 4), unit="a.u."))
    for key, sym, model in (("drift_P", "v", "regression"),
                            ("boundary_P", "a", "condition")):
        for (t1, t2), d in q[key].items():
            if d["minority"] == 0:
                est = round(1 - 1 / d["n"], 4)
                unit = (f"posterior probability, lower bound "
                        f"(0 of {d['n']} draws on the opposite side)")
            else:
                est = round(d["p"], 4)
                unit = "posterior probability"
                if d["p"] < pb.P_BRACKET:
                    unit += f" (unmarked, < {pb.P_BRACKET})"
            rows.append(dict(panel="b", quantity=f"P(sign of Δ{sym}): {t1} vs {t2}",
                             model=model, task="-", estimate=est,
                             ci_low=None, ci_high=None, unit=unit))
    for feat, pn in zip(pce.FEATURES, ("c", "e")):
        for outcome, *_ in pce.MEASURES:
            scale = 100.0 if outcome == "ACC" else 1.0
            unit = "pp per unit z" if outcome == "ACC" else "s per unit z"
            for task in TASKS:
                d = q["behaviour"][feat][outcome][task]
                rows.append(dict(panel=pn, quantity=f"{outcome} slope on {feat}",
                                 model="LMM (random intercept by participant)",
                                 task=task, estimate=round(d["estimate"] * scale, 4),
                                 ci_low=round(d["ci_low"] * scale, 4),
                                 ci_high=round(d["ci_high"] * scale, 4), unit=unit))
    for feat, pn in zip(pdf_.FEATURES, ("d", "f")):
        for task in TASKS:
            d = q["coef"][feat][task]
            # panel_df keys features in lower case while panel_ce keys them as
            # they appear on the axes; the table follows the axis spelling so a
            # reader can match a row to a panel label.
            rows.append(dict(panel=pn, quantity=f"v_{feat.title()} coefficient",
                             model="regression", task=task,
                             estimate=round(d["mean"], 4), ci_low=round(d["lo"], 4),
                             ci_high=round(d["hi"], 4), unit="a.u."))
    return pd.DataFrame(rows)[["panel", "quantity", "model", "task",
                               "estimate", "ci_low", "ci_high", "unit"]]


def table_markdown(tab: pd.DataFrame) -> str:
    """The same table as Markdown, for pasting into the supplement."""
    out = ["# Supplementary Table 1 | Numerical estimates for every panel of Figure 1",
           "",
           "Posterior means with 95% credible intervals for model parameters;",
           "mixed-model slopes with 95% confidence intervals for behavioural",
           "effects. Generated by `nature_panels/generate_figure_1_caption.py`",
           "directly from the fits.",
           ""]
    for pn in ("a", "b", "c", "d", "e", "f"):
        out += [f"## Panel {pn}", "",
                "| Quantity | Source | Task | Estimate | 95% interval | Unit |",
                "|---|---|---|---|---|---|"]
        for _, r in tab[tab["panel"] == pn].iterrows():
            interval = "-" if pd.isna(r["ci_low"]) else f"{r['ci_low']:.4g}, {r['ci_high']:.4g}"
            out.append(f"| {r['quantity']} | {r['model']} | {r['task']} | "
                       f"{r['estimate']:.4g} | {interval} | {r['unit']} |")
        out.append("")
    return "\n".join(out)


def short_caption(q: dict) -> str:
    return f"""**Fig. 1 | Task-dependent evidence accumulation and its modulation by visual and physical scene features.**

**a**, Schematic drift-diffusion trajectories per task: paths drawn to reaction
times sampled from the observed quantiles with added Brownian noise, not
posterior-predictive simulations. Shaded profiles are kernel densities of the
observed $\\mathrm{{RT_{{critical}}}}$ for correct (upper) and error (lower)
responses; the horizontal axis is scaled per task, not shared. Dashed verticals,
mean correct $\\mathrm{{RT_{{critical}}}}$; $v$, baseline drift rate; $t$, shared
non-decision time, marked but not contrasted across tasks.

**b**, Posteriors of boundary separation $a$ and baseline drift rate $v$, one
violin per task ({q['n_draws']} draws; box, interquartile range; whiskers,
2.5th-97.5th percentile; white marker, mean). $a$ comes from the condition model
and $v$ from the regression model (intercept plus group shift, reconstructed
draw-by-draw), because the regression model estimates a single shared boundary
and cannot give a per-task boundary contrast. Stars give the posterior
probability $P$ that the difference has the sign shown -- {q['star_key']} -- and
are posterior probabilities, not p-values; $P$ below {pb.P_BRACKET} is unmarked.

**c**, **e**, Behaviour as a function of $\\mathrm{{Visual_{{Z}}}}$ (**c**) and
$\\mathrm{{Physical_{{Z}}}}$ (**e**), each adjusted for the other. Points are the
{q['n_videos']} stimulus videos within each task, averaged over participants
(median {q['per_point']['median']}). Lines are single-trial linear mixed-model
fits with random intercepts by participant; bands, 95% confidence intervals;
annotated $b$, per-task slopes. Reaction-time panels exclude trials at or below
0.2 s; accuracy is modelled on the linear probability scale (percentage points),
{q['clipped_note']}

**d**, **f**, Posterior densities of the $v_{{\\mathrm{{Visual_{{Z}}}}}}$ (**d**)
and $v_{{\\mathrm{{Physical_{{Z}}}}}}$ (**f**) coefficients. Dashed verticals,
posterior means; the x-axis tick at 0 locates zero.

Colours denote task (Okabe-Ito palette; key at right of b). Task was manipulated between
participants: of {q['n_subjects']} participants, all but {q['n_multi_task']}
performed a single task, each viewing all {q['n_videos']} videos;
{q['n_trials']} trials in **c**, **e**, {q['ddm_trials']} in the
drift-diffusion fits. Numerical estimates, Supplementary Table 1.
"""


def main() -> None:
    q = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figure_1_caption_short.md").write_text(short_caption(q))
    tab = table(q)
    tab.to_csv(OUT_DIR / "supplementary_table_1_figure_estimates.csv", index=False)
    (OUT_DIR / "supplementary_table_1_figure_estimates.md").write_text(table_markdown(tab))
    print(f"    caption: {len(short_caption(q).split())} words")
    print(f"    table:   {len(tab)} rows across panels "
          f"{sorted(tab['panel'].unique())}")


if __name__ == "__main__":
    main()
