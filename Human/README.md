# _organized — Project Archive Root

**Consolidated archive** of the Intuitive Physics behavioral experiments + HDDM/LMM modeling + video feature extraction.
Code, figures, raw data, and stimulus videos that were originally scattered under `AI agent/` have been gathered into this directory and stored in layers by purpose.

> Note: this directory initially held only "code + figures" (data stayed in place). Later, the
> **raw data (`raw_data/`)**, **stimulus videos (`video info/`)**, and
> **feature-extraction code (`feature extraction/`)** were all moved in as well, so it is now a nearly complete copy of the project.

## Top-Level Structure

| Directory / File | Contents | Size |
|---|---|---|
| `code/` | All analysis code (.py / .R / .Rmd / .mjs / .sh) | 63 files · 940 KB |
| `figures/` | All result figures (.png / .pdf / .svg) | 172 files · 48 MB |
| `raw_data/` | **Per-subject raw data** for each task (Pavlovia exports) + demographics | 771 files · 39 MB |
| `video info/` | 48 stimulus videos (.mp4) + `output.csv` (video timeline) | 49 files · 10 MB |
| `feature extraction/` | Video feature extraction (V-JEPA / VideoMAEv2 / COSMOS, etc.) | 179 files · 25 MB |
| `task_question_only_0122.xlsx` | Task / question list | 18 KB |
| `MANIFEST.csv` / `restore.sh` | **Historical record** (see end of file; covers only the original code+figures migration) | — |
| `README.md` | This file | — |

## The Three Behavioral Tasks and Their Naming

The project has 3 tasks, each collected in two batches (part1 / part2). The analysis code uses short names
(`cat` / `sensor` / `voe`), while `raw_data/` uses full names:

| Short name | Full name (`raw_data/` dir) | Per-subject files | Description |
|---|---|---|---|
| `cat` | `concept verification` | 261 | Categorization / concept judgment (= 140 + 121 across two batches) |
| `sensor` | `affordance recognition` | 258 | Sensorimotor / affordance recognition |
| `voe` | `plausibility assessment` | 246 | Violation of expectation / plausibility judgment |
| — | `demographic` | 6 | Demographic questionnaire (cat/sensor/voe × part1/2, as .numbers) |

> The two batches of `cat` raw data are also kept in `AI agent/project1_ex/cat/` (140 people, incl. `1.csv` = subject 347826)
> and `project2_ex/cat/` (121 people). See `raw_data/README.md` for details.

## Data Pipeline (raw → figures)

```
raw_data/<task>/*.csv                 per-subject PsychoPy/Pavlovia exports
  └─ code/project1_ex, project2_ex    aggregate by batch × task → all_summary_*, all_dat_*
       └─ code/data_analysis/         per-task notebooks: descriptive stats + RSA/PCA (*_matrix)
       └─ code/lmm, ex_lmm            linear mixed models (RT / ACC / IES)
       └─ code/hddm_project           HDDM pipeline: prepare → run → postprocess
            └─ code/hddm_ex           final HDDM figures (incl. all panels of paper Figure 1)
feature extraction/                   video-model features → video_summary_*.csv → fed to *_matrix RSA
video info/                           the 48 stimulus videos themselves
```

## `code/` Breakdown

| Subdirectory | Files | Contents |
|---|---|---|
| `data_analysis/` | 17 | Per-task R notebooks: `ex_cat`/`ex_voe`/`ex_sensor`, `*_matrix` (RSA/PCA), `pilot_*`, `time_extract.py` |
| `hddm_ex/` | 19 | Final HDDM analysis and figures; incl. `nature_panels/` (paper Figure 1 panels) and `manuscript_common/figure_style.py` |
| `hddm_project/` | 18 | HDDM modeling pipeline (`prepare_hddm_data` → `run_hddm_models` → `postprocess`) + `plot_*` scripts (generic, take `--input`/`--data`) |
| `lmm/` | 2 | Linear mixed model RT-effect plotting scripts (generic, take `--input`) |
| `ex_lmm/` | 1 | Full-sample LMM notebook `lmm.Rmd` |
| `project1_ex/` | 3 | **Batch 1** raw-data aggregation scripts per task (`cat`/`sensor`/`voe`) |
| `project2_ex/` | 3 | **Batch 2** raw-data aggregation scripts |

## `figures/` Breakdown

Figures preserve the output hierarchy of their respective analyses:

- `figures/hddm_ex/hddm_results_4chains_2000samples/` — final HDDM figures, incl. `nature/` (+ `no_key/`), `rt_condition_sample_style/`, `rt_critical/`, `rt_onset/` (each with `group/ null/ regression/`)
- `figures/data_analysis/<task>/` — behavioral and RSA/PCA figures: `ex`, `ex_cat_matrix`, `ex_sensor_matrix`, `ex_voe_matrix`, `pilot`, `pilot_*_matrix`

## `feature extraction/`

Complete third-party video-model code (extracting pieces would break the package structure, so it is **kept whole**):
`app/`, `src/`, `evals/`, `checkpoints/` (empty), entry scripts `run_my_intphys2.py`,
`video_density.py`, `sci-image.py`, and the outputs `video_summary*.csv` / `frame_metrics.csv`.
These `video_summary_*.csv` files are the feature inputs to each task's `*_matrix` RSA analysis.

## Path Changes Made in the Code

Migration breaks two assumptions — "derive the project root from `__file__`" and "read data by bare relative filename" — both now fixed.
Changed spots are marked with `# --- path repair ---` / `# --- figure output redirected ---` comments:

1. **Module lookup path** — distinguish `BASE_DIR` (original data dir) from `CODE_DIR` (this code dir, used for `sys.path`/`import`). Affects all panel scripts in `nature_panels/` and several export scripts in `hddm_ex`.
2. **Bare relative data reads** — 5 R notebooks now use `file.path(data_dir, ...)` to point back at the original data dir; `effect_sizes_addon.R` gains a `folder_path` fallback.
3. **Figure output paths** — Python uses `OUT_DIR`/`FIGURES_DIR` pointing at `figures/`; the plot scripts in `hddm_project`/`lmm` redirect via `_organized_fig_dir()`; R notebooks gain a `fig_dir`, and all `ggsave()` calls point to it (`write.csv()`/`saveRDS()` still write to the original `save_path`).

## Redundancy Deleted During This Cleanup (irreversible)

- **Small samples (superseded by the full experiment)**: `lmm` (an early 627 KB sample) and the small-sample **results** of `hddm_project`
  (figures + `outputs*/` + small-sample merged CSVs). Kept: both `scripts/` dirs, `hddm_project/vendor/`
  (the HDDM library), and `hddm_project/.venv*` (needed to run `hddm_ex`).
- **Duplicate figures**: 4 pairs of identical-pixel aliases (`cat_RT_video_RDM`, `*_rt_colors`,
  `*_grouped_by_PC`); the canonical RSM/base names were kept. Also removed the redundant
  `make_rsm_plot()`/`ggsave()` calls in `ex_cat_matrix.Rmd` / `ex_voe_matrix.Rmd`, so re-runs won't regenerate them.
- **Misclassified draft figures**: 9 `000010*.png` images in the original `stimuli/` (manual RStudio exports with overlapping axis labels),
  superseded by the official figures in `figures/data_analysis/ex/`; the whole directory was deleted.
- **Miscellaneous**: `.DS_Store` / `.Rhistory` etc. throughout.

## ⚠️ Missing Items

- **`references/`** (3 literature PDFs: `2110.05836v2.pdf`, `301_Benchmarking_Progress_to_I.pdf`,
  `physical reasoning.pdf`, ~18.5 MB) **vanished from disk** during the cleanup — not found in Trash / iCloud / Spotlight,
  and not deleted by this cleanup's scripts. To recover, check Time Machine / backups. The filenames and sizes are still recorded in `MANIFEST.csv`.

## Current Status of MANIFEST.csv / restore.sh

Both are the historical record and rollback script of the **original "code + figures" migration**. They **do not cover** the later-added
`raw_data/`, `video info/`, or `feature extraction/`, and the three `references` rows point to files that no longer exist.
Since the project is now anchored on this directory as the archive home, `restore.sh` (which moves files back to their scattered old locations) is essentially obsolete,
and is kept only for traceability. **The current 238 records = 63 code + 172 figures + 3 (now-invalid references).**
