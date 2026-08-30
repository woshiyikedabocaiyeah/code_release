# Code for *Failing the physical world: Multimodal LLMs share human visual load but lack grounded physical reasoning*

Code for the model side of the study: running the 11 model–mode variants on the three physical
reasoning tasks, turning their activations into the representation matrices we analyse, and
reproducing the reported representational similarity, principal component and probing results.

```
experiment/               running the models: prompts, per-model wrappers, cluster orchestration
representation_pipeline/  raw activations -> pooled matrices -> RDMs -> analysis outputs
analysis/                 RSA, PCA and layer-wise probing
common/                   path resolution and model naming, imported by the analysis scripts
environment-mllm.yml      conda environment for inference
(LICENSE and .gitignore live at the repository root)
```

## Setup

```bash
export NMI_DATA_ROOT=/path/to/the/data/deposit
```

`NMI_DATA_ROOT` is the directory containing `stimuli/`, `behaviour/`, `representations/`,
`pca/`, `probing/` and `alignment_with_human/`. Every script resolves its inputs through
`common/paths.py`; nothing else needs configuring. Results are written to a `derived/`
directory beside each script, or under `NMI_OUT_ROOT` if that is set.

Inference needs `environment-mllm.yml` and a GPU. The analyses need
`environment-analysis.yml` and run on a laptop.

---

## experiment/

Runs the models. `inference/run_main_experiment.py` is the entry point: for each of the 144
items it builds the prompt, dispatches to the wrapper for the model being run, parses the
answer, and appends to the result JSON.

Every prompt opens with the same introduction to the three physical concepts our human
participants read, then presents the task-specific yes/no question. In the embodied conditions
the prompt additionally instructs the model to reason as an embodied agent with a simulated
human-like body schema; in the detailed conditions it additionally includes the object-level
video description from `detailprompt.csv`. Answer options are never shown to the model, which
is required to answer in one word. Video input is eight frames sampled uniformly across the
clip and converted to RGB before model-specific preprocessing.

`inference/` contains that entry point, one inference wrapper per model in `Models/` (each
using the model's native processor and chat template), the fixed prompt components in
`prompt_templates.py`, the assembly logic in `prompt_factory.py`, the yes/no extraction and
output validation in `answer_parser.py`, the checkpoint registry in `model_config.py`, and the
two stimulus files the run needs: `questions.json` and `detailprompt.csv`.

Direct-answer and base regimes use greedy decoding with sampling disabled. For thinking-style
regimes whose wrappers expose sampling parameters we set the effective temperature to zero.
Maximum generation length and repetition penalties are configured per model to prevent
truncation and degenerate repetition, and are recorded in the trial metadata.

`cluster_orchestration/` holds the scripts we used to run this at scale: Slurm submission,
parallel launch across models, result merging, and the fixed-subset sampling-stability
experiment. `tools/` holds checkpoint download and environment validation.

Note that `answer_parser.py` and `run_main_experiment.py` each contain a regular expression
with Chinese alternatives. These are functional, not leftovers: most of the evaluated
checkpoints are Chinese-origin and can prefix a reply with the Chinese word for "assistant" or
use a full-width colon.

## representation_pipeline/

Turns the raw per-trial activations into everything downstream.
`recover_review_package.py` is the single entry point: it mean-pools each activation tensor
over the sequence dimension, groups the resulting vectors by model variant, prompt condition,
task and stage, writes the 396 representation matrices with their row metadata, builds the
cosine-distance dissimilarity matrices, and computes the task performance, principal component,
representational similarity, noise-ceiling and permutation-test outputs.

```bash
bash run_full_recovery.sh
```

`compare_review_package_schema.py` and `repair_schema_compatibility.py` verify and reconcile
output schemas across runs. The `.sbatch` files run the GPU stage on Slurm.

## analysis/

Three pipelines, each with a `run_all.sh` that regenerates the values we report.

```bash
bash llm/analysis/rsa/run_all.sh        # Results: "RSA revealed aligned processing cost..."
bash llm/analysis/pca/run_all.sh        # Results: "PCA revealed divergent representational geometries"
bash llm/analysis/probing/run_all.sh    # Results: "Probing revealed failure to utilize..."
```

**`rsa/`** runs the permutation test for each of the nine human matrices (10,000 permutations
of the model's video labels with the human matrix held fixed, two-sided, Benjamini–Hochberg
corrected across the nine tests), extracts the reported statistics including the split-half and
bootstrap noise ceilings and how many of the 132 configurations per task and measure fall
inside each band, and checks the recomputed values against the paper.

`llm/analysis/rsa/optional/05_build_rt_critical.py` rebuilds the RT(critical) layer from participant-level
human reaction times. The data deposit ships that layer already built in
`alignment_with_human/rt_critical/`, and the pipeline reads it from there, so this script is
not part of `run_all.sh`. Running it requires `behaviour/inclusion_flags.csv` and
`rdm/rt_wide_{cat,voe,sensor}.csv` under `$NMI_DATA_ROOT`, which are not in this deposit; if
you have them, its output takes precedence over the shipped layer.

**`pca/`** computes, for every principal component of every representation matrix, the share of
variance in component score across the 48 videos explained by physical concept, plausibility
and scene template (η² from a one-way analysis of variance), reports the dominant factor per
component, and summarises explained variance by stage.

**`probing/`** re-runs the probes under video-grouped stratified six-fold cross-validation over
three seeds, so that every video is held out exactly once and no video appears on both sides of
a split. Each probe is a multinomial logistic regression; features are standardised and rotated
onto their *n*−1 estimable principal directions, both fitted on the training rows of each fold
only, with regularisation selected on an inner three-fold split of the training folds. The
remaining steps are the controls: the label-permutation control (200 permutations per model),
the nearest-neighbour geometry analysis with its greyscale-pixel baseline, a
non-linear-readout architecture control, and probe-capacity diagnostics. The permutation
control takes about 25 minutes; set `SKIP_PERM=1` to reuse a cached result.

The RSA and probing pipelines each end with a script that recomputes every value quoted in the
corresponding Results section. Point it at the section text to have it compare:

```bash
export NMI_SECTION_MD=/path/to/section.md
```

Without that variable set, the script prints the recomputed values without comparing.

## common/

`paths.py` resolves the layout of the data deposit from `NMI_DATA_ROOT` and is the only place
any input path is written down. `model_naming.py` maps the internal slugs used in file names to
the model and stage names used in the paper. Both are imported by the analysis scripts.

## Licence

MIT. See `LICENSE`.
