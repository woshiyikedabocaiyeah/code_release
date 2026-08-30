#!/usr/bin/env python3
"""Export a reproducible RT_critical data package from the existing analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# --- path repair after reorganisation into _organized/code/ ---------------
# BASE_DIR: original project directory, still the source of data and the
# destination of figures. CODE_DIR: where this script and its sibling
# modules now live.
BASE_DIR = Path(__file__).resolve().parent
SOURCE_RAW = BASE_DIR / "all_dat_merged.csv"
RESULTS_DIR = BASE_DIR / "hddm_results_4chains_2000samples" / "rt_critical"
OUTPUT_DIR = BASE_DIR / "outputs" / "rt_critical_complete_data_20260806"

CONDITION_TO_GROUP = {
    "categorization": "Semantic",
    "Voe": "Intuitive",
    "sensorimotor": "Action",
}

CONDITION_TO_TASK = {
    "categorization": "Concept Verification",
    "Voe": "Plausibility Assessment",
    "sensorimotor": "Affordance Recognition",
}

MODEL_NAMES = ("null", "group", "regression")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if not path.is_file() or path.name == "manifest.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(OUTPUT_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "manifest.csv", index=False)


def build_exclusion_reason(raw: pd.DataFrame) -> pd.Series:
    rt = pd.to_numeric(raw["RT_critical"], errors="coerce")
    reasons = pd.Series("", index=raw.index, dtype="object")
    reasons.loc[rt < 0] = "negative_RT_critical"

    required = {
        "Subject": raw["Subject"],
        "RT_critical": rt,
        "ACC": pd.to_numeric(raw["ACC"], errors="coerce"),
        "condition_mapping": raw["condition"].map(CONDITION_TO_GROUP),
        "Visual_Z": pd.to_numeric(raw["Visual_Z"], errors="coerce"),
        "Physical_Z": pd.to_numeric(raw["Physical_Z"], errors="coerce"),
    }
    for field, values in required.items():
        missing = values.isna() & reasons.eq("")
        reasons.loc[missing] = f"missing_{field}"
    return reasons


def expected_hddm_input(raw: pd.DataFrame, included: pd.Series) -> pd.DataFrame:
    expected = pd.DataFrame(
        {
            "subj_idx": raw["Subject"].astype(str),
            "rt": pd.to_numeric(raw["RT_critical"], errors="coerce"),
            "response": pd.to_numeric(raw["ACC"], errors="coerce").astype(int),
            "group": raw["condition"].map(CONDITION_TO_GROUP),
            "visual_z": pd.to_numeric(raw["Visual_Z"], errors="coerce"),
            "physical_z": pd.to_numeric(raw["Physical_Z"], errors="coerce"),
            "condition": raw["condition"],
        }
    )
    return expected.loc[included].reset_index(drop=True)


def verify_cleaned_data(expected: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    if list(expected.columns) != list(cleaned.columns):
        raise ValueError("The reconstructed HDDM input columns do not match the saved input.")
    if len(expected) != len(cleaned):
        raise ValueError("The reconstructed HDDM input row count does not match the saved input.")

    numeric_columns = ["rt", "response", "visual_z", "physical_z"]
    for column in numeric_columns:
        if not np.allclose(
            pd.to_numeric(expected[column]),
            pd.to_numeric(cleaned[column]),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"Mismatch in cleaned HDDM column: {column}")

    text_columns = ["subj_idx", "group", "condition"]
    for column in text_columns:
        if not expected[column].astype(str).equals(cleaned[column].astype(str)):
            raise ValueError(f"Mismatch in cleaned HDDM column: {column}")


def make_subject_summary(focused: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, condition), frame in focused.groupby(["Subject", "condition"], sort=True):
        included = frame[frame["hddm_included"] == 1]
        rows.append(
            {
                "Subject": subject,
                "condition": condition,
                "task_name": CONDITION_TO_TASK[condition],
                "legacy_group": CONDITION_TO_GROUP[condition],
                "raw_trials": len(frame),
                "included_trials": len(included),
                "excluded_trials": len(frame) - len(included),
                "exclusion_rate": (len(frame) - len(included)) / len(frame),
                "mean_rt_critical_raw": frame["RT_critical"].mean(),
                "median_rt_critical_raw": frame["RT_critical"].median(),
                "mean_rt_critical_included": included["RT_critical"].mean(),
                "median_rt_critical_included": included["RT_critical"].median(),
                "accuracy_raw": frame["ACC"].mean(),
                "accuracy_included": included["ACC"].mean(),
                "mean_visual_z": frame["Visual_Z"].mean(),
                "mean_physical_z": frame["Physical_Z"].mean(),
            }
        )
    return pd.DataFrame(rows)


def make_task_summary(focused: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in ("categorization", "Voe", "sensorimotor"):
        frame = focused[focused["condition"] == condition]
        included = frame[frame["hddm_included"] == 1]
        rows.append(
            {
                "condition": condition,
                "task_name": CONDITION_TO_TASK[condition],
                "legacy_group": CONDITION_TO_GROUP[condition],
                "subjects": frame["Subject"].nunique(),
                "raw_trials": len(frame),
                "included_trials": len(included),
                "excluded_trials": len(frame) - len(included),
                "exclusion_rate": (len(frame) - len(included)) / len(frame),
                "mean_rt_critical_raw": frame["RT_critical"].mean(),
                "median_rt_critical_raw": frame["RT_critical"].median(),
                "mean_rt_critical_included": included["RT_critical"].mean(),
                "median_rt_critical_included": included["RT_critical"].median(),
                "accuracy_raw": frame["ACC"].mean(),
                "accuracy_included": included["ACC"].mean(),
            }
        )
    return pd.DataFrame(rows)


def combine_chain_stats(csv_dir: Path) -> pd.DataFrame:
    frames = []
    for model in MODEL_NAMES:
        for chain in range(1, 5):
            path = RESULTS_DIR / model / f"chain_{chain}_stats.csv"
            frame = pd.read_csv(path, index_col=0).reset_index()
            frame = frame.rename(columns={frame.columns[0]: "parameter"})
            frame.insert(0, "chain", chain)
            frame.insert(0, "model", model)
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(csv_dir / "07_hddm_model_chain_stats.csv", index=False)
    return combined


def combine_ppc_stats(csv_dir: Path) -> pd.DataFrame:
    frames = []
    for model in MODEL_NAMES:
        frame = pd.read_csv(RESULTS_DIR / model / "posterior_predictive_stats.csv")
        frame.insert(0, "model", model)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(csv_dir / "08_hddm_posterior_predictive_stats.csv", index=False)
    return combined


def combine_convergence(csv_dir: Path) -> pd.DataFrame:
    rows = []
    for model in MODEL_NAMES:
        values = json.loads((RESULTS_DIR / model / "gelman_rubin.json").read_text())
        rows.extend(
            {"model": model, "parameter": parameter, "r_hat": value}
            for parameter, value in values.items()
        )
    combined = pd.DataFrame(rows)
    combined.to_csv(csv_dir / "09_hddm_gelman_rubin.csv", index=False)
    return combined


def extract_model_metadata(csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows = []
    posterior_rows = []
    probability_rows = []

    for model in MODEL_NAMES:
        summary = json.loads((RESULTS_DIR / model / "summary.json").read_text())
        for chain in summary["chains"]:
            run_rows.append(
                {
                    "model": model,
                    "chain": chain["chain"],
                    "seed": chain["seed"],
                    "runtime_seconds": chain["runtime_seconds"],
                    "dic": chain["dic"],
                    "loaded_existing": chain["loaded_existing"],
                    "model_path": chain["model_path"],
                    "trace_db_path": chain["trace_db_path"],
                    "model_r_hat_max": summary.get("rhat_max"),
                }
            )

        if model != "regression":
            continue

        hypotheses = summary.get("hypotheses", {})
        for parameter_family in ("drift_intercepts", "visual_slopes", "physical_slopes"):
            for group, stats in hypotheses.get(parameter_family, {}).items():
                legacy_group = group.capitalize()
                condition = next(
                    key for key, value in CONDITION_TO_GROUP.items() if value == legacy_group
                )
                posterior_rows.append(
                    {
                        "parameter_family": parameter_family,
                        "condition": condition,
                        "task_name": CONDITION_TO_TASK[condition],
                        "legacy_group": legacy_group,
                        **stats,
                    }
                )
        probability_rows.extend(
            {"hypothesis": key, "posterior_probability": value}
            for key, value in hypotheses.items()
            if isinstance(value, (int, float))
        )

    run_table = pd.DataFrame(run_rows)
    posterior_table = pd.DataFrame(posterior_rows)
    probability_table = pd.DataFrame(probability_rows)
    run_table.to_csv(csv_dir / "10_hddm_model_runs.csv", index=False)
    posterior_table.to_csv(csv_dir / "11_hddm_regression_posterior_summary.csv", index=False)
    probability_table.to_csv(csv_dir / "12_hddm_hypothesis_probabilities.csv", index=False)
    return run_table, posterior_table, probability_table


def export_effect_sizes(csv_dir: Path) -> pd.DataFrame:
    frames = []
    for predictor, filename in (
        ("Visual_Z", "hddm_visualz_effect_sizes.csv"),
        ("Physical_Z", "hddm_physicalz_effect_sizes.csv"),
    ):
        frame = pd.read_csv(BASE_DIR / filename)
        frame = frame.loc[frame["metric"].eq("RTcritical")].copy()
        frame.insert(0, "predictor", predictor)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(csv_dir / "13_rt_critical_effect_sizes.csv", index=False)
    return combined


def copy_original_analysis_tables() -> None:
    destination = OUTPUT_DIR / "original_analysis_tables"
    for path in sorted(RESULTS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
            target = destination / path.relative_to(RESULTS_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def write_model_file_index(csv_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        packaged = suffix in {".csv", ".json"}
        rows.append(
            {
                "relative_path": str(path.relative_to(RESULTS_DIR)),
                "file_type": suffix or "none",
                "size_bytes": path.stat().st_size,
                "copied_into_package": packaged,
                "package_note": (
                    "Copied under original_analysis_tables"
                    if packaged
                    else "Not duplicated; retained in the original RT_critical results directory"
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(csv_dir / "14_original_model_file_index.csv", index=False)
    return table


def prepare_package() -> None:
    csv_dir = OUTPUT_DIR / "csv"
    metadata_dir = OUTPUT_DIR / "metadata"
    csv_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(SOURCE_RAW)
    cleaned = pd.read_csv(RESULTS_DIR / "rt_critical_cleaned.csv")
    reasons = build_exclusion_reason(raw)
    included = reasons.eq("")

    expected = expected_hddm_input(raw, included)
    verify_cleaned_data(expected, cleaned)

    shutil.copy2(SOURCE_RAW, csv_dir / "01_human_raw_original_all_columns.csv")

    focused = raw.copy()
    focused.insert(0, "source_row", np.arange(2, len(focused) + 2))
    focused.insert(4, "task_name", focused["condition"].map(CONDITION_TO_TASK))
    focused.insert(5, "legacy_group", focused["condition"].map(CONDITION_TO_GROUP))
    focused["hddm_included"] = included.astype(int)
    focused["exclusion_reason"] = reasons
    focused.to_csv(csv_dir / "02_human_rt_critical_with_inclusion_flags.csv", index=False)

    shutil.copy2(
        RESULTS_DIR / "rt_critical_cleaned.csv",
        csv_dir / "03_rt_critical_hddm_input.csv",
    )

    excluded_columns = [
        "source_row",
        "Video",
        "Subject",
        "condition",
        "task_name",
        "legacy_group",
        "RT_onset",
        "RT_critical",
        "ACC",
        "Visual_Z",
        "Physical_Z",
        "exclusion_reason",
    ]
    focused.loc[~included, excluded_columns].to_csv(
        csv_dir / "04_rt_critical_excluded_trials.csv", index=False
    )

    subject_summary = make_subject_summary(focused)
    task_summary = make_task_summary(focused)
    subject_summary.to_csv(csv_dir / "05_rt_critical_subject_summary.csv", index=False)
    task_summary.to_csv(csv_dir / "06_rt_critical_task_summary.csv", index=False)

    chain_stats = combine_chain_stats(csv_dir)
    ppc_stats = combine_ppc_stats(csv_dir)
    convergence = combine_convergence(csv_dir)
    model_runs, posterior_summary, probabilities = extract_model_metadata(csv_dir)
    effect_sizes = export_effect_sizes(csv_dir)
    file_index = write_model_file_index(csv_dir)
    copy_original_analysis_tables()

    for filename in ("rt_summary.json",):
        shutil.copy2(RESULTS_DIR / filename, metadata_dir / filename)
    for model in MODEL_NAMES:
        shutil.copy2(
            RESULTS_DIR / model / "summary.json",
            metadata_dir / f"{model}_summary.json",
        )

    package_summary = {
        "source_file": str(SOURCE_RAW),
        "rt_source_column": "RT_critical",
        "raw_trials": int(len(raw)),
        "included_trials": int(included.sum()),
        "excluded_trials": int((~included).sum()),
        "exclusion_rate": float((~included).mean()),
        "subjects": int(raw["Subject"].nunique()),
        "missing_rt_critical": int(pd.to_numeric(raw["RT_critical"], errors="coerce").isna().sum()),
        "negative_rt_critical": int((pd.to_numeric(raw["RT_critical"], errors="coerce") < 0).sum()),
        "cleaned_input_exactly_reconciled": True,
        "combined_table_rows": {
            "subject_summary": int(len(subject_summary)),
            "task_summary": int(len(task_summary)),
            "model_chain_stats": int(len(chain_stats)),
            "posterior_predictive_stats": int(len(ppc_stats)),
            "gelman_rubin": int(len(convergence)),
            "model_runs": int(len(model_runs)),
            "regression_posterior_summary": int(len(posterior_summary)),
            "hypothesis_probabilities": int(len(probabilities)),
            "effect_sizes": int(len(effect_sizes)),
            "original_model_file_index": int(len(file_index)),
        },
    }
    (OUTPUT_DIR / "package_summary.json").write_text(
        json.dumps(package_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = f"""# RT_critical complete data package

This package collects the trial-level human data and the tabular outputs used by the RT_critical HDDM analysis.

## Reconciliation

- Raw human trials: {len(raw):,}
- HDDM-included trials: {included.sum():,}
- Excluded trials: {(~included).sum():,} ({(~included).mean():.3%})
- Unique participant IDs: {raw['Subject'].nunique():,}
- Missing RT_critical values: {pd.to_numeric(raw['RT_critical'], errors='coerce').isna().sum():,}
- Negative RT_critical values: {(pd.to_numeric(raw['RT_critical'], errors='coerce') < 0).sum():,}
- The reconstructed input exactly matches the saved HDDM cleaned dataset.

## Key files

- `RT_critical_complete_data.xlsx`: formatted workbook with raw, processed, excluded, summary, and model-result sheets.
- `csv/01_human_raw_original_all_columns.csv`: untouched copy of the original merged human trial-level CSV.
- `csv/02_human_rt_critical_with_inclusion_flags.csv`: all human trials with task labels and HDDM inclusion flags.
- `csv/03_rt_critical_hddm_input.csv`: exact trial-level input used for the RT_critical HDDM models.
- `csv/04_rt_critical_excluded_trials.csv`: all excluded trials and explicit exclusion reasons.
- `csv/05_rt_critical_subject_summary.csv`: participant-by-task summary.
- `csv/06_rt_critical_task_summary.csv`: task-level summary.
- `csv/07` through `csv/14`: combined HDDM parameter statistics, posterior predictive checks, convergence diagnostics, model metadata, posterior summaries, effect sizes, and a complete original-file index.
- `original_analysis_tables/`: copies of every CSV and JSON file under the original RT_critical results directory.

## Task labels

- `categorization` / `Semantic` = Concept Verification
- `Voe` / `Intuitive` = Plausibility Assessment
- `sensorimotor` / `Action` = Affordance Recognition

## Scope

Full MCMC trace databases and serialized model binaries are not duplicated in this package because the existing RT_critical result directory is approximately 2.0 GB. Their exact paths and sizes are listed in `csv/14_original_model_file_index.csv`; the files remain available under `{RESULTS_DIR}`.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    write_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Refresh checksums after the workbook has been created.",
    )
    args = parser.parse_args()
    if args.refresh_manifest:
        write_manifest()
    else:
        prepare_package()


if __name__ == "__main__":
    main()
