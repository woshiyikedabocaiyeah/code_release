#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def model_family(model: str) -> str:
    if model.startswith("GLM-4.1V"):
        return "GLM-4.1V"
    if model.startswith("Qwen"):
        return "Qwen"
    if model.startswith("RynnBrain"):
        return "RynnBrain"
    return model


def mode_alias(model: str, mode: str) -> str:
    if model == "RynnBrain-CoP":
        return "think"
    return mode


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def pstdev(values: list[float]) -> float:
    return round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0


def accuracy(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    total = len(rows)
    correct = sum(1 for row in rows if boolish(row.get("correct")))
    percent = round(correct / total * 100, 4) if total else 0.0
    return total, correct, percent


def answer_value(row: dict[str, Any]) -> str:
    for key in ("model_answer", "semantic_answer", "original_letter_parser_answer"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def prompt_setting(prompt_type: str) -> str:
    return "embodied" if prompt_type.startswith("embodied") else "non_embodied"


def load_repeat_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat_dir in sorted(run_root.glob("repeat_*")):
        result_path = repeat_dir / "results.json"
        if not result_path.exists():
            continue
        payload = read_json(result_path)
        try:
            repeat_index = int(repeat_dir.name.rsplit("_", 1)[1])
        except Exception:
            repeat_index = len({row.get("repeat_index") for row in rows}) + 1
        for row in payload.get("results", []):
            item = dict(row)
            model = str(item.get("model") or "")
            prompt_type = str(item.get("prompt_type") or "")
            item["repeat_index"] = repeat_index
            item["model_family"] = model_family(model)
            item["mode_alias"] = mode_alias(model, str(item.get("mode") or ""))
            item["setting"] = prompt_setting(prompt_type)
            rows.append(item)
    if not rows:
        raise FileNotFoundError(f"No repeat result rows found under {run_root}")
    return rows


def repeat_inventory(run_root: Path, analysis_root: Path) -> dict[str, Any]:
    repeats = []
    for repeat_dir in sorted(run_root.glob("repeat_*")):
        result_path = repeat_dir / "results.json"
        if not result_path.exists():
            continue
        payload = read_json(result_path)
        try:
            repeat_index = int(repeat_dir.name.rsplit("_", 1)[1])
        except Exception:
            repeat_index = len(repeats) + 1
        repeats.append(
            {
                "repeat_index": repeat_index,
                "result_path": str(result_path),
                "row_count": len(payload.get("results", [])),
            }
        )
    return {
        "run_root": str(run_root),
        "analysis_root": str(analysis_root),
        "manifest_path": str(run_root / "fixed_subset_manifest.json"),
        "repeats": repeats,
    }


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def build_repeat_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = []
    for key, group in group_rows(rows, ("repeat_index", "model_family", "mode_alias", "prompt_type")).items():
        repeat_index, family, mode, prompt = key
        total, correct, percent = accuracy(group)
        all_rows.append(
            {
                "aggregation_scope": "all_question_types",
                "repeat_index": repeat_index,
                "model_family": family,
                "mode_alias": mode,
                "prompt_type": prompt,
                "total_count": total,
                "correct_count": correct,
                "accuracy_percent": percent,
                "question_ids": sorted(str(row.get("question_id")) for row in group),
            }
        )

    type_rows = []
    for key, group in group_rows(rows, ("repeat_index", "question_type", "model_family", "mode_alias", "prompt_type")).items():
        repeat_index, question_type, family, mode, prompt = key
        total, correct, percent = accuracy(group)
        type_rows.append(
            {
                "aggregation_scope": "question_type",
                "repeat_index": repeat_index,
                "question_type": question_type,
                "model_family": family,
                "mode_alias": mode,
                "prompt_type": prompt,
                "total_count": total,
                "correct_count": correct,
                "accuracy_percent": percent,
                "question_ids": sorted(str(row.get("question_id")) for row in group),
            }
        )
    return sorted(all_rows, key=lambda row: (row["model_family"], row["mode_alias"], row["prompt_type"], row["repeat_index"])), sorted(
        type_rows,
        key=lambda row: (row["question_type"], row["model_family"], row["mode_alias"], row["prompt_type"], row["repeat_index"]),
    )


def build_item_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repeat_indices = sorted({int(row["repeat_index"]) for row in rows})
    output = []

    for key, group in group_rows(rows, ("model_family", "mode_alias", "prompt_type", "question_id")).items():
        family, mode, prompt, question_id = key
        by_repeat = {int(row["repeat_index"]): answer_value(row) for row in group}
        answers = [by_repeat.get(index, "") for index in repeat_indices]
        nonempty = [answer for answer in answers if answer != ""]
        output.append(
            {
                "aggregation_scope": "all_question_types",
                "model_family": family,
                "mode_alias": mode,
                "prompt_type": prompt,
                "question_id": question_id,
                "answers_by_repeat": answers,
                "all_equal": len(nonempty) == len(repeat_indices) and len(set(nonempty)) == 1,
            }
        )

    for key, group in group_rows(rows, ("question_type", "model_family", "mode_alias", "prompt_type", "question_id")).items():
        question_type, family, mode, prompt, question_id = key
        by_repeat = {int(row["repeat_index"]): answer_value(row) for row in group}
        answers = [by_repeat.get(index, "") for index in repeat_indices]
        nonempty = [answer for answer in answers if answer != ""]
        output.append(
            {
                "aggregation_scope": "question_type",
                "question_type": question_type,
                "model_family": family,
                "mode_alias": mode,
                "prompt_type": prompt,
                "question_id": question_id,
                "answers_by_repeat": answers,
                "all_equal": len(nonempty) == len(repeat_indices) and len(set(nonempty)) == 1,
            }
        )

    return sorted(
        output,
        key=lambda row: (
            row["aggregation_scope"],
            row.get("question_type", ""),
            row["model_family"],
            row["mode_alias"],
            row["prompt_type"],
            row["question_id"],
        ),
    )


def consistency_lookup(item_rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        if row["aggregation_scope"] == "all_question_types":
            key = ("all_question_types", row["model_family"], row["mode_alias"], row["prompt_type"])
        else:
            key = ("question_type", row["question_type"], row["model_family"], row["mode_alias"], row["prompt_type"])
        grouped[key].append(row)
    return {
        key: round(sum(1 for row in values if row["all_equal"]) / len(values) * 100, 4) if values else 0.0
        for key, values in grouped.items()
    }


def build_summaries(
    repeat_metrics_all: list[dict[str, Any]],
    repeat_metrics_by_type: list[dict[str, Any]],
    item_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = consistency_lookup(item_rows)
    all_summary = []
    for key, metrics in group_rows(repeat_metrics_all, ("model_family", "mode_alias", "prompt_type")).items():
        family, mode, prompt = key
        metrics = sorted(metrics, key=lambda row: row["repeat_index"])
        accuracies = [row["accuracy_percent"] for row in metrics]
        all_summary.append(
            {
                "aggregation_scope": "all_question_types",
                "model_family": family,
                "mode_alias": mode,
                "prompt_type": prompt,
                "repeat_count": len(metrics),
                "questions_per_repeat": metrics[0]["total_count"] if metrics else 0,
                "repeat_accuracies_percent": accuracies,
                "mean_accuracy_percent": mean(accuracies),
                "std_accuracy_percent": pstdev(accuracies),
                "item_consistency_percent": lookup.get(("all_question_types", family, mode, prompt), 0.0),
            }
        )

    type_summary = []
    for key, metrics in group_rows(repeat_metrics_by_type, ("question_type", "model_family", "mode_alias", "prompt_type")).items():
        question_type, family, mode, prompt = key
        metrics = sorted(metrics, key=lambda row: row["repeat_index"])
        accuracies = [row["accuracy_percent"] for row in metrics]
        type_summary.append(
            {
                "aggregation_scope": "question_type",
                "question_type": question_type,
                "model_family": family,
                "mode_alias": mode,
                "prompt_type": prompt,
                "repeat_count": len(metrics),
                "questions_per_repeat": metrics[0]["total_count"] if metrics else 0,
                "repeat_accuracies_percent": accuracies,
                "mean_accuracy_percent": mean(accuracies),
                "std_accuracy_percent": pstdev(accuracies),
                "item_consistency_percent": lookup.get(("question_type", question_type, family, mode, prompt), 0.0),
            }
        )

    return sorted(all_summary, key=lambda row: (row["model_family"], row["mode_alias"], row["prompt_type"])), sorted(
        type_summary,
        key=lambda row: (row["question_type"], row["model_family"], row["mode_alias"], row["prompt_type"]),
    )


def build_question_inventory(question_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for question in question_payload.get("questions", []):
        metadata = question.get("metadata") or {}
        rows.append(
            {
                "question_id": question.get("id"),
                "video_id": question.get("video_id"),
                "question_type": question.get("question_type"),
                "physical_concept": metadata.get("physical_concept") or metadata.get("condition"),
                "possibility": metadata.get("possibility"),
                "camera": metadata.get("camera"),
                "scene_template": metadata.get("game_name"),
                "occlusion": metadata.get("occlusion"),
                "difficulty": metadata.get("difficulty"),
                "correct_answer": question.get("correct_answer"),
            }
        )
    return rows


def build_overviews(all_summary: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_rows = []
    for key, rows in group_rows(all_summary, ("model_family", "mode_alias")).items():
        family, mode = key
        model_rows.append(
            {
                "model_family": family,
                "mode_alias": mode,
                "prompt_count": len(rows),
                "mean_accuracy_percent": mean([row["mean_accuracy_percent"] for row in rows]),
                "mean_std_accuracy_percent": mean([row["std_accuracy_percent"] for row in rows]),
                "mean_item_consistency_percent": mean([row["item_consistency_percent"] for row in rows]),
            }
        )

    prompt_rows = []
    for key, rows in group_rows(all_summary, ("prompt_type",)).items():
        (prompt,) = key
        prompt_rows.append(
            {
                "prompt_type": prompt,
                "combo_count": len(rows),
                "mean_accuracy_percent": mean([row["mean_accuracy_percent"] for row in rows]),
                "mean_std_accuracy_percent": mean([row["std_accuracy_percent"] for row in rows]),
                "mean_item_consistency_percent": mean([row["item_consistency_percent"] for row in rows]),
            }
        )
    return sorted(model_rows, key=lambda row: (row["model_family"], row["mode_alias"])), sorted(prompt_rows, key=lambda row: row["prompt_type"])


def issue_rows(rows: list[dict[str, Any]], expected_repeats: int) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["repeat_count"] != expected_repeats
        or abs(float(row["std_accuracy_percent"])) > 1e-9
        or float(row["item_consistency_percent"]) < 100.0
    ]


def long_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "repeat_index",
        "setting",
        "model",
        "model_family",
        "mode",
        "mode_alias",
        "prompt_type",
        "question_id",
        "video_id",
        "question_type",
        "model_answer",
        "semantic_answer",
        "correct",
        "success",
        "format_compliant_yesno",
        "answer_parse_status",
        "metadata_path",
        "artifacts_output_dir",
    ]
    return [{field: row.get(field) for field in fields} for row in sorted(rows, key=lambda r: tuple(str(r.get(f, "")) for f in fields[:8]))]


def write_readmes(
    module_root: Path,
    run_root: Path,
    all_summary: list[dict[str, Any]],
    type_summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    expected_repeats: int,
) -> None:
    data_dir = module_root / "Data"
    report_dir = module_root / "Report"
    all_zero = sum(1 for row in all_summary if abs(row["std_accuracy_percent"]) <= 1e-9)
    type_zero = sum(1 for row in type_summary if abs(row["std_accuracy_percent"]) <= 1e-9)
    all_consistent = sum(1 for row in all_summary if row["item_consistency_percent"] == 100.0)
    type_consistent = sum(1 for row in type_summary if row["item_consistency_percent"] == 100.0)
    success_rows = sum(1 for row in rows if boolish(row.get("success")))
    compliant_rows = sum(1 for row in rows if boolish(row.get("format_compliant_yesno")))
    compliance = round(compliant_rows / success_rows * 100, 4) if success_rows else 0.0

    top_combo = max(all_summary, key=lambda row: row["mean_accuracy_percent"]) if all_summary else {}
    question_type_means = {
        key[0]: mean([row["mean_accuracy_percent"] for row in grouped])
        for key, grouped in group_rows(type_summary, ("question_type",)).items()
    }
    prompt_means = {
        key[0]: mean([row["mean_accuracy_percent"] for row in grouped])
        for key, grouped in group_rows(all_summary, ("prompt_type",)).items()
    }

    (module_root / "README.md").write_text(
        "\n".join(
            [
                "# Sampling Stability",
                "",
                "Repeat-stability experiment on a fixed subset of the yes/no items.",
                "",
                "## The run this package describes",
                "",
                f"- Run directory: `{run_root}`",
                "- Design: `4` videos drawn from the full stimulus set, all three question types retained, giving `12` fixed items",
                f"- Repeats: each cell is run `{expected_repeats}` times",
                "- Generation: temperature `0`, fixed sampling and parsing; raw JSON and embeddings are written to `outputs/`",
                "",
                "## Contents",
                "",
                "- `Data/`: aggregated outputs, the fixed item list, the per-item long table and the overview tables",
                "- `Report/`: the descriptive and interpretive reports",
                "",
                "## Summary",
                "",
                f"- Of `{len(all_summary)}` all-question cells, `{all_zero}` have an accuracy standard deviation of `0`",
                f"- Of `{len(type_summary)}` question-type cells, `{type_zero}` have an accuracy standard deviation of `0`",
                f"- All-question cells with `100%` item-level consistency: `{all_consistent} / {len(all_summary)}`",
                f"- Question-type cells with `100%` item-level consistency: `{type_consistent} / {len(type_summary)}`",
                f"- yes/no format compliance: `{compliance:.4f}%`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (data_dir / "README.md").write_text(
        "\n".join(
            [
                "# Sampling Stability Data",
                "",
                "Aggregated data, the fixed item list and the overview tables for the repeat-stability experiment.",
                "",
                "## Files",
                "",
                "- `00_sampling_file_inventory.csv`: file inventory for this module",
                "- `01_repeat_inventory.json`: index of the per-repeat result files",
                "- `02_accuracy_summary_all_questions.csv` / `03_accuracy_summary_all_questions.json`: stability summary at the all-question level",
                "- `04_accuracy_summary_by_question_type.csv` / `05_accuracy_summary_by_question_type.json`: stability summary at the question-type level",
                "- `06_repeat_metrics_all_questions.json`: all-question metrics for each repeat",
                "- `07_repeat_metrics_by_question_type.json`: question-type metrics for each repeat",
                "- `08_item_consistency.json`: item-level answer consistency",
                "- `09_fixed_subset_manifest.json`: the fixed video subset",
                "- `10_fixed_subset_questions.json`: the fixed subset items",
                "- `11_fixed_subset_question_inventory.csv`: tabular form of the fixed subset items",
                "- `12_combo_issue_summary.csv`: all-question cells flagged as unstable",
                "- `13_question_type_issue_summary.csv`: question-type cells flagged as unstable",
                "- `14_model_mode_overview.csv`: summary by model family and mode",
                "- `15_prompt_overview.csv`: summary by prompt condition",
                "- `16_all_results_long.csv`: long table by repeat, item and model",
                "",
                f"Run directory: `{run_root}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    best_qtype = max(question_type_means.items(), key=lambda item: item[1]) if question_type_means else ("NA", 0.0)
    worst_qtype = min(question_type_means.items(), key=lambda item: item[1]) if question_type_means else ("NA", 0.0)
    best_prompt = max(prompt_means.items(), key=lambda item: item[1]) if prompt_means else ("NA", 0.0)
    worst_prompt = min(prompt_means.items(), key=lambda item: item[1]) if prompt_means else ("NA", 0.0)

    (report_dir / "01_data_summary.md").write_text(
        "\n".join(
            [
                "# Sampling stability: data summary",
                "",
                "## 1. Purpose",
                "",
                f"This report covers the fixed-subset repeat-stability experiment, which checks whether accuracy and item-level answers stay identical when the same `model x mode x prompt` cell is run `{expected_repeats}` times.",
                "",
                "## 2. Sources",
                "",
                f"- Run directory: `{run_root}`",
                f"- Module directory: `{module_root}`",
                "",
                "## 3. Design",
                "",
                "- Sampling: fixed random seed `20260328`, `4` video_ids drawn from the full stimulus set",
                "- Items: `SM`, `VoE` and `Category` questions for each video, giving `12` fixed items",
                f"- Repeats: each cell is run `{expected_repeats}` times",
                "- Cell definition: all-question level is `model_family x mode_alias x prompt_type`; question-type level is `question_type x model_family x mode_alias x prompt_type`",
                "",
                "## 4. Statistics",
                "",
                "### 4.1 All-question level",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Cells | `{len(all_summary)}` |",
                f"| Mean accuracy s.d. | `{mean([row['std_accuracy_percent'] for row in all_summary]):.4f}%` |",
                f"| Cells with zero accuracy s.d. | `{all_zero} / {len(all_summary)}` |",
                f"| Cells with `100%` item consistency | `{all_consistent} / {len(all_summary)}` |",
                "",
                "### 4.2 Question-type level",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Cells | `{len(type_summary)}` |",
                f"| Mean accuracy s.d. | `{mean([row['std_accuracy_percent'] for row in type_summary]):.4f}%` |",
                f"| Cells with zero accuracy s.d. | `{type_zero} / {len(type_summary)}` |",
                f"| Cells with `100%` item consistency | `{type_consistent} / {len(type_summary)}` |",
                "",
                "### 4.3 yes/no format compliance",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Successful result rows | `{success_rows}` |",
                f"| Rows answering with a bare yes/no | `{compliant_rows}` |",
                f"| Compliance rate | `{compliance:.4f}%` |",
                "",
                "## 5. Performance structure",
                "",
                f"- Highest-scoring all-question cell: `{top_combo.get('model_family', 'NA')} | {top_combo.get('mode_alias', 'NA')} | {top_combo.get('prompt_type', 'NA')}`, mean accuracy `{top_combo.get('mean_accuracy_percent', 0.0):.4f}%`.",
                f"- Best question type on the fixed subset: `{best_qtype[0]}` at `{best_qtype[1]:.4f}%`; worst: `{worst_qtype[0]}` at `{worst_qtype[1]:.4f}%`.",
                f"- Best prompt condition on the fixed subset: `{best_prompt[0]}` at `{best_prompt[1]:.4f}%`; worst: `{worst_prompt[0]}` at `{worst_prompt[1]:.4f}%`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (report_dir / "02_interpretation_summary.md").write_text(
        "\n".join(
            [
                "# Sampling stability: interpretation",
                "",
                "## 1. The question",
                "",
                "Whether model output drifts between repeated runs on the same fixed videos and questions, under deterministic generation settings.",
                "",
                "## 2. Result",
                "",
                f"- Of `{len(all_summary)}` all-question cells, `{all_zero}` have an accuracy standard deviation of `0`",
                f"- Of `{len(type_summary)}` question-type cells, `{type_zero}` have an accuracy standard deviation of `0`",
                f"- All-question cells with `100%` item-level consistency: `{all_consistent} / {len(all_summary)}`",
                f"- Question-type cells with `100%` item-level consistency: `{type_consistent} / {len(type_summary)}`",
                "",
                "## 3. Implication",
                "",
                "These results support the protocol used for the main experiment. If repeated runs on fixed inputs are stable, the residual risk in the full dataset is not sampling-induced answer drift but whether each trial completed, whether its answer parsed correctly, and whether failed items were correctly re-run.",
                "",
                "## 4. Suggested wording",
                "",
                f"In the fixed-subset repeat experiment we drew 4 videos from the full stimulus set and retained all three task types, giving 12 items. Each `model x mode x prompt` cell was run {expected_repeats} times, and we report the standard deviation of accuracy and the item-level answer consistency.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_run_readme(run_root: Path, module_root: Path, rows: list[dict[str, Any]], expected_repeats: int) -> None:
    repeat_dirs = sorted(path.name for path in run_root.glob("repeat_*") if path.is_dir())
    metadata_files = sum(1 for _ in run_root.glob("repeat_*/*/*_artifacts/**/*.json"))
    embedding_files = sum(1 for _ in run_root.glob("repeat_*/*/*_artifacts/**/*.npy"))
    (run_root / "README.md").write_text(
        "\n".join(
            [
                "# Sampling Stability Main Experiment",
                "",
                "This directory contains the raw paper-level semantic Yes/No sampling stability run.",
                "",
                "## Protocol",
                "",
                "- Fixed seed: `20260328`",
                "- Fixed subset: 4 videos, 12 questions total",
                f"- Repeats: `{expected_repeats}`",
                "- Prompt types: `simple`, `detail`, `embodied_simple`, `embodied_detail`",
                "- Models/modes: all recovered target configurations from `model_config.py`",
                "",
                "## Outputs",
                "",
                f"- Repeat directories: `{len(repeat_dirs)}`",
                f"- Result rows: `{len(rows)}`",
                f"- Metadata JSON files: `{metadata_files}`",
                f"- Embedding `.npy` files: `{embedding_files}`",
                "",
                "## Analysis",
                "",
                f"The summarized CSV/JSON files and reports are in `{module_root}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_inventory(module_root: Path) -> None:
    rows = []
    for path in sorted(module_root.rglob("*")):
        if path.is_file() and path.name != "00_sampling_file_inventory.csv":
            rows.append({"path": str(path.relative_to(module_root)), "bytes": path.stat().st_size})
    write_csv(module_root / "Data" / "00_sampling_file_inventory.csv", rows, ["path", "bytes"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the semantic yes/no sampling stability experiment.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    module_root = Path(args.analysis_root).resolve()
    data_dir = module_root / "Data"
    report_dir = module_root / "Report"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = load_repeat_rows(run_root)
    repeats = sorted({int(row["repeat_index"]) for row in rows})
    expected_repeats = len(repeats)
    repeat_metrics_all, repeat_metrics_by_type = build_repeat_metrics(rows)
    item_rows = build_item_consistency(rows)
    all_summary, type_summary = build_summaries(repeat_metrics_all, repeat_metrics_by_type, item_rows)
    model_overview, prompt_overview = build_overviews(all_summary)

    manifest_path = run_root / "fixed_subset_manifest.json"
    questions_path = run_root / "fixed_subset_questions.json"
    question_payload = read_json(questions_path)
    manifest_payload = read_json(manifest_path)

    write_json(data_dir / "01_repeat_inventory.json", repeat_inventory(run_root, module_root))
    write_csv(
        data_dir / "02_accuracy_summary_all_questions.csv",
        all_summary,
        [
            "aggregation_scope",
            "model_family",
            "mode_alias",
            "prompt_type",
            "repeat_count",
            "questions_per_repeat",
            "repeat_accuracies_percent",
            "mean_accuracy_percent",
            "std_accuracy_percent",
            "item_consistency_percent",
        ],
    )
    write_json(data_dir / "03_accuracy_summary_all_questions.json", {"rows": all_summary})
    write_csv(
        data_dir / "04_accuracy_summary_by_question_type.csv",
        type_summary,
        [
            "aggregation_scope",
            "question_type",
            "model_family",
            "mode_alias",
            "prompt_type",
            "repeat_count",
            "questions_per_repeat",
            "repeat_accuracies_percent",
            "mean_accuracy_percent",
            "std_accuracy_percent",
            "item_consistency_percent",
        ],
    )
    write_json(data_dir / "05_accuracy_summary_by_question_type.json", {"rows": type_summary})
    write_json(data_dir / "06_repeat_metrics_all_questions.json", {"rows": repeat_metrics_all})
    write_json(data_dir / "07_repeat_metrics_by_question_type.json", {"rows": repeat_metrics_by_type})
    write_json(data_dir / "08_item_consistency.json", {"rows": item_rows})
    write_json(data_dir / "09_fixed_subset_manifest.json", manifest_payload)
    write_json(data_dir / "10_fixed_subset_questions.json", question_payload)
    write_csv(
        data_dir / "11_fixed_subset_question_inventory.csv",
        build_question_inventory(question_payload),
        [
            "question_id",
            "video_id",
            "question_type",
            "physical_concept",
            "possibility",
            "camera",
            "scene_template",
            "occlusion",
            "difficulty",
            "correct_answer",
        ],
    )
    write_csv(
        data_dir / "12_combo_issue_summary.csv",
        issue_rows(all_summary, expected_repeats),
        [
            "aggregation_scope",
            "model_family",
            "mode_alias",
            "prompt_type",
            "repeat_count",
            "questions_per_repeat",
            "repeat_accuracies_percent",
            "mean_accuracy_percent",
            "std_accuracy_percent",
            "item_consistency_percent",
        ],
    )
    write_csv(
        data_dir / "13_question_type_issue_summary.csv",
        issue_rows(type_summary, expected_repeats),
        [
            "aggregation_scope",
            "question_type",
            "model_family",
            "mode_alias",
            "prompt_type",
            "repeat_count",
            "questions_per_repeat",
            "repeat_accuracies_percent",
            "mean_accuracy_percent",
            "std_accuracy_percent",
            "item_consistency_percent",
        ],
    )
    write_csv(
        data_dir / "14_model_mode_overview.csv",
        model_overview,
        [
            "model_family",
            "mode_alias",
            "prompt_count",
            "mean_accuracy_percent",
            "mean_std_accuracy_percent",
            "mean_item_consistency_percent",
        ],
    )
    write_csv(
        data_dir / "15_prompt_overview.csv",
        prompt_overview,
        [
            "prompt_type",
            "combo_count",
            "mean_accuracy_percent",
            "mean_std_accuracy_percent",
            "mean_item_consistency_percent",
        ],
    )
    write_csv(
        data_dir / "16_all_results_long.csv",
        long_rows(rows),
        [
            "repeat_index",
            "setting",
            "model",
            "model_family",
            "mode",
            "mode_alias",
            "prompt_type",
            "question_id",
            "video_id",
            "question_type",
            "model_answer",
            "semantic_answer",
            "correct",
            "success",
            "format_compliant_yesno",
            "answer_parse_status",
            "metadata_path",
            "artifacts_output_dir",
        ],
    )
    write_readmes(module_root, run_root, all_summary, type_summary, rows, expected_repeats)
    write_run_readme(run_root, module_root, rows, expected_repeats)
    write_inventory(module_root)
    print(f"Sampling stability analysis written to {module_root}")
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "analysis_root": str(module_root),
                "generated_at": datetime.now().isoformat(),
                "rows": len(rows),
                "repeats": expected_repeats,
                "all_question_combos": len(all_summary),
                "question_type_combos": len(type_summary),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
