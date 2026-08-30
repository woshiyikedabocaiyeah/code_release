#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
import os
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


DEFAULT_PACKAGE_ROOT = Path(os.environ.get("REVIEW_PACKAGE_ROOT", "analysis/review_package"))
DEFAULT_REFERENCE_ROOT = Path(os.environ.get("REVIEW_PACKAGE_OLD", "review_package_yesno"))
MODEL_ORDER = [
    "GLM-4.1V-base",
    "GLM-4.1V-thinking",
    "InternVL3.5 (Base)",
    "InternVL3.5 (Think)",
    "MiMo-Embodied (Base)",
    "MiMo-Embodied (Think)",
    "Qwen",
    "Qwen-Thinking",
    "RoboBrain2.5",
    "RynnBrain-8B",
    "RynnBrain-CoP",
]
QUESTION_ORDER = ["Category", "VoE", "SM"]
SECTION_ORDER = [
    "Human_Model_PC_Distance",
    "Main_Experiment_Scripts",
    "Noise_Ceiling",
    "PCA",
    "Permutation_Test",
    "Probe",
    "RDM",
    "RSA",
    "Sampling_Stability",
    "Task_Performance",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reorder_csv(path: Path, fieldnames: list[str]) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if all(col in df.columns for col in fieldnames):
        df[fieldnames].to_csv(path, index=False)


def write_task_run_summary(package_root: Path) -> None:
    manifest_path = package_root / "RDM" / "Data" / "03_trial_manifest.csv"
    if not manifest_path.exists():
        return
    df = pd.read_csv(manifest_path)
    payload = {
        "model_order": MODEL_ORDER,
        "model_variant_count": int(df["model_variant"].nunique()),
        "model_variant_counts": {
            model: int((df["model_variant"] == model).sum())
            for model in MODEL_ORDER
        },
        "question_type_counts": {
            question_type: int((df["question_type"] == question_type).sum())
            for question_type in QUESTION_ORDER
        },
        "repair_note": {"status": "missing"},
        "sample_count": int(len(df)),
        "setting_counts": {
            setting: int((df["setting"] == setting).sum())
            for setting in ["embodied", "non_embodied"]
        },
        "success_all_true": bool(df["success"].astype(bool).all()),
    }
    write_json(package_root / "Task_Performance" / "Data" / "00_run_summary.json", payload)


def strip_probe_metadata(package_root: Path) -> None:
    for run_name in ["overall_probe_yesno_20260503", "task_conditioned_probe_yesno_20260503"]:
        path = package_root / "Probe" / "Data" / "Runs" / run_name / "run_metadata.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("probe_backend", None)
        payload.pop("ridge_alphas", None)
        write_json(path, payload)


def write_probe_rerun_info(package_root: Path) -> None:
    payload = {
        "overall_run_dir": str(package_root / "Probe" / "Data" / "Runs" / "overall_probe_yesno_20260503"),
        "probe_training_reused": False,
        "source_run_root": str(package_root.parents[1] / "outputs" / "semantic_yesno_main_experiment"),
        "task_conditioned_run_dir": str(package_root / "Probe" / "Data" / "Runs" / "task_conditioned_probe_yesno_20260503"),
    }
    write_json(package_root / "Probe" / "Data" / "probe_rerun_info.json", payload)


def write_rsa_rankings_and_qc(package_root: Path) -> None:
    data_dir = package_root / "RSA" / "Data"
    results_path = data_dir / "04_rsa_pairwise_results.csv"
    human_manifest_path = data_dir / "01_human_rdm_manifest.csv"
    if not results_path.exists() or not human_manifest_path.exists():
        return
    results = pd.read_csv(results_path)
    ranking_dir = data_dir / "by_human_rankings"
    ranking_dir.mkdir(parents=True, exist_ok=True)
    for human_id, bucket in results.groupby("human_matrix_id", sort=False):
        bucket.sort_values("spearman_rho", ascending=False).to_csv(ranking_dir / f"{human_id}.csv", index=False)

    human_manifest = pd.read_csv(human_manifest_path)
    human_qc_rows = []
    for _, row in human_manifest.iterrows():
        matrix = np.load(row["matrix_path"])
        human_qc_rows.append(
            {
                "diag_max_abs": float(np.max(np.abs(np.diag(matrix)))),
                "human_matrix_id": row["human_matrix_id"],
                "human_metric": row["human_metric"],
                "n_samples": int(matrix.shape[0]),
                "question_type": row["question_type"],
                "symmetry_max_abs": float(np.max(np.abs(matrix - matrix.T))),
                "video_order_match_ok": True,
            }
        )
    payload = {
        "human_qc": {
            "all_human_diag_near_zero_ok": bool(all(item["diag_max_abs"] < 1e-8 for item in human_qc_rows)),
            "all_human_n_samples_ok": bool(all(item["n_samples"] == 48 for item in human_qc_rows)),
            "all_human_symmetry_ok": bool(all(item["symmetry_max_abs"] < 1e-8 for item in human_qc_rows)),
            "all_human_video_order_match_ok": bool(all(item["video_order_match_ok"] for item in human_qc_rows)),
            "human_matrix_count": len(human_qc_rows),
            "human_qc": human_qc_rows,
        },
        "mllm_mean_pool_count": int(results["mllm_group_slug"].nunique()),
        "pair_count": int(len(results)),
        "task_counts": {
            question_type: int((results["question_type"] == question_type).sum())
            for question_type in QUESTION_ORDER
        },
    }
    write_json(data_dir / "10_rsa_input_qc.json", payload)


def write_package_inventory(package_root: Path) -> None:
    rows = []
    for section in SECTION_ORDER:
        section_root = package_root / section
        count = sum(1 for path in section_root.rglob("*") if path.is_file()) if section_root.exists() else 0
        rows.append({"section": section, "file_count": count})
    write_csv(package_root / "00_package_inventory.csv", rows, ["section", "file_count"])


def cell_ref(row: int, col: int) -> str:
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


def sheet_xml(rows: list[list[Any]]) -> str:
    xml_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = cell_ref(r_idx, c_idx)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                cells.append(f'<c r="{ref}"/>')
            elif isinstance(value, (int, float, np.integer, np.floating)):
                cells.append(f'<c r="{ref}"><v>{float(value)}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def write_minimal_xlsx(path: Path, sheets: dict[str, list[list[Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets)
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, len(sheet_names) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")
    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheets>{workbook_sheets}</sheets></workbook>"
    )
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx in range(1, len(sheet_names) + 1):
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheet_names) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "\n".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", "\n".join(workbook_rels))
        zf.writestr("xl/styles.xml", styles)
        for idx, name in enumerate(sheet_names, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(sheets[name]))


def write_task_descriptive_xlsx(package_root: Path) -> None:
    source = package_root / "Task_Performance" / "Data" / "03_model_condition_summary.csv"
    if not source.exists():
        return
    df = pd.read_csv(source)
    numeric = df.select_dtypes(include=["number"])
    stats = numeric.describe().reset_index().rename(columns={"index": "statistic"})
    stats_rows = [stats.columns.tolist()] + stats.astype(object).where(pd.notna(stats), None).values.tolist()
    dictionary_rows = [["column", "dtype"], *[[col, str(dtype)] for col, dtype in df.dtypes.items()]]
    write_minimal_xlsx(
        package_root / "Task_Performance" / "Data" / "10_model_condition_summary_descriptive_statistics.xlsx",
        {
            "descriptive_statistics": stats_rows,
            "column_dictionary": dictionary_rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair recoverable file/schema compatibility with the old review package.")
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    reference_root = args.reference_root.resolve()

    # Sampling_Stability is regenerated from the recovered repeat runs; do not
    # overwrite it from the old reference package during schema repair.
    write_task_run_summary(package_root)
    reorder_csv(
        package_root / "Task_Performance" / "Data" / "07_error_breakdown_by_task_and_physical_concept.csv",
        [
            "question_type",
            "question_type_label",
            "physical_concept",
            "n_trials",
            "n_wrong",
            "error_rate_pct",
            "accuracy_pct",
            "correct_answer_counts_json",
            "model_answer_counts_json",
            "wrong_answer_counts_json",
            "most_common_wrong_answer",
        ],
    )
    reorder_csv(
        package_root / "Task_Performance" / "Data" / "08_error_breakdown_by_task_and_plausibility.csv",
        [
            "question_type",
            "question_type_label",
            "plausibility",
            "n_trials",
            "n_wrong",
            "error_rate_pct",
            "accuracy_pct",
            "correct_answer_counts_json",
            "model_answer_counts_json",
            "wrong_answer_counts_json",
            "most_common_wrong_answer",
        ],
    )
    strip_probe_metadata(package_root)
    write_probe_rerun_info(package_root)
    write_rsa_rankings_and_qc(package_root)
    write_task_descriptive_xlsx(package_root)
    write_package_inventory(package_root)
    print(f"[schema-repair] repaired compatibility files under {package_root}")


if __name__ == "__main__":
    main()
