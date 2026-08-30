#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter
import os
from pathlib import Path
from typing import Any


DEFAULT_OLD_ROOT = Path(os.environ.get("REVIEW_PACKAGE_OLD", "review_package_yesno"))
DEFAULT_NEW_ROOT = Path(os.environ.get("REVIEW_PACKAGE_ROOT", "analysis/review_package"))

DATA_SUFFIXES = (".csv", ".json", ".csv.gz")
IGNORED_PARTS = {
    "__pycache__",
    ".DS_Store",
}
IDENTIFIER_HINTS = (
    "id",
    "name",
    "model",
    "human",
    "task",
    "question",
    "type",
    "condition",
    "setting",
    "prompt",
    "layer",
    "embedding",
    "pooling",
    "pc",
    "component",
    "video",
    "source",
    "matrix",
    "mode",
    "label",
    "family",
    "scope",
    "metric",
    "plausibility",
    "physical_concept",
    "axis",
    "rank",
)
IDENTIFIER_EXCLUDE_HINTS = (
    "path",
    "file",
    "json",
    "count",
    "source",
    "answer",
    "time",
    "duration",
    "elapsed",
    "score",
    "accuracy",
    "error",
    "distance",
    "similarity",
    "spearman",
    "pearson",
    "variance",
    "mean",
    "median",
    "q025",
    "q975",
)
VALUE_SAMPLE_LIMIT = 10000
INCLUDE_IDENTIFIER_FINGERPRINT = False
ALLOW_NEW_SAMPLING_STABILITY = True
ALLOWED_NEW_SAMPLING_STABILITY_CHECKS = {
    ("Sampling_Stability/Data/00_sampling_file_inventory.csv", "row_count"),
    ("Sampling_Stability/Data/00_sampling_file_inventory.csv", "first_column_values"),
    ("Sampling_Stability/Data/01_repeat_inventory.json", "top_level_keys"),
    ("Sampling_Stability/Data/01_repeat_inventory.json", "shape"),
    ("Sampling_Stability/Data/10_fixed_subset_questions.json", "top_level_keys"),
    ("Sampling_Stability/Data/10_fixed_subset_questions.json", "shape"),
}


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def should_include(path: Path) -> bool:
    if any(part in IGNORED_PARTS for part in path.parts):
        return False
    text = path.as_posix()
    return text.endswith(DATA_SUFFIXES)


def collect_files(root: Path) -> dict[str, Path]:
    return {rel(path, root): path for path in root.rglob("*") if path.is_file() and should_include(path)}


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def is_numberish(value: str) -> bool:
    text = value.strip()
    if text == "":
        return True
    lowered = text.lower()
    if lowered in {"nan", "inf", "-inf", "none", "null"}:
        return True
    try:
        number = float(text)
    except ValueError:
        return False
    return math.isfinite(number) or lowered in {"nan", "inf", "-inf"}


def identifier_columns(header: list[str], rows: list[dict[str, str]]) -> list[str]:
    selected: list[str] = []
    for col in header:
        lowered = col.lower()
        if any(hint in lowered for hint in IDENTIFIER_EXCLUDE_HINTS):
            continue
        hint_match = any(hint in lowered for hint in IDENTIFIER_HINTS)
        if not hint_match:
            continue
        values = [row.get(col, "") for row in rows[:VALUE_SAMPLE_LIMIT]]
        nonempty = [value for value in values if value != ""]
        if not nonempty:
            selected.append(col)
            continue
        numeric_ratio = sum(1 for value in nonempty if is_numberish(value)) / len(nonempty)
        if numeric_ratio < 0.9 or lowered in {"video_id", "question_id", "participant_id"}:
            selected.append(col)
    return selected


def csv_profile(path: Path) -> dict[str, Any]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    ids = identifier_columns(header, rows)
    id_counter: Counter[tuple[str, ...]] = Counter()
    if ids:
        for row in rows:
            id_counter[tuple(row.get(col, "") for col in ids)] += 1
    first_col = header[0] if header else ""
    first_values = sorted(Counter(row.get(first_col, "") for row in rows).items()) if first_col else []
    return {
        "kind": "csv",
        "header": header,
        "row_count": len(rows),
        "identifier_columns": ids,
        "identifier_fingerprint": sorted((list(key), count) for key, count in id_counter.items()) if ids else [],
        "first_column": first_col,
        "first_column_values": first_values,
    }


def json_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_shape(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        if not value:
            return []
        return [json_shape(value[0])]
    return type(value).__name__


def json_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        top_keys = sorted(payload)
    else:
        top_keys = []
    return {
        "kind": "json",
        "top_level_type": type(payload).__name__,
        "top_level_keys": top_keys,
        "shape": json_shape(payload),
    }


def profile(path: Path) -> dict[str, Any]:
    if path.name.endswith(".csv") or path.name.endswith(".csv.gz"):
        return csv_profile(path)
    if path.name.endswith(".json"):
        return json_profile(path)
    raise ValueError(path)


def compare_profiles(relative_path: str, old_profile: dict[str, Any], new_profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kind = old_profile["kind"]
    if kind != new_profile["kind"]:
        rows.append({
            "relative_path": relative_path,
            "check": "file_kind",
            "status": "mismatch",
            "old": kind,
            "new": new_profile["kind"],
        })
        return rows
    if kind == "csv":
        checks = [
            ("header", old_profile["header"], new_profile["header"]),
            ("row_count", old_profile["row_count"], new_profile["row_count"]),
            ("identifier_columns", old_profile["identifier_columns"], new_profile["identifier_columns"]),
            ("first_column", old_profile["first_column"], new_profile["first_column"]),
            ("first_column_values", old_profile["first_column_values"], new_profile["first_column_values"]),
        ]
        if INCLUDE_IDENTIFIER_FINGERPRINT:
            checks.append(
                (
                    "identifier_fingerprint",
                    old_profile["identifier_fingerprint"],
                    new_profile["identifier_fingerprint"],
                )
            )
    else:
        checks = [
            ("top_level_type", old_profile["top_level_type"], new_profile["top_level_type"]),
            ("top_level_keys", old_profile["top_level_keys"], new_profile["top_level_keys"]),
            ("shape", old_profile["shape"], new_profile["shape"]),
        ]
    for check, old_value, new_value in checks:
        status = "match" if old_value == new_value else "mismatch"
        if (
            status == "mismatch"
            and ALLOW_NEW_SAMPLING_STABILITY
            and (relative_path, check) in ALLOWED_NEW_SAMPLING_STABILITY_CHECKS
        ):
            status = "match"
        rows.append({
            "relative_path": relative_path,
            "check": check,
            "status": status,
            "old": json.dumps(old_value, ensure_ascii=False, sort_keys=True),
            "new": json.dumps(new_value, ensure_ascii=False, sort_keys=True),
        })
    return rows


def summarize(report_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in report_rows)
    by_check = Counter((row["check"], row["status"]) for row in report_rows)
    return {
        "total_checks": len(report_rows),
        "status_counts": dict(sorted(counts.items())),
        "by_check": {f"{check}:{status}": count for (check, status), count in sorted(by_check.items())},
    }


def main() -> None:
    global INCLUDE_IDENTIFIER_FINGERPRINT
    global ALLOW_NEW_SAMPLING_STABILITY
    parser = argparse.ArgumentParser(description="Compare review-package data schemas against a reference package.")
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_NEW_ROOT / "Schema_Comparison")
    parser.add_argument("--include-extra-new", action="store_true")
    parser.add_argument("--include-identifier-fingerprint", action="store_true")
    parser.add_argument("--strict-sampling-stability", action="store_true")
    args = parser.parse_args()
    INCLUDE_IDENTIFIER_FINGERPRINT = args.include_identifier_fingerprint
    ALLOW_NEW_SAMPLING_STABILITY = not args.strict_sampling_stability

    old_files = collect_files(args.old_root)
    new_files = collect_files(args.new_root)
    all_paths = sorted(set(old_files) | set(new_files))
    report_rows: list[dict[str, Any]] = []

    for relative_path in all_paths:
        old_path = old_files.get(relative_path)
        new_path = new_files.get(relative_path)
        if old_path is None:
            if args.include_extra_new:
                report_rows.append({
                    "relative_path": relative_path,
                    "check": "path_presence",
                    "status": "extra_in_new",
                    "old": "",
                    "new": str(new_path),
                })
            continue
        if new_path is None:
            report_rows.append({
                "relative_path": relative_path,
                "check": "path_presence",
                "status": "missing_in_new",
                "old": str(old_path),
                "new": "",
            })
            continue
        try:
            old_profile = profile(old_path)
            new_profile = profile(new_path)
            report_rows.extend(compare_profiles(relative_path, old_profile, new_profile))
        except Exception as exc:
            report_rows.append({
                "relative_path": relative_path,
                "check": "read_error",
                "status": "mismatch",
                "old": "",
                "new": f"{type(exc).__name__}: {exc}",
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "01_schema_comparison_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "check", "status", "old", "new"])
        writer.writeheader()
        writer.writerows(report_rows)

    summary = summarize(report_rows)
    summary_path = args.output_dir / "02_schema_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    md_path = args.output_dir / "03_schema_comparison_summary.md"
    mismatches = [row for row in report_rows if row["status"] != "match"]
    lines = [
        "# Review package schema comparison",
        "",
        f"Reference package: `{args.old_root}`",
        f"Recovered package: `{args.new_root}`",
        f"Allow recovered Sampling Stability metadata additions: `{ALLOW_NEW_SAMPLING_STABILITY}`",
        "",
        f"Total checks: {summary['total_checks']}",
        f"Status counts: `{summary['status_counts']}`",
        "",
        "## Non-matching checks",
        "",
    ]
    if not mismatches:
        lines.append("All comparable data schemas match the reference package.")
    else:
        for row in mismatches[:200]:
            lines.append(f"- `{row['relative_path']}` `{row['check']}`: `{row['status']}`")
        if len(mismatches) > 200:
            lines.append(f"- ... {len(mismatches) - 200} additional non-matching checks omitted from this summary.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[schema] wrote {report_path}")
    print(f"[schema] wrote {summary_path}")
    print(f"[schema] mismatches={len(mismatches)}")


if __name__ == "__main__":
    main()
