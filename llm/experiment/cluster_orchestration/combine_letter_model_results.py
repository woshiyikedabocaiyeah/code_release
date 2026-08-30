#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SETTINGS = {
    "non_embodied": "final_non_embodied_results.json",
    "embodied": "final_embodied_results.json",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return list(payload.get("results", []))


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("question_id")),
        str(row.get("model")),
        str(row.get("mode")),
        str(row.get("prompt_type")),
    )


def build_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('model')}_{row.get('mode')}_{row.get('prompt_type')}"
        bucket = stats.setdefault(key, {"total": 0, "correct": 0, "success": 0, "missing_answer": 0, "inference_times": []})
        bucket["total"] += 1
        if row.get("success"):
            bucket["success"] += 1
        if row.get("correct"):
            bucket["correct"] += 1
        if row.get("model_answer") in (None, ""):
            bucket["missing_answer"] += 1
        if isinstance(row.get("inference_time_seconds"), (int, float)):
            bucket["inference_times"].append(float(row["inference_time_seconds"]))

    output = {}
    for key, value in stats.items():
        total = value["total"]
        times = value["inference_times"]
        output[key] = {
            "total": total,
            "success": value["success"],
            "correct": value["correct"],
            "missing_answer": value["missing_answer"],
            "accuracy_percent": round(value["correct"] / total * 100, 2) if total else 0.0,
            "avg_inference_time_seconds": round(sum(times) / len(times), 3) if times else None,
        }
    return output


def combine_setting(source_root: Path, output_root: Path, setting: str, filename: str) -> dict[str, Any]:
    source_files = sorted(source_root.glob(f"*/{setting}/{filename}"))
    rows_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for source_file in source_files:
        for row in read_rows(source_file):
            rows_by_key[row_key(row)] = row
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    output_path = output_root / setting / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_info": {
            "combined_at": datetime.now().isoformat(),
            "source_root": str(source_root),
            "source_files": [str(path) for path in source_files],
            "total_results": len(rows),
            "model_statistics": build_stats(rows),
        },
        "results": rows,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return {
        "setting": setting,
        "output": str(output_path),
        "source_files": len(source_files),
        "total_results": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-model AB/BA letter-choice experiment outputs.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary", help="Optional summary JSON path.")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    summary = [combine_setting(source_root, output_root, setting, filename) for setting, filename in SETTINGS.items()]
    summary_path = Path(args.summary) if args.summary else output_root / "combine_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump({"combined_at": datetime.now().isoformat(), "settings": summary}, file, indent=2, ensure_ascii=False)
    print(f"Combined outputs written under {output_root}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
