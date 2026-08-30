#!/usr/bin/env python
"""Prepare the CSV data for HDDM and write a small audit report.

This script intentionally uses only the Python standard library so it can run
before the HDDM scientific stack is installed.
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


GROUP_MAP = {
    "categorization": "Semantic",
    "sensorimotor": "Action",
    "Voe": "Voe",
}


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def prepare_rows(rows, rt_column, min_rt, max_rt):
    prepared = []
    skipped = Counter()

    for row in rows:
        try:
            rt = float(row[rt_column])
            response = int(float(row["ACC"]))
            visual_z = float(row["Visual_Z"])
            physical_z = float(row["Physical_Z"])
        except (TypeError, ValueError, KeyError):
            skipped["parse_error"] += 1
            continue

        if row.get("condition") not in GROUP_MAP:
            skipped["unknown_group"] += 1
            continue
        if response not in (0, 1):
            skipped["bad_response"] += 1
            continue
        if not math.isfinite(rt) or rt <= min_rt:
            skipped["rt_too_short_or_nonpositive"] += 1
            continue
        if max_rt is not None and rt > max_rt:
            skipped["rt_too_long"] += 1
            continue

        prepared.append(
            {
                "subj_idx": row["Subject"],
                "rt": f"{rt:.6f}",
                "response": str(response),
                "group": GROUP_MAP[row["condition"]],
                "condition_original": row["condition"],
                "visual_z": f"{visual_z:.12g}",
                "physical_z": f"{physical_z:.12g}",
                "video": row.get("Video", ""),
            }
        )

    return prepared, skipped


def describe_numeric(rows, column):
    vals = []
    for row in rows:
        try:
            value = float(row[column])
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(value):
            vals.append(value)

    return {
        "n": len(vals),
        "min": min(vals) if vals else float("nan"),
        "q25": quantile(vals, 0.25),
        "median": quantile(vals, 0.50),
        "q75": quantile(vals, 0.75),
        "max": max(vals) if vals else float("nan"),
    }


def write_prepared(path, rows):
    fieldnames = [
        "subj_idx",
        "rt",
        "response",
        "group",
        "condition_original",
        "visual_z",
        "physical_z",
        "video",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit(path, source_path, raw_rows, prepared_rows, skipped, rt_column, min_rt, max_rt):
    group_counts = Counter(row["group"] for row in prepared_rows)
    original_counts = Counter(row["condition"] for row in raw_rows)
    response_counts = Counter(row["response"] for row in prepared_rows)
    subj_counts = Counter(row["subj_idx"] for row in prepared_rows)

    lines = [
        "# HDDM data audit",
        "",
        f"- Source file: `{source_path}`",
        f"- Source rows: {len(raw_rows)}",
        f"- Prepared rows: {len(prepared_rows)}",
        f"- RT column: `{rt_column}`",
        f"- RT filter: `rt > {min_rt}`"
        + (f" and `rt <= {max_rt}`" if max_rt is not None else ""),
        f"- Subjects retained: {len(subj_counts)}",
        f"- Trial count per subject: min {min(subj_counts.values()) if subj_counts else 0}, "
        f"max {max(subj_counts.values()) if subj_counts else 0}",
        "",
        "## Original condition counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(original_counts.items()))
    lines.extend(["", "## Prepared group counts", ""])
    lines.extend(f"- {key}: {value}" for key, value in sorted(group_counts.items()))
    lines.extend(["", "## Response counts", ""])
    lines.extend(f"- response={key}: {value}" for key, value in sorted(response_counts.items()))

    if skipped:
        lines.extend(["", "## Skipped rows", ""])
        lines.extend(f"- {key}: {value}" for key, value in sorted(skipped.items()))

    lines.extend(["", "## Raw numeric summaries", ""])
    for column in ["RT_onset", "RT_critical", "Visual_Z", "Physical_Z"]:
        stats = describe_numeric(raw_rows, column)
        lines.append(
            "- {column}: n={n}, min={min:.4f}, q25={q25:.4f}, median={median:.4f}, "
            "q75={q75:.4f}, max={max:.4f}".format(column=column, **stats)
        )

    lines.extend(["", "## Prepared RT summary", ""])
    stats = describe_numeric(prepared_rows, "rt")
    lines.append(
        "- rt: n={n}, min={min:.4f}, q25={q25:.4f}, median={median:.4f}, "
        "q75={q75:.4f}, max={max:.4f}".format(**stats)
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="all_dat_merged.csv")
    parser.add_argument("--output", default="outputs/prepared_hddm_data.csv")
    parser.add_argument("--audit", default="outputs/data_audit.md")
    parser.add_argument("--rt-column", default="RT_critical")
    parser.add_argument("--min-rt", type=float, default=0.2)
    parser.add_argument("--max-rt", type=float, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = read_rows(input_path)
    prepared_rows, skipped = prepare_rows(raw_rows, args.rt_column, args.min_rt, args.max_rt)
    write_prepared(output_path, prepared_rows)
    write_audit(
        audit_path,
        input_path,
        raw_rows,
        prepared_rows,
        skipped,
        args.rt_column,
        args.min_rt,
        args.max_rt,
    )

    print(f"Wrote {output_path} with {len(prepared_rows)} rows.")
    print(f"Wrote {audit_path}.")


if __name__ == "__main__":
    main()
