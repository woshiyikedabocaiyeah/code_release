#!/usr/bin/env python
"""Build a manuscript-friendly HDDM results table from existing summaries."""

import csv
import json
from pathlib import Path


DISPLAY_GROUP = {
    "Action": "Action",
    "Semantic": "Semantic",
    "Voe": "Intuitive",
}


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_max_rhat(table_dir):
    out = {}
    for model in ["null", "group", "regression"]:
        rows = read_csv_rows(table_dir / f"{model}_rhat.csv")
        out[model] = max(float(row["rhat"]) for row in rows)
    return out


def fmt_num(x, digits=3):
    return f"{float(x):.{digits}f}"


def fmt_interval(lo, hi, digits=3):
    return f"[{fmt_num(lo, digits)}, {fmt_num(hi, digits)}]"


def main():
    root = Path.cwd()
    table_dir = root / "outputs" / "tables"

    model_rows = read_csv_rows(table_dir / "model_comparison_dic.csv")
    posterior_rows = read_csv_rows(table_dir / "group_posterior_plot_summary.csv")
    visual_rows = read_csv_rows(table_dir / "visual_z_double_dissociation_summary.csv")
    physical_rows = read_csv_rows(table_dir / "physical_z_reversal_summary.csv")
    hypothesis = json.loads((table_dir / "hypothesis_tests.json").read_text(encoding="utf-8"))
    max_rhat = load_max_rhat(table_dir)

    out_rows = []

    for row in model_rows:
        model = row["model"]
        out_rows.append(
            {
                "section": "Model comparison",
                "model": model,
                "parameter": "DIC",
                "condition": "",
                "estimate": fmt_num(row["dic"], 2),
                "interval": "",
                "notes": f"max R-hat = {fmt_num(max_rhat[model], 3)}",
            }
        )

    for row in posterior_rows:
        param = row["parameter"]
        group = DISPLAY_GROUP[row["group"]]
        if param == "a":
            section = "Group model"
            model = "group"
            parameter = "boundary separation a"
        elif param == "baseline_v":
            section = "Regression model"
            model = "regression"
            parameter = "baseline drift rate v"
        else:
            continue
        out_rows.append(
            {
                "section": section,
                "model": model,
                "parameter": parameter,
                "condition": group,
                "estimate": fmt_num(row["mean"]),
                "interval": fmt_interval(row["hdi_2.5"], row["hdi_97.5"]),
                "notes": "",
            }
        )

    for row in visual_rows:
        if row["section"] != "hddm_visual_slope":
            continue
        group = DISPLAY_GROUP[row["group"]]
        out_rows.append(
            {
                "section": "Regression slopes",
                "model": "regression",
                "parameter": "Visual_Z → v slope",
                "condition": group,
                "estimate": fmt_num(row["value_1"]),
                "interval": fmt_interval(row["value_2"], row["value_3"]),
                "notes": "",
            }
        )

    for row in physical_rows:
        if row["section"] != "hddm_posterior_slope":
            continue
        group = row["group"]
        out_rows.append(
            {
                "section": "Regression slopes",
                "model": "regression",
                "parameter": "Physical_Z → v slope",
                "condition": group,
                "estimate": fmt_num(row["estimate"]),
                "interval": fmt_interval(row["ci_low"], row["ci_high"]),
                "notes": "",
            }
        )

    boundary = hypothesis["boundary"]
    out_rows.append(
        {
            "section": "Key contrasts",
            "model": "group",
            "parameter": "P(a_Action > a_Semantic)",
            "condition": "",
            "estimate": fmt_num(boundary["probability"]),
            "interval": fmt_interval(boundary["diff_hdi_2.5"], boundary["diff_hdi_97.5"]),
            "notes": f"mean diff = {fmt_num(boundary['diff_mean'])}",
        }
    )

    csv_path = table_dir / "hddm_results_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "model", "parameter", "condition", "estimate", "interval", "notes"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    md_path = root / "outputs" / "hddm_results_table.md"
    lines = [
        "# HDDM Results Table",
        "",
        "| Section | Model | Parameter | Condition | Estimate | 95% HDI / CI | Notes |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in out_rows:
        lines.append(
            f"| {row['section']} | {row['model']} | {row['parameter']} | {row['condition']} | "
            f"{row['estimate']} | {row['interval']} | {row['notes']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
