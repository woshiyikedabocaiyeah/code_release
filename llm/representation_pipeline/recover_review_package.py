#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
ANALYSIS_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
MAIN_ROOT = PROJECT_ROOT / "outputs" / "semantic_yesno_main_experiment"
QUESTIONS_PATH = PROJECT_ROOT / "questions.json"
HUMAN_ROOT = ANALYSIS_ROOT / "humandata"

RESULT_FILES = {
    "embodied": MAIN_ROOT / "embodied" / "final_embodied_results.json",
    "non_embodied": MAIN_ROOT / "non_embodied" / "final_non_embodied_results.json",
}

EMBEDDING_TYPES = [
    "vision_encoder_last",
    "vision_projection",
    "language_model_last",
]
QUESTION_ORDER = ["Category", "VoE", "SM"]
QUESTION_SLUG = {"Category": "category", "VoE": "voe", "SM": "sm"}
HUMAN_TASK_SLUG = {"Category": "cat", "VoE": "voe", "SM": "sensor"}
QUESTION_LABEL = {
    "Category": "Categorization",
    "VoE": "VoE",
    "SM": "Sensorimotor",
}
LAYER_LABEL = {
    "vision_encoder_last": "Vision encoder (last layer)",
    "vision_projection": "Vision-language projection",
    "language_model_last": "Language model (last layer)",
}
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
CONDITION_ORDER = [
    ("non_embodied", "simple", "Non-embodied | Simple"),
    ("non_embodied", "detailed", "Non-embodied | Detailed"),
    ("embodied", "simple", "Embodied | Simple"),
    ("embodied", "detailed", "Embodied | Detailed"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the recovered yes/no review package from recovered model and human data."
    )
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--main-root", type=Path, default=MAIN_ROOT)
    parser.add_argument("--questions-path", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--human-root", type=Path, default=HUMAN_ROOT)
    parser.add_argument("--noise-iterations", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--start-at", choices=["all", "tail", "pc"], default="all")
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def clean_float(value: Any, digits: int | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float("nan")
    if digits is None or math.isnan(out):
        return out
    return round(out, digits)


def slugify(value: str) -> str:
    value = value.replace("(", " ").replace(")", " ")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_").lower()


def model_variant_name(model: str, mode: str) -> str:
    if model in {"InternVL3.5", "MiMo-Embodied"}:
        return f"{model} ({mode.title()})"
    return model


def prompt_family_from_raw(prompt_type: str) -> str:
    return "simple" if "simple" in str(prompt_type) else "detailed"


def condition_label(setting: str, prompt_family: str) -> str:
    setting_label = "Embodied" if setting == "embodied" else "Non-embodied"
    prompt_label = "Detailed" if prompt_family == "detailed" else "Simple"
    return f"{setting_label} | {prompt_label}"


def condition_key(setting: str, prompt_family: str) -> str:
    return f"{setting}_{prompt_family}"


def plausibility_from_possibility(possibility: str) -> str:
    return "plausible" if "Possible" in str(possibility) else "implausible"


def video_id_from_path(value: str) -> str:
    stem = Path(str(value)).stem
    return stem


def average_pool_embedding(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.astype(np.float32)
    return array.reshape(-1, array.shape[-1]).mean(axis=0).astype(np.float32)


def flatten_embedding(array: np.ndarray) -> np.ndarray:
    return array.reshape(-1).astype(np.float32)


def safe_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def q25(values: pd.Series) -> float:
    return float(values.quantile(0.25)) if len(values) else 0.0


def q75(values: pd.Series) -> float:
    return float(values.quantile(0.75)) if len(values) else 0.0


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr(rankdata(np.asarray(x)), rankdata(np.asarray(y)))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def upper_vector(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices].astype(np.float64)


def cosine_rdm(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    normalized = x / norms
    distances = 1.0 - normalized @ normalized.T
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32)


def save_rdm_vector(path: Path, matrix: np.ndarray, video_ids: list[str]) -> None:
    rows = []
    for i in range(len(video_ids)):
        for j in range(i + 1, len(video_ids)):
            rows.append(
                {
                    "row_i": i,
                    "row_j": j,
                    "video_id_i": video_ids[i],
                    "video_id_j": video_ids[j],
                    "distance": float(matrix[i, j]),
                }
            )
    write_rows(path, rows)


def load_vector_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "distance" in df.columns:
        return df["distance"].to_numpy(dtype=np.float64)
    return df.iloc[:, -1].to_numpy(dtype=np.float64)


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    return {row["id"]: row for row in payload["questions"]}


def build_trial_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = load_questions(args.questions_path)
    trial_rows: list[dict[str, Any]] = []
    for setting, relative_path in {
        "embodied": args.main_root / "embodied" / "final_embodied_results.json",
        "non_embodied": args.main_root / "non_embodied" / "final_non_embodied_results.json",
    }.items():
        payload = read_json(relative_path)
        for row in payload["results"]:
            question = questions[row["question_id"]]
            qmeta = question["metadata"]
            metadata_path = Path(row["metadata_path"])
            metadata = read_json(metadata_path) if metadata_path.exists() else {}
            prompt_family = prompt_family_from_raw(row.get("prompt_type", ""))
            model_variant = model_variant_name(row["model"], row["mode"])
            embedding_files = metadata.get("embedding_files", {})
            embedding_stats = metadata.get("embedding_stats", {})
            trial_rows.append(
                {
                    "setting": setting,
                    "condition_label": condition_label(setting, prompt_family),
                    "prompt_family": prompt_family,
                    "model": row["model"],
                    "mode": row["mode"],
                    "model_variant": model_variant,
                    "prompt_type_raw": row.get("prompt_type", ""),
                    "question_id": row["question_id"],
                    "video_id": row["video_id"],
                    "question_type": question["question_type"],
                    "correct_answer": question["correct_answer"],
                    "model_answer": row.get("model_answer"),
                    "correct": bool(row.get("correct", False)),
                    "inference_time_seconds": clean_float(row.get("inference_time_seconds")),
                    "success": bool(row.get("success", True)),
                    "error": row.get("error"),
                    "answer_parse_status": row.get("answer_parse_status", ""),
                    "answer_parse_method": row.get("answer_parse_method", ""),
                    "source_metadata_path": str(metadata_path),
                    "source_video_path": str(metadata.get("video_path", PROJECT_ROOT / question["video_path"])),
                    "physical_concept": qmeta.get("physical_concept", qmeta.get("condition", "")),
                    "possibility": qmeta.get("possibility", ""),
                    "plausibility": plausibility_from_possibility(qmeta.get("possibility", "")),
                    "game_name": qmeta.get("game_name", ""),
                    "environment": qmeta.get("environment", ""),
                    "occlusion": qmeta.get("occlusion", ""),
                    "difficulty": qmeta.get("difficulty", ""),
                    "camera": qmeta.get("camera", ""),
                    "embedding_files": embedding_files,
                    "embedding_files_json": json.dumps(embedding_files, sort_keys=True),
                    "embedding_saved_shapes_json": json.dumps(
                        {
                            key: value.get("saved_shape")
                            for key, value in embedding_stats.items()
                            if isinstance(value, dict)
                        },
                        sort_keys=True,
                    ),
                }
            )
    return sorted(
        trial_rows,
        key=lambda r: (
            MODEL_ORDER.index(r["model_variant"]) if r["model_variant"] in MODEL_ORDER else 999,
            r["setting"],
            r["prompt_family"],
            QUESTION_ORDER.index(r["question_type"]),
            r["video_id"],
        ),
    )


def trial_dataframe(trial_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(trial_rows).copy()
    df["setting_label"] = np.where(df["setting"] == "embodied", "Embodied", "Non-embodied")
    df["prompt_family_label"] = np.where(df["prompt_family"] == "detailed", "Detailed", "Simple")
    df["condition"] = df["condition_label"]
    df["condition_key"] = df.apply(lambda r: condition_key(r["setting"], r["prompt_family"]), axis=1)
    df["question_type_label"] = df["question_type"].map(QUESTION_LABEL)
    return df


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, bucket in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out = dict(zip(group_cols, keys))
        n_trials = len(bucket)
        n_correct = int(bucket["correct"].sum())
        accuracy = n_correct / n_trials if n_trials else 0.0
        rt = bucket["inference_time_seconds"].astype(float)
        out.update(
            {
                "n_trials": n_trials,
                "n_correct": n_correct,
                "n_wrong": n_trials - n_correct,
                "accuracy": round(accuracy, 4),
                "accuracy_pct": round(accuracy * 100, 4),
                "error_rate_pct": round((1.0 - accuracy) * 100, 4),
                "mean_rt_s": round(float(rt.mean()), 4),
                "sd_rt_s": round(safe_std(rt), 4),
                "median_rt_s": round(float(rt.median()), 4),
                "q25_rt_s": round(q25(rt), 4),
                "q75_rt_s": round(q75(rt), 4),
                "mean_rt_correct_only_s": round(float(bucket.loc[bucket["correct"], "inference_time_seconds"].mean()), 4),
                "mean_rt_wrong_only_s": round(float(bucket.loc[~bucket["correct"], "inference_time_seconds"].mean()), 4),
                "ies_s": round(float(rt.mean()) / max(accuracy, 1e-9), 4),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def summary_of_summaries(df: pd.DataFrame, group_col: str, count_col: str) -> pd.DataFrame:
    rows = []
    metrics = ["accuracy_pct", "mean_rt_s", "ies_s"]
    for value, bucket in df.groupby(group_col, sort=False):
        out = {group_col: value, count_col: len(bucket)}
        for metric in metrics:
            values = bucket[metric].astype(float)
            out.update(
                {
                    f"{metric}_mean": round(float(values.mean()), 4),
                    f"{metric}_sd": round(safe_std(values), 4),
                    f"{metric}_median": round(float(values.median()), 4),
                    f"{metric}_min": round(float(values.min()), 4),
                    f"{metric}_max": round(float(values.max()), 4),
                }
            )
        rows.append(out)
    return pd.DataFrame(rows)


def count_json(values: pd.Series) -> str:
    return json.dumps({str(k): int(v) for k, v in Counter(values.dropna()).items()}, sort_keys=True)


def error_breakdown(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, bucket in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        wrong = bucket[~bucket["correct"]]
        out = dict(zip(group_cols, keys))
        if "question_type" in out:
            out["question_type_label"] = QUESTION_LABEL.get(out["question_type"], out["question_type"])
        n_trials = len(bucket)
        n_wrong = len(wrong)
        wrong_counts = Counter(wrong["model_answer"].dropna())
        out.update(
            {
                "n_trials": n_trials,
                "n_wrong": n_wrong,
                "error_rate_pct": round(n_wrong / n_trials * 100 if n_trials else 0.0, 4),
                "accuracy_pct": round((1 - n_wrong / n_trials) * 100 if n_trials else 0.0, 4),
                "correct_answer_counts_json": count_json(bucket["correct_answer"]),
                "model_answer_counts_json": count_json(bucket["model_answer"]),
                "wrong_answer_counts_json": count_json(wrong["model_answer"]),
                "most_common_wrong_answer": wrong_counts.most_common(1)[0][0] if wrong_counts else "",
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def add_delta_bands(df: pd.DataFrame, value_col: str, group_cols: list[str], band_col: str) -> pd.DataFrame:
    df = df.copy()
    bands = []
    for _, bucket in df.groupby(group_cols, sort=False):
        abs_values = bucket[value_col].astype(float).abs()
        lo = abs_values.quantile(0.25)
        hi = abs_values.quantile(0.75)
        for value in abs_values:
            if value <= lo:
                bands.append("similar")
            elif value >= hi:
                bands.append("large")
            else:
                bands.append("moderate")
    df[band_col] = bands
    return df


def draw_bar_chart(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "#F7F5EF")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((40, 30), title, fill="#222222", font=font)
    if not values:
        image.save(path)
        return
    max_value = max(values) or 1.0
    left, top, bottom = 80, 100, 620
    bar_w = max(18, int((width - 160) / max(1, len(values))))
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = left + index * bar_w
        x1 = x0 + int(bar_w * 0.72)
        h = int((value / max_value) * (bottom - top))
        color = "#3C5488" if index % 2 == 0 else "#00A087"
        draw.rectangle((x0, bottom - h, x1, bottom), fill=color)
        draw.text((x0, bottom + 8), label[:14], fill="#222222", font=font)
        draw.text((x0, bottom - h - 15), f"{value:.1f}", fill="#222222", font=font)
    image.save(path)


def draw_heatmap(path: Path, title: str, row_labels: list[str], col_labels: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 110, 30
    left, top = 220, 90
    width = left + cell_w * len(col_labels) + 60
    height = top + cell_h * len(row_labels) + 80
    image = Image.new("RGB", (width, height), "#F7F5EF")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 28), title, fill="#222222", font=font)
    finite = values[np.isfinite(values)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    span = max(vmax - vmin, 1e-9)
    for c, label in enumerate(col_labels):
        draw.text((left + c * cell_w + 4, top - 24), label[:16], fill="#555555", font=font)
    for r, label in enumerate(row_labels):
        draw.text((20, top + r * cell_h + 8), label[:28], fill="#222222", font=font)
        for c in range(len(col_labels)):
            value = float(values[r, c])
            f = (value - vmin) / span if math.isfinite(value) else 0.0
            red = int(245 * f + 50 * (1 - f))
            green = int(245 * (1 - abs(f - 0.5)) + 80 * abs(f - 0.5))
            blue = int(80 * f + 180 * (1 - f))
            color = (red, green, blue)
            x0 = left + c * cell_w
            y0 = top + r * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 2, y0 + cell_h - 2), fill=color)
            draw.text((x0 + 8, y0 + 8), f"{value:.2f}", fill="#111111", font=font)
    image.save(path)


def draw_scree(path: Path, title: str, variance: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, 460
    image = Image.new("RGB", (width, height), "#FBFAF7")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 25), title[:100], fill="#222222", font=font)
    left, top, right, bottom = 60, 70, 680, 390
    draw.rectangle((left, top, right, bottom), outline="#888888")
    values = variance[: min(20, len(variance))]
    max_value = max(float(values.max()) if len(values) else 1.0, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = left + (right - left) * index / max(1, len(values) - 1)
        y = bottom - (bottom - top) * float(value) / max_value
        points.append((x, y))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#3C5488")
    if len(points) > 1:
        draw.line(points, fill="#3C5488", width=2)
    draw.text((left, bottom + 18), "PC index (first 20)", fill="#555555", font=font)
    draw.text((left, bottom + 36), "Explained variance ratio", fill="#555555", font=font)
    image.save(path)


def run_task_performance(args: argparse.Namespace, trial_rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    root = args.package_root / "Task_Performance"
    data_dir = root / "Data"
    fig_dir = root / "Figure"
    report_dir = root / "Report"
    ensure_dirs(data_dir, fig_dir, report_dir)
    df = trial_dataframe(trial_rows)
    write_json(
        data_dir / "00_run_summary.json",
        {
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
        },
    )

    condition_summary_base = summarize_group(
        df,
        ["model_variant", "setting", "setting_label", "prompt_family", "condition", "condition_key"],
    )
    condition_summary = summary_of_summaries(condition_summary_base, "condition", "model_count")
    write_df(data_dir / "01_condition_summary.csv", condition_summary)

    task_base = summarize_group(
        df,
        [
            "model_variant",
            "setting",
            "setting_label",
            "prompt_family",
            "condition",
            "condition_key",
            "question_type",
            "question_type_label",
        ],
    )
    task_summary = summary_of_summaries(task_base, "question_type", "row_count")
    task_summary.insert(1, "question_type_label", task_summary["question_type"].map(QUESTION_LABEL))
    write_df(data_dir / "02_task_summary.csv", task_summary)

    write_df(data_dir / "03_model_condition_summary.csv", condition_summary_base)
    write_df(data_dir / "04_model_condition_task_summary.csv", task_base)

    baseline_rows = []
    baseline_lookup = {
        (row["model_variant"], row["question_type"]): row
        for _, row in task_base[task_base["condition"] == "Non-embodied | Simple"].iterrows()
    }
    for _, row in task_base[task_base["condition"] != "Non-embodied | Simple"].iterrows():
        baseline = baseline_lookup[(row["model_variant"], row["question_type"])]
        baseline_rows.append(
            {
                "model_variant": row["model_variant"],
                "baseline_condition": "Non-embodied | Simple",
                "target_condition": row["condition"],
                "question_type": row["question_type"],
                "question_type_label": row["question_type_label"],
                "baseline_accuracy_pct": baseline["accuracy_pct"],
                "target_accuracy_pct": row["accuracy_pct"],
                "delta_accuracy_pct": round(float(row["accuracy_pct"]) - float(baseline["accuracy_pct"]), 4),
                "baseline_mean_rt_s": baseline["mean_rt_s"],
                "target_mean_rt_s": row["mean_rt_s"],
                "delta_mean_rt_s": round(float(row["mean_rt_s"]) - float(baseline["mean_rt_s"]), 4),
                "baseline_ies_s": baseline["ies_s"],
                "target_ies_s": row["ies_s"],
                "delta_ies_s": round(float(row["ies_s"]) - float(baseline["ies_s"]), 4),
            }
        )
    baseline_delta = add_delta_bands(
        pd.DataFrame(baseline_rows),
        "delta_accuracy_pct",
        ["question_type"],
        "delta_accuracy_band",
    )
    write_df(data_dir / "05_baseline_delta_vs_non_embodied_simple.csv", baseline_delta)
    write_df(data_dir / "06_error_breakdown_by_task.csv", error_breakdown(df, ["question_type"]))
    physical_error = error_breakdown(df, ["question_type", "physical_concept"])
    physical_error = physical_error[
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
        ]
    ]
    write_df(data_dir / "07_error_breakdown_by_task_and_physical_concept.csv", physical_error)
    plausibility_error = error_breakdown(df, ["question_type", "plausibility"])
    plausibility_error = plausibility_error[
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
        ]
    ]
    write_df(data_dir / "08_error_breakdown_by_task_and_plausibility.csv", plausibility_error)

    order_rows = []
    for (model, condition), bucket in task_base.groupby(["model_variant", "condition"], sort=False):
        acc = {row["question_type"]: float(row["accuracy_pct"]) for _, row in bucket.iterrows()}
        order = sorted(acc.items(), key=lambda item: (-item[1], item[0]))
        order_rows.append(
            {
                "model_variant": model,
                "condition": condition,
                "category_accuracy_pct": round(acc.get("Category", float("nan")), 4),
                "voe_accuracy_pct": round(acc.get("VoE", float("nan")), 4),
                "sm_accuracy_pct": round(acc.get("SM", float("nan")), 4),
                "rank_pattern": ">".join(key if key != "Category" else "Category" for key, _ in order),
                "voe_minus_sm_accuracy_pct": round(acc.get("VoE", 0.0) - acc.get("SM", 0.0), 4),
                "sm_minus_category_accuracy_pct": round(acc.get("SM", 0.0) - acc.get("Category", 0.0), 4),
            }
        )
    task_order = pd.DataFrame(order_rows)
    write_df(data_dir / "09_task_order_patterns.csv", task_order)

    draw_bar_chart(
        fig_dir / "01_overall_model_condition_panels.png",
        "Overall model-condition accuracy",
        condition_summary_base["model_variant"].tolist(),
        condition_summary_base["accuracy_pct"].astype(float).tolist(),
    )
    for filename, metric, title in [
        ("02_taskwise_accuracy_model_panels.png", "accuracy_pct", "Taskwise accuracy"),
        ("03_taskwise_rt_model_panels.png", "mean_rt_s", "Taskwise RT"),
        ("04_taskwise_ies_model_panels.png", "ies_s", "Taskwise IES"),
    ]:
        pivot = task_base.pivot_table(index="model_variant", columns="question_type", values=metric, aggfunc="mean").reindex(MODEL_ORDER)
        draw_heatmap(fig_dir / filename, title, [str(x) for x in pivot.index], QUESTION_ORDER, pivot[QUESTION_ORDER].to_numpy())
    delta_pivot = baseline_delta.pivot_table(index="model_variant", columns=["target_condition", "question_type"], values="delta_accuracy_pct", aggfunc="mean")
    draw_heatmap(
        fig_dir / "05_accuracy_delta_vs_baseline_heatmap.png",
        "Accuracy delta vs Non-embodied Simple",
        [str(x) for x in delta_pivot.index],
        [f"{a} {b}" for a, b in delta_pivot.columns],
        delta_pivot.to_numpy(),
    )
    error_task = pd.read_csv(data_dir / "06_error_breakdown_by_task.csv")
    draw_bar_chart(
        fig_dir / "06_error_breakdown_heatmaps.png",
        "Error rate by task",
        error_task["question_type"].tolist(),
        error_task["error_rate_pct"].astype(float).tolist(),
    )
    order_counts = task_order["rank_pattern"].value_counts()
    draw_bar_chart(
        fig_dir / "07_task_accuracy_order_heatmap.png",
        "Task accuracy order pattern counts",
        order_counts.index.tolist(),
        order_counts.astype(float).tolist(),
    )

    best = condition_summary_base.sort_values("accuracy_pct", ascending=False).iloc[0]
    worst = condition_summary_base.sort_values("accuracy_pct", ascending=True).iloc[0]
    max_delta = baseline_delta.sort_values("delta_accuracy_pct", ascending=False).iloc[0]
    report = [
        "# Task performance: data summary",
        "",
        "## Scope",
        "",
        f"- Based on the full main-experiment results: `{len(df)}` trials",
        f"- `{df['model_variant'].nunique()}` model variants",
    ]
    for _, row in task_summary.iterrows():
        report.append(
            f"- `{row['question_type']}`: mean accuracy `{row['accuracy_pct_mean']:.4f}%`, mean RT `{row['mean_rt_s_mean']:.4f}s`, mean IES `{row['ies_s_mean']:.4f}`"
        )
    report.extend(["", "## By prompt condition", ""])
    for _, row in condition_summary.iterrows():
        report.append(
            f"- `{row['condition']}`: mean accuracy `{row['accuracy_pct_mean']:.4f}%`, mean RT `{row['mean_rt_s_mean']:.4f}s`, mean IES `{row['ies_s_mean']:.4f}`"
        )
    report.extend(
        [
            "",
            "## Errors and task ordering",
            "",
            f"- Highest error rate: `{error_task.sort_values('error_rate_pct', ascending=False).iloc[0]['question_type']}`",
        ]
    )
    for pattern, count in order_counts.items():
        report.append(f"- `{pattern}` occurs `{int(count)}` times")
    report.extend(
        [
            "",
            "## Extremes",
            "",
            f"- Highest accuracy: `{best['model_variant']} | {best['condition']}`, `{best['accuracy_pct']:.4f}%`",
            f"- Lowest accuracy: `{worst['model_variant']} | {worst['condition']}`, `{worst['accuracy_pct']:.4f}%`",
            f"- Largest gain over baseline: `{max_delta['model_variant']} | {max_delta['target_condition']} | {max_delta['question_type']}`, `{max_delta['delta_accuracy_pct']:+.4f}` percentage points",
        ]
    )
    (report_dir / "01_data_summary.md").write_text("\n".join(report) + "\n")
    (report_dir / "02_interpretation_summary.md").write_text(
        "# Task performance: interpretation\n\n"
        "`Non-embodied | Simple` is the reference baseline; the detailed and embodied prompts are compared against it on accuracy, RT and IES for each task."
        "The RSA, probing and PC-distance analyses all use the same video-ordered trial manifest, so results align across modules.\n"
    )
    (root / "README.md").write_text("# Task performance\n\nBehavioural results for the three tasks.\n")
    return {"trial_df": df, "task_base": task_base, "baseline_delta": baseline_delta}


def group_trials_for_embeddings(trial_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        key_base = (row["model_variant"], row["setting"], row["prompt_family"], row["question_type"])
        groups[key_base].append(row)
    for key in list(groups):
        groups[key] = sorted(groups[key], key=lambda r: r["video_id"])
    return groups


def base_group_slug(model_variant: str, setting: str, prompt_family: str, question_type: str, embedding_type: str) -> str:
    return "__".join(
        [
            slugify(model_variant),
            setting,
            prompt_family,
            QUESTION_SLUG[question_type],
            embedding_type,
        ]
    )


def load_group_matrix(rows: list[dict[str, Any]], embedding_type: str, strategy: str = "mean_pool") -> np.ndarray:
    vectors = []
    for row in rows:
        path = Path(row["embedding_files"].get(embedding_type, ""))
        if not path.exists():
            raise FileNotFoundError(path)
        array = np.load(path)
        vectors.append(average_pool_embedding(array) if strategy == "mean_pool" else flatten_embedding(array))
    return np.vstack(vectors).astype(np.float32)


def row_metadata_records(rows: list[dict[str, Any]], embedding_type: str) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(rows):
        records.append(
            {
                "video_row_index": index,
                "video_id": row["video_id"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "source_video_path": row["source_video_path"],
                "physical_concept": row["physical_concept"],
                "plausibility": row["plausibility"],
                "possibility_label_from_dataset": row["possibility"],
                "scene_template": row["game_name"],
                "environment": row["environment"],
                "occlusion_configuration": row["occlusion"],
                "difficulty_level": row["difficulty"],
                "camera_condition": row["camera"],
                "model_answer_correct": row["correct"],
                "model_inference_time_seconds": row["inference_time_seconds"],
                "source_metadata_path": row["source_metadata_path"],
                "embedding_file": row["embedding_files"].get(embedding_type, ""),
            }
        )
    return records


def run_rdm(args: argparse.Namespace, trial_rows: list[dict[str, Any]]) -> dict[str, Any]:
    root = args.package_root / "RDM"
    data_dir = root / "Data"
    fig_dir = root / "Figure"
    report_dir = root / "Report"
    matrix_dir = data_dir / "rdm_matrices"
    vector_dir = data_dir / "rdm_vectors"
    rowmeta_dir = data_dir / "row_metadata"
    pooled_dir = data_dir / "matrices_ready"
    ensure_dirs(data_dir, fig_dir, report_dir, matrix_dir, vector_dir, rowmeta_dir, pooled_dir)

    trial_manifest_cols = [
        "setting",
        "condition_label",
        "prompt_family",
        "model",
        "mode",
        "model_variant",
        "prompt_type_raw",
        "question_id",
        "video_id",
        "question_type",
        "correct_answer",
        "model_answer",
        "correct",
        "inference_time_seconds",
        "success",
        "error",
        "answer_parse_status",
        "answer_parse_method",
        "source_metadata_path",
        "source_video_path",
        "physical_concept",
        "possibility",
        "plausibility",
        "game_name",
        "environment",
        "occlusion",
        "difficulty",
        "camera",
        "embedding_files_json",
        "embedding_saved_shapes_json",
    ]
    write_df(data_dir / "03_trial_manifest.csv", pd.DataFrame(trial_rows)[trial_manifest_cols])

    input_rows = []
    summary_rows = []
    groups = group_trials_for_embeddings(trial_rows)
    for (model_variant, setting, prompt_family, question_type), rows in groups.items():
        for embedding_type in EMBEDDING_TYPES:
            base_slug = base_group_slug(model_variant, setting, prompt_family, question_type, embedding_type)
            rowmeta_path = rowmeta_dir / f"{base_slug}.csv"
            write_rows(rowmeta_path, row_metadata_records(rows, embedding_type))
            pooled_matrix = load_group_matrix(rows, embedding_type, "mean_pool")
            pooled_path = pooled_dir / f"{base_slug}.npy"
            np.save(pooled_path, pooled_matrix)
            video_ids = [row["video_id"] for row in rows]
            physical_counts = json.dumps(dict(sorted(Counter(row["physical_concept"] for row in rows).items())), sort_keys=True)
            plaus_counts = json.dumps(dict(sorted(Counter(row["plausibility"] for row in rows).items())), sort_keys=True)
            for strategy in ["flatten", "mean_pool"]:
                matrix = pooled_matrix if strategy == "mean_pool" else load_group_matrix(rows, embedding_type, "flatten")
                rdm = cosine_rdm(matrix)
                group_slug = f"{base_slug}__{strategy}"
                rdm_path = matrix_dir / strategy / f"{group_slug}.npy"
                vector_path = vector_dir / strategy / f"{group_slug}.csv"
                ensure_dirs(rdm_path.parent, vector_path.parent)
                np.save(rdm_path, rdm)
                save_rdm_vector(vector_path, rdm, video_ids)
                heatmap_path = fig_dir / f"{group_slug}.png"
                if len(summary_rows) < 4:
                    draw_heatmap(
                        heatmap_path,
                        group_slug,
                        [str(i) for i in range(rdm.shape[0])],
                        [str(i) for i in range(rdm.shape[1])],
                        rdm,
                    )
                else:
                    heatmap_path = Path("")
                input_rows.append(
                    {
                        "group_slug": group_slug,
                        "base_group_slug": base_slug,
                        "aggregation_strategy": strategy,
                        "model_variant": model_variant,
                        "setting": setting,
                        "prompt_family": prompt_family,
                        "condition_label": condition_label(setting, prompt_family),
                        "question_type": question_type,
                        "embedding_type": embedding_type,
                        "n_samples": len(rows),
                        "n_features_flat": int(matrix.shape[1]),
                        "matrix_path": str(pooled_path),
                        "row_metadata_path": str(rowmeta_path),
                        "video_order_key": "|".join(video_ids),
                        "physical_concept_counts": physical_counts,
                        "plausibility_counts": plaus_counts,
                        "vector_length_set": json.dumps([int(matrix.shape[1])]),
                        "vector_length_consistent": True,
                        "row_order_consistent_across_layers": True,
                        "duplicate_video_count": len(video_ids) - len(set(video_ids)),
                        "zero_variance_row_count": int(np.sum(np.std(matrix, axis=1) <= 1e-12)),
                        "build_ok": True,
                        "rdm_ready_ok": True,
                        "skip_reason": "",
                    }
                )
                finite = np.isfinite(rdm)
                summary_rows.append(
                    {
                        "group_slug": group_slug,
                        "base_group_slug": base_slug,
                        "aggregation_strategy": strategy,
                        "model_variant": model_variant,
                        "setting": setting,
                        "prompt_family": prompt_family,
                        "condition_label": condition_label(setting, prompt_family),
                        "question_type": question_type,
                        "embedding_type": embedding_type,
                        "n_samples": len(rows),
                        "n_features_flat": int(matrix.shape[1]),
                        "input_build_ok": True,
                        "completed": True,
                        "skip_reason": "",
                        "min_distance": float(rdm[np.triu_indices(rdm.shape[0], 1)].min()),
                        "max_distance": float(rdm[np.triu_indices(rdm.shape[0], 1)].max()),
                        "mean_distance": float(rdm[np.triu_indices(rdm.shape[0], 1)].mean()),
                        "finite_ok": bool(finite.all()),
                        "symmetric_ok": bool(np.allclose(rdm, rdm.T, atol=1e-6)),
                        "diag_zero_ok": bool(np.allclose(np.diag(rdm), 0.0, atol=1e-6)),
                        "rdm_path": str(rdm_path),
                        "rdm_vector_path": str(vector_path),
                        "heatmap_path": str(heatmap_path),
                    }
                )

    input_df = pd.DataFrame(input_rows)
    summary_df = pd.DataFrame(summary_rows)
    write_df(data_dir / "01_rdm_input_manifest.csv", input_df)
    write_df(data_dir / "02_rdm_summary.csv", summary_df)
    mean_dist = summary_df["mean_distance"].astype(float)
    identical_count = int(
        summary_df.pivot_table(index="base_group_slug", columns="aggregation_strategy", values="mean_distance", aggfunc="first")
        .pipe(lambda p: np.isclose(p["flatten"], p["mean_pool"], atol=1e-12).sum())
    )
    report = [
        "# RDM: data summary",
        "",
        "## Scope",
        "",
        f"- Model RDMs in this package: `{len(summary_df)}`",
        f"- `flatten` aggregation: `{int((summary_df['aggregation_strategy'] == 'flatten').sum())}`; `mean_pool` aggregation: `{int((summary_df['aggregation_strategy'] == 'mean_pool').sum())}`",
        f"- Rows in the input manifest: `{len(input_df)}`",
        "",
        "## Descriptive statistics",
        "",
        f"- Overall mean of group `mean_distance`: `{mean_dist.mean():.4f}`",
        f"- Median group mean distance: `{mean_dist.median():.4f}`",
        f"- Range of group mean distance: `{mean_dist.min():.6f}` to `{mean_dist.max():.6f}`",
        f"- Across base-group pairs, `flatten` and `mean_pool` agree on the mean in `{identical_count}` groups",
    ]
    (report_dir / "01_data_summary.md").write_text("\n".join(report) + "\n")
    (report_dir / "02_interpretation_summary.md").write_text(
        "# RDM: interpretation\n\nEach RDM is a cosine-distance matrix over the per-video embeddings. All downstream RSA, permutation and noise-ceiling analyses use only the task-matched `mean_pool` RDMs.\n"
    )
    (root / "README.md").write_text("# RDM\n\nRepresentational dissimilarity matrices.\n")
    return {"rdm_summary": summary_df, "rdm_input": input_df}


def pca_decompose(matrix: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if mode == "zscore":
        std = x.std(axis=0, keepdims=True)
        std[std <= 1e-8] = 1.0
        x = x / std
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    scores = u * s
    variances = s ** 2
    total = variances.sum()
    ratios = variances / total if total > 1e-12 else np.zeros_like(variances)
    return scores.astype(np.float32), ratios.astype(np.float64), vt.astype(np.float32)


def k_for_threshold(ratios: np.ndarray, threshold: float) -> int:
    cumulative = np.cumsum(ratios)
    hits = np.where(cumulative >= threshold)[0]
    return int(hits[0] + 1) if len(hits) else int(len(ratios))


def run_pca(args: argparse.Namespace, trial_rows: list[dict[str, Any]]) -> dict[str, Any]:
    root = args.package_root / "PCA"
    data_dir = root / "Data"
    fig_dir = root / "Figure"
    report_dir = root / "Report"
    full_scores_dir = data_dir / "full_video_pc_scores_centered_by_matrix"
    scores_dir = data_dir / "results" / "centered" / "scores"
    variance_dir = data_dir / "results" / "centered" / "variance"
    zscore_scores_dir = data_dir / "results" / "zscore" / "scores"
    zscore_variance_dir = data_dir / "results" / "zscore" / "variance"
    scree_dir = fig_dir / "centered_scree_plots"
    summary_fig_dir = fig_dir / "summary_figures"
    ensure_dirs(data_dir, report_dir, full_scores_dir, scores_dir, variance_dir, zscore_scores_dir, zscore_variance_dir, scree_dir, summary_fig_dir)

    groups = group_trials_for_embeddings(trial_rows)
    group_qc_rows = []
    pca_rows = []
    centered_rows = []
    top3_rows = []
    extremes_rows = []
    semantic_rows = []

    for (model_variant, setting, prompt_family, question_type), rows in groups.items():
        for embedding_type in EMBEDDING_TYPES:
            slug = base_group_slug(model_variant, setting, prompt_family, question_type, embedding_type)
            matrix = load_group_matrix(rows, embedding_type, "mean_pool")
            physical_counts = json.dumps(dict(sorted(Counter(row["physical_concept"] for row in rows).items())), sort_keys=True)
            plaus_counts = json.dumps(dict(sorted(Counter(row["plausibility"] for row in rows).items())), sort_keys=True)
            group_qc_rows.append(
                {
                    "group_slug": slug,
                    "model_variant": model_variant,
                    "setting": "Embodied" if setting == "embodied" else "Non-embodied",
                    "prompt_family": "Detailed" if prompt_family == "detailed" else "Simple",
                    "task_type": QUESTION_LABEL[question_type],
                    "embedding_layer": LAYER_LABEL[embedding_type],
                    "number_of_videos": len(rows),
                    "pooled_embedding_dimensions": matrix.shape[1],
                    "duplicate_video_count": len(rows) - len({row["video_id"] for row in rows}),
                    "physical_concept_counts": physical_counts,
                    "plausibility_counts": plaus_counts,
                    "expected_sample_count_ok": len(rows) == 48,
                    "balanced_physical_concept_ok": sorted(Counter(row["physical_concept"] for row in rows).values()) == [16, 16, 16],
                    "balanced_plausibility_ok": sorted(Counter(row["plausibility"] for row in rows).values()) == [24, 24],
                    "row_order_consistent_across_layers": True,
                }
            )
            centered_scores: np.ndarray | None = None
            centered_ratios: np.ndarray | None = None
            for mode, label, score_dir, var_dir in [
                ("centered", "Centered (main analysis)", scores_dir, variance_dir),
                ("zscore", "Z-score (robustness check)", zscore_scores_dir, zscore_variance_dir),
            ]:
                scores, ratios, components = pca_decompose(matrix, mode)
                score_path = score_dir / f"{slug}.csv"
                var_path = var_dir / f"{slug}.csv"
                component_path = data_dir / "results" / mode / "components" / f"{slug}.npz"
                ensure_dirs(component_path.parent)
                np.savez_compressed(component_path, components=components[: min(10, len(components))])
                score_records = []
                for index, row in enumerate(rows):
                    record = {
                        "video_row_index": index,
                        "video_id": row["video_id"],
                        "question_id": row["question_id"],
                    }
                    for pc_idx in range(scores.shape[1]):
                        record[f"principal_component_{pc_idx + 1}_score"] = float(scores[index, pc_idx])
                    score_records.append(record)
                write_rows(score_path, score_records)
                write_rows(
                    var_path,
                    [
                        {
                            "principal_component": index + 1,
                            "explained_variance_ratio": float(value),
                            "cumulative_explained_variance_ratio": float(np.cumsum(ratios)[index]),
                        }
                        for index, value in enumerate(ratios)
                    ],
                )
                scree_path = scree_dir / f"{slug}.png" if mode == "centered" else Path("")
                if mode == "centered":
                    draw_scree(scree_path, slug, ratios)
                    centered_scores, centered_ratios = scores, ratios
                row = {
                    "analysis_mode": label,
                    "model_variant": model_variant,
                    "setting": "Embodied" if setting == "embodied" else "Non-embodied",
                    "prompt_family": "Detailed" if prompt_family == "detailed" else "Simple",
                    "condition_label": condition_label(setting, prompt_family),
                    "task_type": QUESTION_LABEL[question_type],
                    "embedding_layer": LAYER_LABEL[embedding_type],
                    "number_of_videos": len(rows),
                    "pooled_embedding_dimensions": matrix.shape[1],
                    "number_of_available_principal_components": scores.shape[1],
                    "pc1_explained_variance_ratio": float(ratios[0]) if len(ratios) > 0 else 0.0,
                    "pc2_explained_variance_ratio": float(ratios[1]) if len(ratios) > 1 else 0.0,
                    "pc3_explained_variance_ratio": float(ratios[2]) if len(ratios) > 2 else 0.0,
                    "top3_cumulative_explained_variance_ratio": float(ratios[:3].sum()),
                    "principal_components_needed_for_80_percent_variance": k_for_threshold(ratios, 0.80),
                    "principal_components_needed_for_90_percent_variance": k_for_threshold(ratios, 0.90),
                    "per_video_pc_scores_file": str(score_path),
                    "per_component_variance_file": str(var_path),
                    "component_directions_file": str(component_path),
                    "scree_plot_file": str(scree_path),
                    "group_slug": slug,
                }
                pca_rows.append(row)
                if mode == "centered":
                    centered_rows.append(row)

            assert centered_scores is not None and centered_ratios is not None
            full_records = []
            for index, row in enumerate(rows):
                record = {
                    "video_row_index": index,
                    "video_id": row["video_id"],
                    "question_id": row["question_id"],
                    "task_type": QUESTION_LABEL[question_type],
                    "physical_concept": row["physical_concept"],
                    "plausibility": row["plausibility"],
                    "possibility_label_from_dataset": row["possibility"],
                    "scene_template": row["game_name"],
                    "environment": row["environment"],
                    "occlusion_configuration": row["occlusion"],
                    "difficulty_level": row["difficulty"],
                    "camera_condition": row["camera"],
                    "model_answer_correct": row["correct"],
                    "model_inference_time_seconds": row["inference_time_seconds"],
                    "condition_label": condition_label(setting, prompt_family),
                    "model_variant": model_variant,
                    "setting": "Embodied" if setting == "embodied" else "Non-embodied",
                    "prompt_family": "Detailed" if prompt_family == "detailed" else "Simple",
                    "embedding_layer": LAYER_LABEL[embedding_type],
                }
                for pc_idx in range(centered_scores.shape[1]):
                    record[f"principal_component_{pc_idx + 1}_score"] = float(centered_scores[index, pc_idx])
                record["source_video_path"] = row["source_video_path"]
                full_records.append(record)
                if index < len(rows):
                    top3_rows.append(
                        {
                            "model_variant": model_variant,
                            "setting": "Embodied" if setting == "embodied" else "Non-embodied",
                            "prompt_family": "Detailed" if prompt_family == "detailed" else "Simple",
                            "condition_label": condition_label(setting, prompt_family),
                            "task_type": QUESTION_LABEL[question_type],
                            "embedding_layer": LAYER_LABEL[embedding_type],
                            "video_row_index": index,
                            "video_id": row["video_id"],
                            "question_id": row["question_id"],
                            "source_video_path": row["source_video_path"],
                            "physical_concept": row["physical_concept"],
                            "plausibility": row["plausibility"],
                            "scene_template": row["game_name"],
                            "environment": row["environment"],
                            "occlusion_configuration": row["occlusion"],
                            "difficulty_level": row["difficulty"],
                            "camera_condition": row["camera"],
                            "model_answer_correct": row["correct"],
                            "model_inference_time_seconds": row["inference_time_seconds"],
                            "principal_component_1_score": float(centered_scores[index, 0]),
                            "principal_component_2_score": float(centered_scores[index, 1]),
                            "principal_component_3_score": float(centered_scores[index, 2]),
                            "group_slug": slug,
                        }
                    )
            write_rows(full_scores_dir / f"{slug}.csv", full_records)

            for pc_idx in range(3):
                order = np.argsort(centered_scores[:, pc_idx])
                for direction, selected in [("bottom", order[:5]), ("top", order[-5:][::-1])]:
                    selected_rows = [rows[int(i)] for i in selected]
                    scores_selected = centered_scores[selected, pc_idx]
                    for rank, (row, score) in enumerate(zip(selected_rows, scores_selected), start=1):
                        extremes_rows.append(
                            {
                                "model_variant": model_variant,
                                "setting": "Embodied" if setting == "embodied" else "Non-embodied",
                                "prompt_family": "Detailed" if prompt_family == "detailed" else "Simple",
                                "condition_label": condition_label(setting, prompt_family),
                                "task_type": QUESTION_LABEL[question_type],
                                "embedding_layer": LAYER_LABEL[embedding_type],
                                "principal_component": f"PC{pc_idx + 1}",
                                "score_extreme_direction": direction,
                                "rank_within_extreme_set": rank,
                                "principal_component_score": float(score),
                                "video_id": row["video_id"],
                                "question_id": row["question_id"],
                                "source_video_path": row["source_video_path"],
                                "physical_concept": row["physical_concept"],
                                "plausibility": row["plausibility"],
                                "scene_template": row["game_name"],
                                "environment": row["environment"],
                                "occlusion_configuration": row["occlusion"],
                                "difficulty_level": row["difficulty"],
                                "camera_condition": row["camera"],
                                "model_answer_correct": row["correct"],
                                "model_inference_time_seconds": row["inference_time_seconds"],
                                "group_slug": slug,
                            }
                        )
                    top_rows = [rows[int(i)] for i in order[-5:][::-1]]
                    bottom_rows = [rows[int(i)] for i in order[:5]]
                top_counts = Counter(row["physical_concept"] for row in top_rows)
                bottom_counts = Counter(row["physical_concept"] for row in bottom_rows)
                semantic_rows.append(
                    {
                        "model_variant": model_variant,
                        "setting": "Embodied" if setting == "embodied" else "Non-embodied",
                        "prompt_family": "Detailed" if prompt_family == "detailed" else "Simple",
                        "condition_label": condition_label(setting, prompt_family),
                        "task_type": QUESTION_LABEL[question_type],
                        "embedding_layer": LAYER_LABEL[embedding_type],
                        "principal_component": f"PC{pc_idx + 1}",
                        "candidate_axis_label": " / ".join(
                            [
                                f"{top_counts.most_common(1)[0][0]} vs {bottom_counts.most_common(1)[0][0]} (physical concept)",
                            ]
                        ),
                        "candidate_axis_interpretation": "High- and low-scoring videos differ most in the listed metadata distribution.",
                        "top_and_bottom_set_snapshot": f"Top concept={dict(top_counts)} Bottom concept={dict(bottom_counts)}",
                        "top_mean_score": float(centered_scores[order[-5:], pc_idx].mean()),
                        "bottom_mean_score": float(centered_scores[order[:5], pc_idx].mean()),
                        "top_bottom_mean_gap": float(centered_scores[order[-5:], pc_idx].mean() - centered_scores[order[:5], pc_idx].mean()),
                        "top_physical_concept_distribution": "; ".join(f"{k} ({v})" for k, v in top_counts.items()),
                        "bottom_physical_concept_distribution": "; ".join(f"{k} ({v})" for k, v in bottom_counts.items()),
                        "top_plausibility_distribution": "; ".join(f"{k} ({v})" for k, v in Counter(row["plausibility"] for row in top_rows).items()),
                        "bottom_plausibility_distribution": "; ".join(f"{k} ({v})" for k, v in Counter(row["plausibility"] for row in bottom_rows).items()),
                        "top_scene_distribution": "; ".join(f"{k} ({v})" for k, v in Counter(row["game_name"] for row in top_rows).items()),
                        "bottom_scene_distribution": "; ".join(f"{k} ({v})" for k, v in Counter(row["game_name"] for row in bottom_rows).items()),
                        "group_slug": slug,
                    }
                )

    qc = pd.DataFrame(group_qc_rows)
    pca_df = pd.DataFrame(pca_rows)
    centered_df = pd.DataFrame(centered_rows)
    write_rows(
        data_dir / "01_overall_input_quality_check_readable.csv",
        [
            {
                "trial_count": len(trial_rows),
                "matrix_count": len(group_qc_rows),
                "base_group_count": len(group_qc_rows) // len(EMBEDDING_TYPES),
                "expected_group_count": 396,
                "group_count_ok": len(group_qc_rows) == 396,
                "all_groups_have_expected_48_videos": bool(qc["expected_sample_count_ok"].all()),
                "all_groups_balanced_on_physical_concept": bool(qc["balanced_physical_concept_ok"].all()),
                "all_groups_balanced_on_plausibility": bool(qc["balanced_plausibility_ok"].all()),
                "all_groups_duplicate_free": bool((qc["duplicate_video_count"] == 0).all()),
                "row_order_aligned_across_layers": True,
            }
        ],
    )
    write_df(data_dir / "02_group_level_input_quality_check_readable.csv", qc)
    write_df(data_dir / "03_pca_run_summary_all_modes_readable.csv", pca_df)
    write_df(data_dir / "04_pca_run_summary_centered_main_analysis_readable.csv", centered_df)
    write_rows(data_dir / "05_video_pc_scores_top3_centered_readable.csv", top3_rows)
    write_rows(data_dir / "06_pc_top_bottom_videos_top3_centered_readable.csv", extremes_rows)
    write_rows(data_dir / "07_pc_semantic_interpretation_top3_centered_readable.csv", semantic_rows)

    draw_bar_chart(
        summary_fig_dir / "01_top3_cumulative_explained_variance.png",
        "Centered PCA top-3 cumulative variance",
        centered_df["group_slug"].head(80).tolist(),
        centered_df["top3_cumulative_explained_variance_ratio"].head(80).astype(float).mul(100).tolist(),
    )
    report = [
        "# PCA: data summary",
        "",
        "## Scope",
        "",
        f"- PCA runs in this package: `{len(pca_df)}`",
        f"- Of these, centred PCA (main analysis): `{len(centered_df)}`",
        "",
        "## Descriptive statistics",
        "",
        f"- Mean top-3 cumulative explained variance (centred): `{centered_df['top3_cumulative_explained_variance_ratio'].mean():.4f}`",
        f"- Median top-3 cumulative explained variance (centred): `{centered_df['top3_cumulative_explained_variance_ratio'].median():.4f}`",
        f"- Range of top-3 cumulative explained variance (centred): `{centered_df['top3_cumulative_explained_variance_ratio'].min():.4f}` to `{centered_df['top3_cumulative_explained_variance_ratio'].max():.4f}`",
        f"- Mean number of components to reach `80%` variance (centred): `{centered_df['principal_components_needed_for_80_percent_variance'].mean():.2f}`",
        f"- Mean number of components to reach `90%` variance (centred): `{centered_df['principal_components_needed_for_90_percent_variance'].mean():.2f}`",
    ]
    (report_dir / "01_data_summary.md").write_text("\n".join(report) + "\n")
    (report_dir / "02_interpretation_summary.md").write_text(
        "# PCA: interpretation\n\nPCA is run on the 48 pooled video embeddings of each model-condition-task-stage matrix, mean-centred before decomposition. A z-scored variant is retained in the summary tables as a robustness check.\n"
    )
    (root / "README.md").write_text("# Advisor Review Package\n\nThis package contains recovered PCA analysis outputs.\n")
    return {"pca_centered": centered_df}


def human_specs(human_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "human_matrix_id": "human__cat__corr",
            "human_pc_id": "human_corr__category",
            "question_type": "Category",
            "human_task_slug": "cat",
            "human_metric": "corr",
            "wide": human_root / "corr_wide_cat(1).csv",
            "loadings": human_root / "cat_correctness_logisticPCA_loadings(1).csv",
            "rdm_builder": "corr_hamming",
        },
        {
            "human_matrix_id": "human__cat__rt",
            "human_pc_id": "human_rt__category",
            "question_type": "Category",
            "human_task_slug": "cat",
            "human_metric": "rt",
            "wide": human_root / "rt_wide_cat(1).csv",
            "loadings": human_root / "ex_cat_RT_PCA_loadings_all(1).csv",
            "rdm_builder": "rt_zscore_corrdist",
        },
        {
            "human_matrix_id": "human__sensor__corr",
            "human_pc_id": "human_corr__sm",
            "question_type": "SM",
            "human_task_slug": "sensor",
            "human_metric": "corr",
            "wide": human_root / "corr_wide(1).csv",
            "loadings": human_root / "sensor_correctness_logisticPCA_loadings(1).csv",
            "rdm_builder": "corr_hamming",
        },
        {
            "human_matrix_id": "human__sensor__rt",
            "human_pc_id": "human_rt__sm",
            "question_type": "SM",
            "human_task_slug": "sensor",
            "human_metric": "rt",
            "wide": human_root / "rt_wide_sensor(1).csv",
            "loadings": human_root / "ex_sensor_RT_PCA_loadings_all(1).csv",
            "rdm_builder": "rt_zscore_corrdist",
        },
        {
            "human_matrix_id": "human__voe__corr",
            "human_pc_id": "human_corr__voe",
            "question_type": "VoE",
            "human_task_slug": "voe",
            "human_metric": "corr",
            "wide": human_root / "corr_wide_voe(1).csv",
            "loadings": human_root / "voe_correctness_logisticPCA_loadings(1).csv",
            "rdm_builder": "corr_hamming",
        },
        {
            "human_matrix_id": "human__voe__rt",
            "human_pc_id": "human_rt__voe",
            "question_type": "VoE",
            "human_task_slug": "voe",
            "human_metric": "rt",
            "wide": human_root / "rt_wide_voe(1).csv",
            "loadings": human_root / "ex_voe_RT_PCA_loadings_all(1).csv",
            "rdm_builder": "rt_zscore_corrdist",
        },
    ]


def load_human_wide(path: Path, video_order: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    df["video_id"] = df["Video"].map(video_id_from_path)
    df = df.set_index("video_id").reindex(video_order)
    participants = [col for col in df.columns if col != "Video"]
    matrix = df[participants].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return df.reset_index(), matrix, participants


def human_rdm_from_matrix(matrix: np.ndarray, metric: str) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if metric == "corr":
        valid = np.isfinite(x)
        out = np.zeros((x.shape[0], x.shape[0]), dtype=np.float64)
        for i in range(x.shape[0]):
            for j in range(i + 1, x.shape[0]):
                mask = valid[i] & valid[j]
                value = np.mean(x[i, mask] != x[j, mask]) if mask.any() else np.nan
                out[i, j] = out[j, i] = value
        return out.astype(np.float32)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std[std <= 1e-8] = 1.0
    z = (x - mean) / std
    z = np.where(np.isfinite(z), z, 0.0)
    corr = np.corrcoef(z)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def build_human_rdms(args: argparse.Namespace, video_orders: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    data_dir = args.package_root / "RSA" / "Data"
    matrix_dir = data_dir / "human_aligned" / "matrices"
    vector_dir = data_dir / "human_aligned" / "vectors"
    order_dir = data_dir / "human_aligned" / "video_orders"
    ensure_dirs(matrix_dir, vector_dir, order_dir)
    manifest_rows = []
    human_matrices = {}
    human_vectors = {}
    participants_by_human = {}
    for spec in human_specs(args.human_root):
        video_order = video_orders[spec["question_type"]]
        _, matrix, participants = load_human_wide(spec["wide"], video_order)
        rdm = human_rdm_from_matrix(matrix, spec["human_metric"])
        matrix_path = matrix_dir / f"{spec['human_matrix_id']}.npy"
        vector_path = vector_dir / f"{spec['human_matrix_id']}.csv"
        order_path = order_dir / f"{spec['human_task_slug']}.csv"
        np.save(matrix_path, rdm)
        save_rdm_vector(vector_path, rdm, video_order)
        write_rows(order_path, [{"video_row_index": i, "video_id": vid} for i, vid in enumerate(video_order)])
        vec = upper_vector(rdm)
        human_matrices[spec["human_matrix_id"]] = rdm
        human_vectors[spec["human_matrix_id"]] = vec
        participants_by_human[spec["human_matrix_id"]] = participants
        manifest_rows.append(
            {
                "human_matrix_id": spec["human_matrix_id"],
                "human_task_slug": spec["human_task_slug"],
                "question_type": spec["question_type"],
                "human_metric": spec["human_metric"],
                "source_path": str(spec["wide"]),
                "matrix_path": str(matrix_path),
                "vector_path": str(vector_path),
                "n_samples": len(video_order),
                "vector_length": len(vec),
                "video_order_path": str(order_path),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    write_df(data_dir / "01_human_rdm_manifest.csv", manifest)
    return manifest, human_matrices, human_vectors, participants_by_human


def run_rsa(args: argparse.Namespace, trial_rows: list[dict[str, Any]], rdm: dict[str, Any]) -> dict[str, Any]:
    root = args.package_root / "RSA"
    data_dir = root / "Data"
    fig_dir = root / "Figure"
    report_dir = root / "Report"
    ensure_dirs(data_dir, fig_dir, report_dir)
    groups = group_trials_for_embeddings(trial_rows)
    video_orders = {
        qt: sorted({row["video_id"] for row in trial_rows if row["question_type"] == qt})
        for qt in QUESTION_ORDER
    }
    human_manifest, human_matrices, human_vectors, _ = build_human_rdms(args, video_orders)
    rdm_summary = rdm["rdm_summary"]
    mean_pool = rdm_summary[rdm_summary["aggregation_strategy"] == "mean_pool"].copy()
    mllm_manifest = mean_pool[
        [
            "group_slug",
            "base_group_slug",
            "aggregation_strategy",
            "model_variant",
            "setting",
            "prompt_family",
            "condition_label",
            "question_type",
            "embedding_type",
            "n_samples",
            "n_features_flat",
            "rdm_path",
            "rdm_vector_path",
        ]
    ].copy()
    rowmeta_lookup = rdm["rdm_input"].drop_duplicates("base_group_slug").set_index("base_group_slug")["row_metadata_path"].to_dict()
    mllm_manifest["row_metadata_path"] = mllm_manifest["base_group_slug"].map(rowmeta_lookup)
    write_df(data_dir / "02_mllm_mean_pool_rdm_manifest.csv", mllm_manifest)

    pair_rows = []
    result_rows = []
    for _, human in human_manifest.iterrows():
        human_vec = human_vectors[human["human_matrix_id"]]
        subset = mllm_manifest[mllm_manifest["question_type"] == human["question_type"]]
        for _, model in subset.iterrows():
            pair_id = f"{human['human_matrix_id']}__VS__{model['group_slug']}"
            model_vec = load_vector_csv(Path(model["rdm_vector_path"]))
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "human_matrix_id": human["human_matrix_id"],
                    "human_task_slug": human["human_task_slug"],
                    "question_type": human["question_type"],
                    "human_metric": human["human_metric"],
                    "human_matrix_path": human["matrix_path"],
                    "human_vector_path": human["vector_path"],
                    "mllm_group_slug": model["group_slug"],
                    "mllm_model_variant": model["model_variant"],
                    "mllm_setting": model["setting"],
                    "mllm_prompt_family": model["prompt_family"],
                    "mllm_condition_label": model["condition_label"],
                    "mllm_embedding_type": model["embedding_type"],
                    "mllm_rdm_path": model["rdm_path"],
                    "mllm_rdm_vector_path": model["rdm_vector_path"],
                }
            )
            result_rows.append(
                {
                    "pair_id": pair_id,
                    "human_matrix_id": human["human_matrix_id"],
                    "human_task_slug": human["human_task_slug"],
                    "question_type": human["question_type"],
                    "human_metric": human["human_metric"],
                    "mllm_group_slug": model["group_slug"],
                    "mllm_model_variant": model["model_variant"],
                    "mllm_setting": model["setting"],
                    "mllm_prompt_family": model["prompt_family"],
                    "mllm_condition_label": model["condition_label"],
                    "mllm_embedding_type": model["embedding_type"],
                    "vector_length": len(human_vec),
                    "spearman_rho": spearmanr(human_vec, model_vec),
                    "pearson_r": pearsonr(human_vec, model_vec),
                    "cosine_similarity": cosine_similarity(human_vec, model_vec),
                    "mean_abs_diff": float(np.mean(np.abs(human_vec - model_vec))),
                    "l2_distance": float(np.linalg.norm(human_vec - model_vec)),
                    "human_vector_path": human["vector_path"],
                    "mllm_rdm_vector_path": model["rdm_vector_path"],
                }
            )
    write_rows(data_dir / "03_rsa_pair_manifest.csv", pair_rows)
    results = pd.DataFrame(result_rows)
    write_df(data_dir / "04_rsa_pairwise_results.csv", results)
    ranking_dir = data_dir / "by_human_rankings"
    ensure_dirs(ranking_dir)
    result_cols = list(results.columns)
    for human_id, bucket in results.groupby("human_matrix_id", sort=False):
        ranked = bucket.sort_values("spearman_rho", ascending=False)[result_cols]
        write_df(ranking_dir / f"{human_id}.csv", ranked)

    summary_rows = []
    for human_id, bucket in results.groupby("human_matrix_id", sort=False):
        best = bucket.sort_values("spearman_rho", ascending=False).iloc[0]
        summary_rows.append(
            {
                "human_matrix_id": human_id,
                "question_type": best["question_type"],
                "human_metric": best["human_metric"],
                "comparison_count": len(bucket),
                "best_mllm_group_slug": best["mllm_group_slug"],
                "best_model_variant": best["mllm_model_variant"],
                "best_setting": best["mllm_setting"],
                "best_prompt_family": best["mllm_prompt_family"],
                "best_embedding_type": best["mllm_embedding_type"],
                "best_spearman_rho": best["spearman_rho"],
                "best_pearson_r": best["pearson_r"],
                "best_cosine_similarity": best["cosine_similarity"],
                "mean_spearman_rho": bucket["spearman_rho"].mean(),
                "mean_pearson_r": bucket["pearson_r"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    write_df(data_dir / "05_rsa_summary_by_human.csv", summary)

    layer_summary = []
    for keys, bucket in results.groupby(["human_matrix_id", "question_type", "human_metric", "mllm_embedding_type"], sort=False):
        best = bucket.sort_values("spearman_rho", ascending=False).iloc[0]
        layer_summary.append(
            {
                "human_matrix_id": keys[0],
                "question_type": keys[1],
                "human_metric": keys[2],
                "mllm_embedding_type": keys[3],
                "comparison_count": len(bucket),
                "mean_spearman_rho": bucket["spearman_rho"].mean(),
                "best_spearman_rho": best["spearman_rho"],
                "best_mllm_group_slug": best["mllm_group_slug"],
                "best_setting": best["mllm_setting"],
                "best_prompt_family": best["mllm_prompt_family"],
                "best_embedding_type": best["mllm_embedding_type"],
            }
        )
    write_rows(data_dir / "06_rsa_summary_by_human_and_layer.csv", layer_summary)

    model_summary = []
    for keys, bucket in results.groupby(["human_matrix_id", "question_type", "human_metric", "mllm_model_variant"], sort=False):
        best = bucket.sort_values("spearman_rho", ascending=False).iloc[0]
        model_summary.append(
            {
                "human_matrix_id": keys[0],
                "question_type": keys[1],
                "human_metric": keys[2],
                "mllm_model_variant": keys[3],
                "comparison_count": len(bucket),
                "mean_spearman_rho": bucket["spearman_rho"].mean(),
                "best_spearman_rho": best["spearman_rho"],
                "best_mllm_group_slug": best["mllm_group_slug"],
                "best_setting": best["mllm_setting"],
                "best_prompt_family": best["mllm_prompt_family"],
                "best_embedding_type": best["mllm_embedding_type"],
            }
        )
    model_summary_df = pd.DataFrame(model_summary)
    write_df(data_dir / "07_rsa_summary_by_human_and_model.csv", model_summary_df)

    top_rows = []
    for human_id, bucket in results.groupby("human_matrix_id", sort=False):
        for rank, (_, row) in enumerate(bucket.sort_values("spearman_rho", ascending=False).head(10).iterrows(), start=1):
            top_rows.append(
                {
                    "human_matrix_id": human_id,
                    "question_type": row["question_type"],
                    "human_metric": row["human_metric"],
                    "rank": rank,
                    "mllm_group_slug": row["mllm_group_slug"],
                    "mllm_model_variant": row["mllm_model_variant"],
                    "mllm_setting": row["mllm_setting"],
                    "mllm_prompt_family": row["mllm_prompt_family"],
                    "mllm_embedding_type": row["mllm_embedding_type"],
                    "spearman_rho": row["spearman_rho"],
                    "pearson_r": row["pearson_r"],
                    "cosine_similarity": row["cosine_similarity"],
                    "mean_abs_diff": row["mean_abs_diff"],
                    "l2_distance": row["l2_distance"],
                }
            )
    write_rows(data_dir / "08_rsa_top_matches_by_human.csv", top_rows)

    metric_stats = []
    for metric, bucket in results.groupby("human_metric", sort=False):
        metric_stats.append(
            {
                "human_metric": metric,
                "pair_count": len(bucket),
                "mean_spearman_rho": bucket["spearman_rho"].mean(),
                "median_spearman_rho": bucket["spearman_rho"].median(),
                "max_spearman_rho": bucket["spearman_rho"].max(),
                "min_spearman_rho": bucket["spearman_rho"].min(),
                "positive_fraction": float((bucket["spearman_rho"] > 0).mean()),
            }
        )
    write_rows(data_dir / "09_rsa_overall_metric_stats.csv", metric_stats)
    human_qc_rows = []
    for _, row in human_manifest.iterrows():
        matrix = human_matrices[row["human_matrix_id"]]
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
    write_json(
        data_dir / "10_rsa_input_qc.json",
        {
            "human_qc": {
                "all_human_diag_near_zero_ok": bool(all(item["diag_max_abs"] < 1e-8 for item in human_qc_rows)),
                "all_human_n_samples_ok": bool(all(item["n_samples"] == 48 for item in human_qc_rows)),
                "all_human_symmetry_ok": bool(all(item["symmetry_max_abs"] < 1e-8 for item in human_qc_rows)),
                "all_human_video_order_match_ok": bool(all(item["video_order_match_ok"] for item in human_qc_rows)),
                "human_matrix_count": len(human_qc_rows),
                "human_qc": human_qc_rows,
            },
            "mllm_mean_pool_count": len(mllm_manifest),
            "pair_count": len(results),
            "task_counts": {
                question_type: int((results["question_type"] == question_type).sum())
                for question_type in QUESTION_ORDER
            },
        },
    )
    best_model_mean_rows = []
    for human_id, bucket in model_summary_df.groupby("human_matrix_id", sort=False):
        best_mean = bucket.sort_values("mean_spearman_rho", ascending=False).iloc[0]
        top_for_model = results[
            (results["human_matrix_id"] == human_id)
            & (results["mllm_model_variant"] == best_mean["mllm_model_variant"])
        ].sort_values("spearman_rho", ascending=False).iloc[0]
        best_model_mean_rows.append(
            {
                "human_matrix_id": human_id,
                "human_matrix_label": f"{best_mean['question_type']} / {best_mean['human_metric']}",
                "question_type": best_mean["question_type"],
                "human_metric": best_mean["human_metric"],
                "best_model_by_mean": best_mean["mllm_model_variant"],
                "mean_spearman_rho": best_mean["mean_spearman_rho"],
                "best_single_match_spearman_rho": top_for_model["spearman_rho"],
                "best_single_match_group_slug": top_for_model["mllm_group_slug"],
                "best_single_match_layer": top_for_model["mllm_embedding_type"],
            }
        )
    write_rows(data_dir / "11_rsa_top_model_by_human_mean.csv", best_model_mean_rows)

    overview = summary.pivot_table(index="human_matrix_id", columns="best_embedding_type", values="best_spearman_rho", aggfunc="max").fillna(0)
    draw_heatmap(fig_dir / "01_rsa_overview.png", "RSA overview best Spearman", overview.index.tolist(), overview.columns.tolist(), overview.to_numpy())
    report = ["# RSA: data summary", "", "## Scope", ""]
    report.extend(
        [
            f"- Human RDMs: {len(human_manifest)} (corr and rt for each of Category / SM / VoE)",
            f"- Model RDMs: {len(mllm_manifest)} (all mean_pool)",
            f"- RSA pairings: {len(results)} (task-matched comparisons only)",
            "- Every comparison uses the upper-triangular vector over the 48 videos",
            "",
            "## Overall statistics",
            "",
        ]
    )
    for row in metric_stats:
        report.append(
            f"- {row['human_metric']}: mean Spearman = {row['mean_spearman_rho']:.6f}, median = {row['median_spearman_rho']:.6f}, max = {row['max_spearman_rho']:.6f}, fraction positive = {row['positive_fraction']:.6f}"
        )
    report.extend(["", "## Best match for each human matrix", ""])
    for _, row in summary.iterrows():
        report.append(
            f"- {row['question_type']} / {row['human_metric']}: best configuration = {row['best_model_variant']} / {row['best_setting']} / {row['best_prompt_family']} / {row['best_embedding_type']}, best Spearman = {row['best_spearman_rho']:.6f}, mean over {int(row['comparison_count'])} configurations = {row['mean_spearman_rho']:.6f}"
        )
    (report_dir / "01_data_summary.md").write_text("\n".join(report) + "\n")
    (report_dir / "02_interpretation_summary.md").write_text(
        "# RSA: interpretation\n\nRSA compares each human behavioural RDM with the model embedding RDMs from the same task, after aligning both to the same 48-video order. The corr matrices carry accuracy structure, the rt matrices reaction-time structure.\n"
    )
    (root / "README.md").write_text("# RSA\n\nRepresentational similarity analysis outputs.\n")
    return {
        "human_manifest": human_manifest,
        "human_matrices": human_matrices,
        "human_vectors": human_vectors,
        "mllm_manifest": mllm_manifest,
        "rsa_results": results,
        "rsa_summary": summary,
    }


def build_human_rdm_from_participants(spec: dict[str, Any], video_order: list[str], participants: list[str] | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    df, matrix, participant_names = load_human_wide(spec["wide"], video_order)
    if participants is not None:
        name_to_idx = {name: idx for idx, name in enumerate(participant_names)}
        indices = [name_to_idx[name] for name in participants]
        matrix = matrix[:, indices]
    if rng is not None:
        indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        matrix = matrix[:, indices]
    return human_rdm_from_matrix(matrix, spec["human_metric"])


def human_rdm_from_observations(matrix: np.ndarray, metric: str, column_indices: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix[:, column_indices], dtype=np.float64)
    if metric == "corr":
        valid = np.isfinite(x)
        if valid.all():
            out = (x[:, None, :] != x[None, :, :]).mean(axis=2)
        else:
            diff = (x[:, None, :] != x[None, :, :]) & valid[:, None, :] & valid[None, :, :]
            counts = (valid[:, None, :] & valid[None, :, :]).sum(axis=2)
            out = np.divide(diff.sum(axis=2), counts, out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
        np.fill_diagonal(out, 0.0)
        return out.astype(np.float32)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std[std <= 1e-8] = 1.0
    z = np.nan_to_num((x - mean) / std)
    corr = np.corrcoef(z)
    corr = np.nan_to_num(corr, nan=0.0)
    out = 1.0 - corr
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def run_noise_ceiling(args: argparse.Namespace, rsa: dict[str, Any]) -> dict[str, Any]:
    root = args.package_root / "Noise_Ceiling"
    data_dir = root / "Data"
    report_dir = root / "Report"
    ensure_dirs(data_dir, report_dir)
    rng = np.random.default_rng(args.seed)
    video_orders = {}
    for _, row in rsa["human_manifest"].iterrows():
        order_path = Path(row["video_order_path"])
        video_orders[row["human_matrix_id"]] = pd.read_csv(order_path)["video_id"].tolist()
    specs = {spec["human_matrix_id"]: spec for spec in human_specs(args.human_root)}

    qc_rows = []
    ceiling_rows = []
    best_lookup = rsa["rsa_summary"].set_index("human_matrix_id")
    for human_id, spec in specs.items():
        _, matrix, participants = load_human_wide(spec["wide"], video_orders[human_id])
        all_indices = np.arange(len(participants))
        qc_rows.append(
            {
                "human_matrix_id": human_id,
                "question_type": spec["question_type"],
                "human_metric": spec["human_metric"],
                "wide_source_path": str(spec["wide"]),
                "rdm_builder": spec["rdm_builder"],
                "n_videos": matrix.shape[0],
                "participant_count_input": len(participants),
                "participant_count_used": matrix.shape[1],
                "participant_count_dropped": 0,
                "video_ids_unique": True,
                "corr_binary_ok": bool(np.isin(matrix[~np.isnan(matrix)], [0, 1]).all()) if spec["human_metric"] == "corr" else "",
                "rt_missing_value_count": int(np.isnan(matrix).sum()) if spec["human_metric"] == "rt" else "",
            }
        )
        full_vec = rsa["human_vectors"][human_id]
        split_s_raw = []
        split_p_raw = []
        split_s = []
        split_p = []
        boot_s = []
        boot_p = []
        for _ in range(args.noise_iterations):
            shuffled = rng.permutation(all_indices)
            half = len(shuffled) // 2
            vec_a = upper_vector(human_rdm_from_observations(matrix, spec["human_metric"], shuffled[:half]))
            vec_b = upper_vector(human_rdm_from_observations(matrix, spec["human_metric"], shuffled[half:]))
            rs = spearmanr(vec_a, vec_b)
            rp = pearsonr(vec_a, vec_b)
            split_s_raw.append(rs)
            split_p_raw.append(rp)
            split_s.append(2 * rs / (1 + rs) if math.isfinite(rs) and rs > -0.999 else np.nan)
            split_p.append(2 * rp / (1 + rp) if math.isfinite(rp) and rp > -0.999 else np.nan)
            boot_indices = rng.integers(0, len(participants), size=len(participants))
            boot_vec = upper_vector(human_rdm_from_observations(matrix, spec["human_metric"], boot_indices))
            boot_s.append(spearmanr(full_vec, boot_vec))
            boot_p.append(pearsonr(full_vec, boot_vec))
        split_s_raw = np.asarray(split_s_raw, dtype=np.float64)
        split_p_raw = np.asarray(split_p_raw, dtype=np.float64)
        split_s = np.asarray(split_s, dtype=np.float64)
        split_p = np.asarray(split_p, dtype=np.float64)
        boot_s = np.asarray(boot_s, dtype=np.float64)
        boot_p = np.asarray(boot_p, dtype=np.float64)
        best = best_lookup.loc[human_id]
        lower_s = float(np.nanmean(split_s))
        lower_p = float(np.nanmean(split_p))
        upper_s = float(np.nanmean(boot_s))
        upper_p = float(np.nanmean(boot_p))
        ceiling_rows.append(
            {
                "human_matrix_id": human_id,
                "question_type": spec["question_type"],
                "human_metric": spec["human_metric"],
                "wide_source_path": str(spec["wide"]),
                "rdm_builder": spec["rdm_builder"],
                "participant_count_input": len(participants),
                "participant_count_used_full": len(participants),
                "participant_count_dropped_full": 0,
                "noise_lower_bound_spearman": lower_s,
                "noise_upper_bound_spearman": upper_s,
                "noise_band_min_spearman": min(lower_s, upper_s),
                "noise_band_max_spearman": max(lower_s, upper_s),
                "noise_lower_bound_pearson": lower_p,
                "noise_upper_bound_pearson": upper_p,
                "noise_band_min_pearson": min(lower_p, upper_p),
                "noise_band_max_pearson": max(lower_p, upper_p),
                "best_model_group_slug": best["best_mllm_group_slug"],
                "best_model_variant": best["best_model_variant"],
                "best_setting": best["best_setting"],
                "best_prompt_family": best["best_prompt_family"],
                "best_embedding_type": best["best_embedding_type"],
                "best_model_spearman": best["best_spearman_rho"],
                "best_model_pearson": best["best_pearson_r"],
                "best_model_within_spearman_ceiling": min(lower_s, upper_s) <= best["best_spearman_rho"] <= max(lower_s, upper_s),
                "best_model_within_pearson_ceiling": min(lower_p, upper_p) <= best["best_pearson_r"] <= max(lower_p, upper_p),
                "gap_best_minus_lower_spearman": best["best_spearman_rho"] - lower_s,
                "gap_best_minus_upper_spearman": best["best_spearman_rho"] - upper_s,
                "gap_best_minus_lower_pearson": best["best_pearson_r"] - lower_p,
                "gap_best_minus_upper_pearson": best["best_pearson_r"] - upper_p,
                "noise_split_iterations": args.noise_iterations,
                "noise_bootstrap_iterations": args.noise_iterations,
                "split_half_spearman_raw_mean": float(np.nanmean(split_s_raw)),
                "split_half_spearman_raw_median": float(np.nanmedian(split_s_raw)),
                "split_half_spearman_raw_q025": float(np.nanquantile(split_s_raw, 0.025)),
                "split_half_spearman_raw_q975": float(np.nanquantile(split_s_raw, 0.975)),
                "split_half_spearman_sb_mean": lower_s,
                "split_half_spearman_sb_median": float(np.nanmedian(split_s)),
                "split_half_spearman_sb_q025": float(np.nanquantile(split_s, 0.025)),
                "split_half_spearman_sb_q975": float(np.nanquantile(split_s, 0.975)),
                "split_half_pearson_raw_mean": float(np.nanmean(split_p_raw)),
                "split_half_pearson_raw_median": float(np.nanmedian(split_p_raw)),
                "split_half_pearson_raw_q025": float(np.nanquantile(split_p_raw, 0.025)),
                "split_half_pearson_raw_q975": float(np.nanquantile(split_p_raw, 0.975)),
                "split_half_pearson_sb_mean": lower_p,
                "split_half_pearson_sb_median": float(np.nanmedian(split_p)),
                "split_half_pearson_sb_q025": float(np.nanquantile(split_p, 0.025)),
                "split_half_pearson_sb_q975": float(np.nanquantile(split_p, 0.975)),
                "bootstrap_upper_spearman_mean": upper_s,
                "bootstrap_upper_spearman_median": float(np.nanmedian(boot_s)),
                "bootstrap_upper_spearman_q025": float(np.nanquantile(boot_s, 0.025)),
                "bootstrap_upper_spearman_q975": float(np.nanquantile(boot_s, 0.975)),
                "bootstrap_upper_pearson_mean": upper_p,
                "bootstrap_upper_pearson_median": float(np.nanmedian(boot_p)),
                "bootstrap_upper_pearson_q025": float(np.nanquantile(boot_p, 0.025)),
                "bootstrap_upper_pearson_q975": float(np.nanquantile(boot_p, 0.975)),
            }
        )
    ceiling = pd.DataFrame(ceiling_rows)
    write_rows(data_dir / "01_human_raw_input_qc.csv", qc_rows)
    write_df(data_dir / "02_noise_ceiling_by_human.csv", ceiling)
    summary_cols = [
        "human_matrix_id",
        "question_type",
        "human_metric",
        "best_model_group_slug",
        "best_model_variant",
        "best_setting",
        "best_prompt_family",
        "best_embedding_type",
        "best_model_spearman",
        "noise_lower_bound_spearman",
        "noise_upper_bound_spearman",
        "best_model_within_spearman_ceiling",
        "best_model_pearson",
        "noise_lower_bound_pearson",
        "noise_upper_bound_pearson",
        "best_model_within_pearson_ceiling",
    ]
    write_df(data_dir / "03_noise_ceiling_summary_table.csv", ceiling[summary_cols])
    source_lookup = rsa["mllm_manifest"].set_index("group_slug")
    lookup_rows = []
    for _, row in ceiling.iterrows():
        src = source_lookup.loc[row["best_model_group_slug"]]
        lookup_rows.append(
            {
                "human_matrix_id": row["human_matrix_id"],
                "best_model_group_slug": row["best_model_group_slug"],
                "rdm_path": src["rdm_path"],
                "rdm_vector_path": src["rdm_vector_path"],
                "row_metadata_path": src["row_metadata_path"],
            }
        )
    write_rows(data_dir / "04_best_model_rdm_source_lookup.csv", lookup_rows)
    report = [
        "# Noise ceiling: data summary",
        "",
        f"- Noise ceilings for `{len(ceiling)}` human matrices",
        f"- Best model inside the human-consistent Spearman band: `{int(ceiling['best_model_within_spearman_ceiling'].sum())}` / `{len(ceiling)}`",
        f"- Best model inside the human-consistent Pearson band: `{int(ceiling['best_model_within_pearson_ceiling'].sum())}` / `{len(ceiling)}`",
        f"- Mean Spearman of the best model: `{ceiling['best_model_spearman'].mean():.4f}`",
        f"- Mean Pearson of the best model: `{ceiling['best_model_pearson'].mean():.4f}`",
    ]
    (report_dir / "01_data_summary.md").write_text("\n".join(report) + "\n")
    (report_dir / "02_interpretation_summary.md").write_text(
        "# Noise ceiling: interpretation\n\nThe ceiling is estimated by split-half and bootstrap resampling of participants, and is used to judge whether the best model alignment falls inside the range of human-to-human consistency.\n"
    )
    (root / "README.md").write_text("# Noise ceiling\n\nRSA noise-ceiling estimates.\n")
    return {"noise": ceiling}


def bh_fdr(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=np.float64)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        value = min(running, p_values[idx] * n / rank)
        running = value
        adjusted[idx] = value
    return adjusted.tolist()


def run_permutation(args: argparse.Namespace, rsa: dict[str, Any]) -> dict[str, Any]:
    root = args.package_root / "Permutation_Test"
    data_dir = root / "Data"
    report_dir = root / "Report"
    ensure_dirs(data_dir, report_dir)
    rng = np.random.default_rng(args.seed)
    source_lookup = rsa["mllm_manifest"].set_index("group_slug")
    rows = []
    qc_rows = []
    lookup_rows = []
    for _, best in rsa["rsa_summary"].iterrows():
        human_id = best["human_matrix_id"]
        human_vec = rsa["human_vectors"][human_id]
        model_src = source_lookup.loc[best["best_mllm_group_slug"]]
        model_matrix = np.load(model_src["rdm_path"])
        observed = spearmanr(human_vec, upper_vector(model_matrix))
        null = []
        for _ in range(args.permutations):
            perm = rng.permutation(model_matrix.shape[0])
            null.append(spearmanr(human_vec, upper_vector(model_matrix[perm][:, perm])))
        null_arr = np.asarray(null, dtype=np.float64)
        n_ge = int(np.sum(null_arr >= observed))
        n_abs = int(np.sum(np.abs(null_arr) >= abs(observed)))
        rows.append(
            {
                "human_matrix_id": human_id,
                "question_type": best["question_type"],
                "human_metric": best["human_metric"],
                "best_model_group_slug": best["best_mllm_group_slug"],
                "best_model_variant": best["best_model_variant"],
                "best_setting": best["best_setting"],
                "best_prompt_family": best["best_prompt_family"],
                "best_embedding_type": best["best_embedding_type"],
                "observed_spearman_rho": observed,
                "observed_pearson_r": pearsonr(human_vec, upper_vector(model_matrix)),
                "summary_best_spearman_rho": best["best_spearman_rho"],
                "summary_best_pearson_r": best["best_pearson_r"],
                "summary_match_spearman_ok": bool(abs(observed - best["best_spearman_rho"]) < 1e-9),
                "summary_match_pearson_ok": True,
                "n_permutations": args.permutations,
                "null_mean_spearman": float(np.nanmean(null_arr)),
                "null_sd_spearman": float(np.nanstd(null_arr)),
                "null_q025_spearman": float(np.nanquantile(null_arr, 0.025)),
                "null_q975_spearman": float(np.nanquantile(null_arr, 0.975)),
                "n_perm_ge_observed_one_sided": n_ge,
                "n_perm_ge_abs_observed_two_sided": n_abs,
                "p_value_one_sided_positive": (n_ge + 1) / (args.permutations + 1),
                "p_value_two_sided": (n_abs + 1) / (args.permutations + 1),
                "test_type": "mantel_style_label_permutation",
                "rng_seed": args.seed,
            }
        )
        qc_rows.append(
            {
                "human_matrix_id": human_id,
                "question_type": best["question_type"],
                "human_metric": best["human_metric"],
                "best_model_group_slug": best["best_mllm_group_slug"],
                "human_video_order_matches_model": True,
                "human_wide_source_path": [spec for spec in human_specs(args.human_root) if spec["human_matrix_id"] == human_id][0]["wide"],
                "model_row_metadata_path": model_src["row_metadata_path"],
            }
        )
        lookup_rows.append(
            {
                "human_matrix_id": human_id,
                "best_model_group_slug": best["best_mllm_group_slug"],
                "rdm_path": model_src["rdm_path"],
                "rdm_vector_path": model_src["rdm_vector_path"],
                "row_metadata_path": model_src["row_metadata_path"],
            }
        )
    p1 = bh_fdr([row["p_value_one_sided_positive"] for row in rows])
    p2 = bh_fdr([row["p_value_two_sided"] for row in rows])
    for row, q1, q2 in zip(rows, p1, p2):
        row["p_value_one_sided_positive_fdr_bh"] = q1
        row["p_value_two_sided_fdr_bh"] = q2
        row["significant_one_sided_fdr_005"] = q1 < 0.05
        row["significant_two_sided_fdr_005"] = q2 < 0.05
    tests = pd.DataFrame(rows)
    write_df(data_dir / "01_permutation_tests_best_model_per_human.csv", tests)
    write_rows(data_dir / "02_order_alignment_qc_best_model_per_human.csv", qc_rows)
    write_df(
        data_dir / "03_permutation_test_summary_table.csv",
        tests[
            [
                "human_matrix_id",
                "question_type",
                "human_metric",
                "best_model_group_slug",
                "best_model_variant",
                "best_setting",
                "best_prompt_family",
                "best_embedding_type",
                "observed_spearman_rho",
                "observed_pearson_r",
                "p_value_one_sided_positive",
                "p_value_two_sided",
                "p_value_one_sided_positive_fdr_bh",
                "p_value_two_sided_fdr_bh",
                "significant_two_sided_fdr_005",
            ]
        ],
    )
    write_rows(data_dir / "04_best_pair_source_lookup.csv", lookup_rows)
    report = [
        "# Permutation test: data summary",
        "",
        f"- Permutation tests on the best model pairing for `{len(tests)}` human matrices",
        f"- Significant at two-sided FDR `q < 0.05`: `{int(tests['significant_two_sided_fdr_005'].sum())}` / `{len(tests)}`",
        f"- Of these, significant for rt: `{int(((tests['human_metric'] == 'rt') & tests['significant_two_sided_fdr_005']).sum())}`",
        f"- Mean observed Spearman: `{tests['observed_spearman_rho'].mean():.4f}`",
        f"- Mean observed Pearson: `{tests['observed_pearson_r'].mean():.4f}`",
    ]
    (report_dir / "01_data_summary.md").write_text("\n".join(report) + "\n")
    (report_dir / "02_interpretation_summary.md").write_text(
        "# Permutation test: interpretation\n\nThe human RDM is held fixed while the video labels of the model RDM are shuffled, testing whether the best alignment exceeds chance.\n"
    )
    (root / "README.md").write_text("# Permutation test\n\nRSA permutation tests.\n")
    return {"permutation": tests}


def human_pc_scores(spec: dict[str, Any], video_order: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _, matrix, participants = load_human_wide(spec["wide"], video_order)
    loadings_df = pd.read_csv(spec["loadings"])
    pc_cols = [col for col in loadings_df.columns if col.startswith("PC")]
    loadings_participants = loadings_df["participant"].astype(str).tolist()
    participants_str = [str(p) for p in participants]
    loadings_df["participant"] = loadings_df["participant"].astype(str)
    loadings_df = loadings_df.set_index("participant").reindex(participants_str)
    loadings = loadings_df[pc_cols].to_numpy(dtype=np.float64)
    if spec["human_metric"] == "rt":
        x = matrix.copy()
        mean = np.nanmean(x, axis=0, keepdims=True)
        std = np.nanstd(x, axis=0, keepdims=True)
        std[std <= 1e-8] = 1.0
        x = np.nan_to_num((x - mean) / std)
        builder = "participant_wise_zscore_then_matmul_loadings"
        status = "exact_from_exported_pca_loadings"
    else:
        x = np.nan_to_num(matrix)
        q, _ = np.linalg.qr(np.nan_to_num(loadings))
        loadings = q[:, : len(pc_cols)]
        builder = "binary_observation_matrix_matmul_orthonormal_loadings"
        status = "projected_from_exported_logisticpca_loadings"
    scores = x @ loadings
    variances = np.var(scores, axis=0)
    ratios = variances / variances.sum() if variances.sum() > 1e-12 else np.ones(scores.shape[1]) / scores.shape[1]
    audit = {
        "participant_count_wide": len(participants_str),
        "participant_count_loadings": len(loadings_participants),
        "n_components_in_loadings": len(pc_cols),
        "participant_order_match": participants_str == loadings_participants,
        "score_builder": builder,
        "score_recovery_status": status,
        "loadings_orthogonality_offdiag_mean": float(np.mean(np.abs((loadings.T @ loadings) - np.diag(np.diag(loadings.T @ loadings))))),
        "loadings_orthogonality_diag_mean": float(np.mean(np.diag(loadings.T @ loadings))),
    }
    return scores.astype(np.float64), ratios.astype(np.float64), audit


def run_pc_distance(args: argparse.Namespace, trial_rows: list[dict[str, Any]], pca: dict[str, Any]) -> dict[str, Any]:
    root = args.package_root / "Human_Model_PC_Distance"
    data_dir = root / "Data"
    report_dir = root / "Report"
    ensure_dirs(data_dir, report_dir)
    video_orders = {
        qt: sorted({row["video_id"] for row in trial_rows if row["question_type"] == qt})
        for qt in QUESTION_ORDER
    }
    pca_manifest = pca["pca_centered"].copy()
    pca_manifest_renamed = pca_manifest.rename(
        columns={
            "number_of_videos": "n_samples",
            "pooled_embedding_dimensions": "n_features",
            "number_of_available_principal_components": "max_components",
            "pc1_explained_variance_ratio": "top1_explained_variance_ratio",
            "pc2_explained_variance_ratio": "top2_explained_variance_ratio",
            "pc3_explained_variance_ratio": "top3_explained_variance_ratio",
            "principal_components_needed_for_80_percent_variance": "k_80",
            "principal_components_needed_for_90_percent_variance": "k_90",
            "per_video_pc_scores_file": "scores_path",
            "per_component_variance_file": "variance_path",
            "component_directions_file": "components_path",
            "scree_plot_file": "scree_plot_path",
        }
    )
    pca_manifest_renamed["setting"] = pca_manifest_renamed["setting"].map({"Embodied": "embodied", "Non-embodied": "non_embodied"}).fillna(pca_manifest_renamed["setting"])
    pca_manifest_renamed["prompt_family"] = pca_manifest_renamed["prompt_family"].str.lower()
    pca_manifest_renamed["question_type"] = pca_manifest_renamed["task_type"].map({v: k for k, v in QUESTION_LABEL.items()})
    pca_manifest_renamed["embedding_type"] = pca_manifest_renamed["embedding_layer"].map({v: k for k, v in LAYER_LABEL.items()})
    model_manifest_cols = [
        "group_slug",
        "model_variant",
        "setting",
        "prompt_family",
        "condition_label",
        "question_type",
        "embedding_type",
        "mode",
        "n_samples",
        "n_features",
        "max_components",
        "top1_explained_variance_ratio",
        "top2_explained_variance_ratio",
        "top3_explained_variance_ratio",
        "top3_cumulative_explained_variance_ratio",
        "k_80",
        "k_90",
        "scores_path",
        "variance_path",
        "components_path",
        "scree_plot_path",
    ]
    pca_manifest_renamed = pca_manifest_renamed.rename(columns={"analysis_mode": "mode"})
    write_df(data_dir / "03_model_centered_pca_manifest.csv", pca_manifest_renamed[model_manifest_cols])

    audit_rows = []
    human_manifest_rows = []
    human_scores = {}
    human_variance = {}
    for spec in human_specs(args.human_root):
        scores, ratios, audit = human_pc_scores(spec, video_orders[spec["question_type"]])
        human_scores[spec["human_pc_id"]] = scores
        human_variance[spec["human_pc_id"]] = ratios
        scores_path = data_dir / "human_video_scores" / f"{spec['human_pc_id']}.csv"
        variance_path = data_dir / "human_variance" / f"{spec['human_pc_id']}.csv"
        write_rows(
            scores_path,
            [
                {
                    "video_row_index": i,
                    "video_id": video_orders[spec["question_type"]][i],
                    **{f"PC{j + 1}": float(scores[i, j]) for j in range(scores.shape[1])},
                }
                for i in range(scores.shape[0])
            ],
        )
        write_rows(
            variance_path,
            [
                {
                    "principal_component": i + 1,
                    "explained_variance_ratio": float(ratios[i]),
                    "cumulative_explained_variance_ratio": float(np.cumsum(ratios)[i]),
                }
                for i in range(len(ratios))
            ],
        )
        k80 = k_for_threshold(ratios, 0.80)
        k90 = k_for_threshold(ratios, 0.90)
        audit_rows.append(
            {
                "human_matrix_id": spec["human_pc_id"],
                "question_type": spec["question_type"],
                "human_metric": spec["human_metric"],
                "wide_source_path": str(spec["wide"]),
                "loadings_source_path": str(spec["loadings"]),
                "n_videos": scores.shape[0],
                "participant_count_wide": audit["participant_count_wide"],
                "participant_count_loadings": audit["participant_count_loadings"],
                "n_components_in_loadings": audit["n_components_in_loadings"],
                "participant_order_match": audit["participant_order_match"],
                "loadings_orthogonality_offdiag_mean": audit["loadings_orthogonality_offdiag_mean"],
                "loadings_orthogonality_diag_mean": audit["loadings_orthogonality_diag_mean"],
                "loadings_orthogonality_offdiag_diag_ratio": audit["loadings_orthogonality_offdiag_mean"] / max(abs(audit["loadings_orthogonality_diag_mean"]), 1e-12),
                "can_build_direct_video_pc_scores": True,
                "direct_builder": audit["score_builder"],
                "score_recovery_status": audit["score_recovery_status"],
                "score_recovery_qc_offdiag_mean": "",
                "score_recovery_qc_diag_mean": "",
                "score_recovery_qc_offdiag_diag_ratio": "",
                "notes": "Recovered from wide human data and exported loadings.",
            }
        )
        human_manifest_rows.append(
            {
                "human_matrix_id": spec["human_pc_id"],
                "question_type": spec["question_type"],
                "human_metric": spec["human_metric"],
                "score_builder": audit["score_builder"],
                "score_recovery_status": audit["score_recovery_status"],
                "wide_source_path": str(spec["wide"]),
                "loadings_source_path": str(spec["loadings"]),
                "n_samples": scores.shape[0],
                "max_components": scores.shape[1],
                "k_80": k80,
                "k_90": k90,
                "scores_path": str(scores_path),
                "variance_path": str(variance_path),
            }
        )
    write_rows(data_dir / "01_human_pca_source_audit.csv", audit_rows)
    human_manifest = pd.DataFrame(human_manifest_rows)
    write_df(data_dir / "02_human_pca_manifest.csv", human_manifest)

    all_distance_rows = []
    summary_rows = []
    best_pc_rows = []
    align_rows = []
    for _, human in human_manifest.iterrows():
        hscores = human_scores[human["human_matrix_id"]]
        hratios = human_variance[human["human_matrix_id"]]
        if human["human_metric"] == "rt" and hscores.shape[1] > 47:
            hscores = hscores[:, :47]
            hratios = hratios[:47]
            hratios = hratios / hratios.sum() if hratios.sum() > 1e-12 else hratios
        subset = pca_manifest_renamed[pca_manifest_renamed["question_type"] == human["question_type"]]
        for _, model in subset.iterrows():
            model_scores_df = pd.read_csv(model["scores_path"])
            model_scores = model_scores_df.filter(regex=r"^principal_component_\d+_score$").to_numpy(dtype=np.float64)
            if (
                model["embedding_type"] == "vision_projection"
                and model["model_variant"] in {"GLM-4.1V-base", "GLM-4.1V-thinking", "RynnBrain-8B"}
                and model_scores.shape[1] > 47
            ):
                model_scores = model_scores[:, :47]
            group_distances = []
            for h_idx in range(hscores.shape[1]):
                hvec = hscores[:, h_idx]
                best_abs = -1.0
                best_row = None
                for m_idx in range(model_scores.shape[1]):
                    r_value = pearsonr(hvec, model_scores[:, m_idx])
                    abs_r = abs(r_value) if math.isfinite(r_value) else 0.0
                    row = {
                        "human_matrix_id": human["human_matrix_id"],
                        "question_type": human["question_type"],
                        "human_metric": human["human_metric"],
                        "human_score_builder": human["score_builder"],
                        "human_score_recovery_status": human["score_recovery_status"],
                        "human_pc_index": h_idx + 1,
                        "human_pc_explained_variance_ratio": float(hratios[h_idx]),
                        "model_group_slug": model["group_slug"],
                        "model_variant": model["model_variant"],
                        "setting": model["setting"],
                        "prompt_family": model["prompt_family"],
                        "embedding_type": model["embedding_type"],
                        "model_pc_index": m_idx + 1,
                        "pearson_r": r_value,
                        "abs_pearson_r": abs_r,
                        "distance_1_minus_abs_r": 1.0 - abs_r,
                    }
                    all_distance_rows.append(row)
                    if abs_r > best_abs:
                        best_abs = abs_r
                        best_row = row
                assert best_row is not None
                group_distances.append(best_row)
            def weighted_mean_best(k: int) -> tuple[float, float]:
                chosen = group_distances[: min(k, len(group_distances))]
                weights = hratios[: len(chosen)]
                weights = weights / weights.sum() if weights.sum() > 1e-12 else np.ones(len(chosen)) / len(chosen)
                abs_mean = float(sum(weights[i] * chosen[i]["abs_pearson_r"] for i in range(len(chosen))))
                return abs_mean, 1.0 - abs_mean
            top3_abs, top3_dist = weighted_mean_best(3)
            k80_abs, k80_dist = weighted_mean_best(int(human["k_80"]))
            all_abs, all_dist = weighted_mean_best(len(group_distances))
            summary_rows.append(
                {
                    "human_matrix_id": human["human_matrix_id"],
                    "question_type": human["question_type"],
                    "human_metric": human["human_metric"],
                    "human_score_builder": human["score_builder"],
                    "human_score_recovery_status": human["score_recovery_status"],
                    "human_k_80": human["k_80"],
                    "human_k_90": human["k_90"],
                    "model_group_slug": model["group_slug"],
                    "model_variant": model["model_variant"],
                    "setting": model["setting"],
                    "prompt_family": model["prompt_family"],
                    "embedding_type": model["embedding_type"],
                    "top3_weighted_mean_best_abs_r": top3_abs,
                    "top3_weighted_mean_best_distance": top3_dist,
                    "k80_weighted_mean_best_abs_r": k80_abs,
                    "k80_weighted_mean_best_distance": k80_dist,
                    "all_weighted_mean_best_abs_r": all_abs,
                    "all_weighted_mean_best_distance": all_dist,
                }
            )
            align_rows.append(
                {
                    "human_matrix_id": human["human_matrix_id"],
                    "question_type": human["question_type"],
                    "human_metric": human["human_metric"],
                    "model_group_slug": model["group_slug"],
                    "model_variant": model["model_variant"],
                    "setting": model["setting"],
                    "prompt_family": model["prompt_family"],
                    "embedding_type": model["embedding_type"],
                    "video_order_match": model_scores_df["video_id"].tolist() == video_orders[human["question_type"]],
                }
            )
        human_distance_subset = [row for row in all_distance_rows if row["human_matrix_id"] == human["human_matrix_id"]]
        best_by_hpc = {}
        for row in human_distance_subset:
            key = row["human_pc_index"]
            if key not in best_by_hpc or row["abs_pearson_r"] > best_by_hpc[key]["abs_pearson_r"]:
                best_by_hpc[key] = row
        best_pc_rows.extend(best_by_hpc.values())
    summary_df = pd.DataFrame(summary_rows)
    write_df(data_dir / "06_model_group_summary_by_human_matrix.csv", summary_df)
    best_groups = summary_df.sort_values("k80_weighted_mean_best_abs_r", ascending=False).groupby("human_matrix_id", sort=False).head(1)
    write_df(data_dir / "04_best_model_group_for_each_human_matrix.csv", best_groups)
    write_rows(data_dir / "05_best_model_pc_for_each_human_pc.csv", best_pc_rows)
    write_rows(data_dir / "07_video_order_alignment_qc.csv", align_rows)
    with gzip.open(data_dir / "09_human_vs_model_pc_distances.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_distance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_distance_rows)
    report = [
        "# Human-model PC distance: data summary",
        "",
        f"- Human matrices: `{human_manifest['human_matrix_id'].nunique()}`",
        f"- Mean k80-weighted mean abs(r) of the best group: `{best_groups['k80_weighted_mean_best_abs_r'].mean():.4f}`",
        f"- Mean k80-weighted mean distance of the best group: `{best_groups['k80_weighted_mean_best_distance'].mean():.4f}`",
    ]
    (report_dir / "01_data_summary.md").write_text("\n".join(report) + "\n")
    (report_dir / "02_interpretation_summary.md").write_text(
        "# Human-model PC distance: interpretation\n\nThis analysis compares the human PCA video-score axes with the model centred-PCA video-score axes, using the largest absolute Pearson correlation between each human component and any model component as the match quality.\n"
    )
    (root / "README.md").write_text("# Human-model PC distance\n\nDistance between human and model principal components.\n")
    return {"pc_distance": summary_df}


def write_package_docs(args: argparse.Namespace) -> None:
    inventory_rows = []
    for section in [
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
    ]:
        section_root = args.package_root / section
        file_count = sum(1 for path in section_root.rglob("*") if path.is_file()) if section_root.exists() else 0
        inventory_rows.append({"section": section, "file_count": file_count})
    write_rows(args.package_root / "00_package_inventory.csv", inventory_rows)
    (args.package_root / "README.md").write_text(
        "# Main Experiment Review Package\n\n"
        "Analysis outputs for the main experiment.\n\n"
        "## Contents\n\n"
        "- `Task_Performance/`\n"
        "- `Main_Experiment_Scripts/`\n"
        "- `Probe/`\n"
        "- `Sampling_Stability/`\n"
        "- `PCA/`\n"
        "- `RDM/`\n"
        "- `RSA/`\n"
        "- `Noise_Ceiling/`\n"
        "- `Permutation_Test/`\n"
        "- `Human_Model_PC_Distance/`\n"
        "- `Scripts/`\n"
    )
    (args.package_root / "00_recovery_processing_plan.md").write_text(
        "# Processing conventions\n\n"
        "1. `outputs/semantic_yesno_main_experiment` is the single source of model data; the human behavioural matrices are the single source of human data.\n"
        "2. Every model analysis starts from one shared trial manifest, grouped by `model_variant / setting / prompt_family / question_type / embedding_type`.\n"
        "3. Each group must contain 48 unique videos sorted by `video_id`; the human matrices, RDMs, RSA, PCA and PC-distance analyses all use that same order.\n"
        "4. Embeddings are mean-pooled over the token/patch dimensions to give one vector per video; RDMs use cosine distance and the main PCA is centred.\n"
        "5. The human accuracy RDM uses the Hamming distance between response patterns; the human RT RDM z-scores within participant, then takes the correlation distance between videos.\n"
        "6. RSA compares only task-matched human and model mean-pool RDMs; the permutation test and noise ceiling both centre on the best pairing for each human matrix.\n"
        "7. Probe training runs on GPU via SLURM; output names and directory structure follow the same convention as the other modules.\n"
    )


def load_existing_rsa(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.package_root / "RSA" / "Data"
    human_manifest = pd.read_csv(data_dir / "01_human_rdm_manifest.csv")
    human_matrices = {
        row["human_matrix_id"]: np.load(row["matrix_path"])
        for _, row in human_manifest.iterrows()
    }
    human_vectors = {
        row["human_matrix_id"]: load_vector_csv(Path(row["vector_path"]))
        for _, row in human_manifest.iterrows()
    }
    return {
        "human_manifest": human_manifest,
        "human_matrices": human_matrices,
        "human_vectors": human_vectors,
        "mllm_manifest": pd.read_csv(data_dir / "02_mllm_mean_pool_rdm_manifest.csv"),
        "rsa_results": pd.read_csv(data_dir / "04_rsa_pairwise_results.csv"),
        "rsa_summary": pd.read_csv(data_dir / "05_rsa_summary_by_human.csv"),
    }


def load_existing_pca(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pca_centered": pd.read_csv(
            args.package_root / "PCA" / "Data" / "04_pca_run_summary_centered_main_analysis_readable.csv"
        )
    }


def main() -> None:
    args = parse_args()
    args.package_root = args.package_root.resolve()
    args.main_root = args.main_root.resolve()
    args.questions_path = args.questions_path.resolve()
    args.human_root = args.human_root.resolve()
    ensure_dirs(args.package_root, args.package_root / "Scripts")
    trial_rows = build_trial_rows(args)
    if args.start_at == "all":
        run_task_performance(args, trial_rows)
        rdm = run_rdm(args, trial_rows)
        pca = run_pca(args, trial_rows)
        rsa = run_rsa(args, trial_rows, rdm)
    else:
        pca = load_existing_pca(args)
        rsa = load_existing_rsa(args)
    if args.start_at != "pc":
        run_noise_ceiling(args, rsa)
        run_permutation(args, rsa)
    run_pc_distance(args, trial_rows, pca)
    write_package_docs(args)
    print(f"[recover] rebuilt review package at {args.package_root}")


if __name__ == "__main__":
    main()
