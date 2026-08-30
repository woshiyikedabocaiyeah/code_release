#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt construction for the semantic Yes/No main experiment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

from prompt_templates import (
    build_detail_prompt,
    build_embodied_detail_prompt,
    build_embodied_simple_prompt,
    build_simple_prompt,
)


def normalize_prompt_type(prompt_type: str) -> str:
    """Normalize user-facing prompt types."""
    normalized = (prompt_type or "").strip().lower()
    if normalized in {"embodied_simple", "embodied-simple", "embodied simple", "embodied+simple"}:
        return "embodied_simple"
    if normalized in {"embodied_detail", "embodied-detail", "embodied detail", "embodied+detailed", "embodied+detail"}:
        return "embodied_detail"
    if normalized in {"detail", "detailed"}:
        return "detail"
    if normalized == "simple":
        return "simple"
    raise ValueError(f"Unsupported prompt type: {prompt_type}")


def is_detail_prompt_type(prompt_type: str) -> bool:
    normalized = normalize_prompt_type(prompt_type)
    return normalized in {"detail", "embodied_detail"}


def to_script_prompt_type(prompt_type: str) -> str:
    """Map normalized prompt types to chat script prompt-type values (`simple`/`detailed`)."""
    return "detailed" if is_detail_prompt_type(prompt_type) else "simple"


def load_video_descriptions_from_csv(csv_path: str) -> Dict[str, str]:
    """Load {video_id: description} from CSV with headers: video_filename, description."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Video description CSV not found: {csv_path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return {}

        header_map = {field.strip().lower(): field for field in reader.fieldnames if field}
        filename_col = header_map.get("video_filename")
        description_col = header_map.get("description")
        if not filename_col or not description_col:
            raise ValueError("Expected CSV columns: `video_filename`, `description`")

        mapping: Dict[str, str] = {}
        for row in reader:
            filename = (row.get(filename_col) or "").strip()
            description = (row.get(description_col) or "").strip()
            if not filename or not description:
                continue

            stem = Path(filename).stem.strip()
            if stem:
                mapping[stem] = description
        return mapping


def build_prompt(
    question_data: Dict[str, object],
    prompt_type: str,
    video_descriptions: Dict[str, str],
) -> str:
    """Build the semantic Yes/No prompt for one question."""
    normalized_prompt_type = normalize_prompt_type(prompt_type)
    video_id = str(question_data.get("video_id", "")).strip()
    detail_info = video_descriptions.get(video_id, "")

    if normalized_prompt_type == "detail":
        return build_detail_prompt(question_data, detail_info)
    if normalized_prompt_type == "embodied_detail":
        return build_embodied_detail_prompt(question_data, detail_info)
    if normalized_prompt_type == "embodied_simple":
        return build_embodied_simple_prompt(question_data)

    return build_simple_prompt(question_data)


def build_semantic_prompt(
    question_data: Dict[str, object],
    prompt_type: str,
    video_descriptions: Dict[str, str],
) -> str:
    """Explicit alias for the released main-experiment prompt."""
    return build_prompt(question_data, prompt_type, video_descriptions)
