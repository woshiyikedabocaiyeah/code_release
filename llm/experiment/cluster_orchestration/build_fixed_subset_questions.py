#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_VIDEO_IDS = [
    "0f8eb818bd7bed48614d6572dc54034f8cb40c3f616c098fcdc95273d624f878",
    "4cc16b464b92bba7f4db58971b5363596dc309e7275dd4533464b3b008db6896",
    "613bda98c804e7e4c715ac5f7989e4314b788e582f99a585a7f364ffca1b69be",
    "88fdec3b88f1f9df00b7f9cbd68f5e67f1a6fd53f5abb773d30d0f1cd0d89630",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed 12-question sampling-stability subset.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=20260328)
    parser.add_argument("--video-ids", nargs="*", default=DEFAULT_VIDEO_IDS)
    args = parser.parse_args()

    with Path(args.questions).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    selected_ids = set(args.video_ids)
    questions = [question for question in payload["questions"] if question["video_id"] in selected_ids]
    type_order = {"SM": 0, "VoE": 1, "Category": 2}
    questions.sort(key=lambda item: (args.video_ids.index(item["video_id"]), type_order.get(item["question_type"], 99)))
    if len(questions) != len(args.video_ids) * 3:
        raise ValueError(f"Expected {len(args.video_ids) * 3} questions, got {len(questions)}")

    question_ids_by_type: dict[str, list[str]] = {"SM": [], "VoE": [], "Category": []}
    for question in questions:
        question_ids_by_type.setdefault(question["question_type"], []).append(question["id"])

    manifest = {
        "seed": args.seed,
        "sample_size": len(args.video_ids),
        "sampled_video_ids": args.video_ids,
        "question_ids_by_type": question_ids_by_type,
        "counts_by_type": {key: len(value) for key, value in question_ids_by_type.items()},
    }

    subset_payload = dict(payload)
    subset_payload["total_questions"] = len(questions)
    subset_payload["subset_manifest"] = manifest
    subset_payload["questions"] = questions

    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(subset_payload, file, indent=2, ensure_ascii=False)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(f"Wrote {len(questions)} fixed-subset questions to {output_path}")


if __name__ == "__main__":
    main()
