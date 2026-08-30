#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def flip_answer(answer: Any) -> str:
    letter = str(answer).strip().upper()
    if letter == "A":
        return "B"
    if letter == "B":
        return "A"
    raise ValueError(f"BA conversion only supports A/B answers, got {answer!r}")


def convert_question(question: dict[str, Any]) -> dict[str, Any]:
    converted = json.loads(json.dumps(question, ensure_ascii=False))
    options = dict(converted.get("options") or {})
    if set(options) != {"A", "B"}:
        raise ValueError(f"Expected exactly A/B options for {converted.get('id')}")
    converted["options"] = {"A": options["B"], "B": options["A"]}
    converted["correct_answer"] = flip_answer(converted.get("correct_answer"))
    metadata = dict(converted.get("metadata") or {})
    metadata["answer_order"] = "BA"
    metadata["source_answer_order"] = "AB"
    converted["metadata"] = metadata
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Create BA-order questions from the AB question file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    payload["questions"] = [convert_question(question) for question in payload["questions"]]
    payload["answer_order"] = "BA"
    payload["source_questions"] = str(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"Wrote {len(payload['questions'])} BA questions to {output_path}")


if __name__ == "__main__":
    main()
