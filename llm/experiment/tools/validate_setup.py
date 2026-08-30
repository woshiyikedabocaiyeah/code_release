#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"OK: {message}")


def main() -> None:
    questions_path = SCRIPT_DIR / "questions.json"
    with questions_path.open("r", encoding="utf-8") as file:
        questions_payload = json.load(file)
    questions = questions_payload["questions"]
    check(len(questions) == 144, "main question count is 144")
    check(len({question["video_id"] for question in questions}) == 48, "main video count is 48")
    check(len(list((ROOT / "experiment_videos").glob("*.mp4"))) == 48, "48 video files copied")

    sys.path.insert(0, str(SCRIPT_DIR))
    import model_config  # noqa: WPS433

    configs = model_config.get_experiment_configs()
    check(len(configs) == 11, "11 model-mode configurations")
    for model_name, _, config in configs:
        check(Path(config["script_path"]).exists(), f"inference script exists for {model_name}")

    ba_path = SCRIPT_DIR / "questions_ba.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_ba_questions.py"),
            "--input",
            str(questions_path),
            "--output",
            str(ba_path),
        ],
        check=True,
    )
    with ba_path.open("r", encoding="utf-8") as file:
        ba_payload = json.load(file)
    check(len(ba_payload["questions"]) == 144, "BA question count is 144")

    subset_path = ROOT / "outputs" / "semantic_yesno_repeat_stability_run" / "fixed_subset_questions.json"
    manifest_path = ROOT / "outputs" / "semantic_yesno_repeat_stability_run" / "fixed_subset_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_fixed_subset_questions.py"),
            "--questions",
            str(questions_path),
            "--output",
            str(subset_path),
            "--manifest",
            str(manifest_path),
        ],
        check=True,
    )
    with subset_path.open("r", encoding="utf-8") as file:
        subset_payload = json.load(file)
    check(len(subset_payload["questions"]) == 12, "sampling fixed subset has 12 questions")
    print("Setup validation complete.")


if __name__ == "__main__":
    main()
