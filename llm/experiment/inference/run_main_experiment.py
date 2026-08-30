#!/usr/bin/env python3
from __future__ import annotations

"""Main experiment runner for the semantic Yes/No condition.

This is the primary Python entry point for reproducing the released Yes/No
experiment. It builds prompts that ask for a direct "yes" or "no" response,
runs the selected model scripts, and maps the semantic response back to the
question option letters for compatibility with downstream analysis.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "model_config.py").exists() and (path / "run_main_experiment.py").exists():
            return path
    raise RuntimeError("Could not locate Embodied_LLM project root")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from answer_parser import extract_final_answer  # noqa: E402
from model_config import EXPERIMENT_ROOT, VIDEO_DIR, get_experiment_configs  # noqa: E402
from prompt_factory import (  # noqa: E402
    build_semantic_prompt,
    is_detail_prompt_type,
    load_video_descriptions_from_csv,
    normalize_prompt_type,
    to_script_prompt_type,
)


LOGGER = logging.getLogger("semantic_yesno")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

YES_NO_TAG_RE = re.compile(r"<\s*answer\s*>\s*(yes|no)\s*<\s*/\s*answer\s*>", re.I | re.S)
ANSWER_TAG_BODY_RE = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.I | re.S)
BOXED_YESNO_RE = re.compile(r"<\|begin_of_box\|>\s*(yes|no)\s*<\|end_of_box\|>", re.I | re.S)
LETTER_TAG_RE = re.compile(r"<\s*answer\s*>\s*([A-D])\s*<\s*/\s*answer\s*>", re.I | re.S)
FINAL_YESNO_RE = re.compile(
    r"\b(?:final answer|answer|the answer is|so the answer is|therefore the answer is)"
    # The full-width colon is functional: Chinese-origin checkpoints can emit one.
    r"\s*[:：-]?\s*(yes|no)\b",
    re.I,
)
STANDALONE_YESNO_RE = re.compile(r"^\s*(yes|no)\s*[\.\!。]?\s*$", re.I)
YESNO_WORD_RE = re.compile(r"\b(yes|no)\b", re.I)


DEFAULT_VIDEO_DESCRIPTIONS_CSV = str(Path(EXPERIMENT_ROOT) / "detailprompt.csv")
DEFAULT_PROMPT_TYPES = ["simple", "detail", "embodied_simple", "embodied_detail"]
SETTING_PROMPTS = {
    "non_embodied": ["simple", "detail"],
    "embodied": ["embodied_simple", "embodied_detail"],
}
OUTPUT_FILES = {
    "non_embodied": "semantic_yesno_non_embodied_results.json",
    "embodied": "semantic_yesno_embodied_results.json",
}
ARTIFACT_DIRS = {
    "non_embodied": "semantic_yesno_non_embodied_artifacts",
    "embodied": "semantic_yesno_embodied_artifacts",
}

_OPTION_ONLY_PATTERN = re.compile(r"^\s*([A-Da-d])\s*[\.\)]?\s*$")


def normalize_option_letter(value: Any) -> Optional[str]:
    """Return A/B/C/D when value is a standalone option letter."""
    if value is None:
        return None
    match = _OPTION_ONLY_PATTERN.match(str(value))
    if match:
        return match.group(1).upper()
    return None


def resolve_model_answer(
    model_json: Optional[Dict[str, Any]],
    stdout_text: str = "",
) -> Tuple[Optional[str], str, str]:
    """Resolve a final option letter from model JSON or stdout text."""
    if model_json is not None:
        final_answer = normalize_option_letter(model_json.get("final_answer"))
        if final_answer is not None:
            return final_answer, "final_answer_field", "ok"

        response_field = normalize_option_letter(model_json.get("response"))
        if response_field is not None:
            return response_field, "response_field", "ok"

        response_text = str(model_json.get("response") or "")
        if response_text:
            answer, method = extract_final_answer(response_text)
            if answer is not None:
                return answer, f"response_text_{method}", "ok"

        for key in ("original_output", "raw_response"):
            text = str(model_json.get(key) or "")
            if not text:
                continue
            answer, method = extract_final_answer(text)
            if answer is not None:
                return answer, f"{key}_{method}", "ok"

    stdout = str(stdout_text or "")
    if stdout:
        answer, method = extract_final_answer(stdout)
        if answer is not None:
            return answer, f"stdout_{method}", "ok"

    return None, "not_found", "not_found"


def parse_json_from_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from stdout, allowing log lines around it."""
    text = (stdout or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            parsed = json.loads(text[first : last + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

    return None


def get_inference_time_from_metadata(metadata_path: Optional[str]) -> Optional[float]:
    """Read inference time from a model metadata file when available."""
    if not metadata_path or not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        record = data[-1] if isinstance(data, list) and data else data
        if not isinstance(record, dict):
            return None
        for key in ["processing_duration_seconds", "inference_time_seconds"]:
            value = record.get(key)
            if value is not None:
                return float(value)
    except Exception:
        return None
    return None


def build_inference_command(
    mode: str,
    config: Dict[str, Any],
    video_path: str,
    prompt_text: str,
    script_prompt_type: str,
    video_id: str,
    output_dir: str,
) -> List[str]:
    """Build the command used to call one model inference script."""
    cmd = [
        "python3",
        config["script_path"],
        "--video",
        video_path,
        "--prompt",
        prompt_text,
        "--prompt-type",
        script_prompt_type,
        "--video-id",
        video_id,
        "--output",
        output_dir,
    ]

    model_path = config.get("model_path")
    if model_path:
        cmd.extend(["--model", str(Path(model_path).expanduser())])

    mode_arg_name = config.get("mode_arg_name")
    mode_arg_values = config.get("mode_arg_values", {})
    if mode_arg_name:
        mapped_mode = mode_arg_values.get(mode, mode)
        cmd.extend([f"--{mode_arg_name}", mapped_mode])
    elif config.get("supports_think_tag", False):
        cmd.extend(["--mode", "thinking" if mode == "think" else "base"])

    return cmd


def run_single_inference(
    model_name: str,
    mode: str,
    config: Dict[str, Any],
    question_data: Dict[str, Any],
    prompt_type: str,
    script_prompt_type: str,
    prompt_text: str,
    output_dir: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Run one model on one question and return one structured result row."""
    script_dir = os.path.dirname(config["script_path"])
    output_dir = os.path.abspath(output_dir)
    cmd = build_inference_command(
        mode=mode,
        config=config,
        video_path=question_data["video_path"],
        prompt_text=prompt_text,
        script_prompt_type=script_prompt_type,
        video_id=question_data["video_id"],
        output_dir=output_dir,
    )

    result = {
        "model": model_name,
        "mode": mode,
        "prompt_type": prompt_type,
        "question_id": question_data["id"],
        "video_id": question_data["video_id"],
        "question_type": question_data["question_type"],
        "correct_answer": question_data["correct_answer"],
        "model_answer": None,
        "correct": None,
        "inference_time_seconds": None,
        "raw_response": None,
        "original_output": None,
        "thinking_content": None,
        "answer_parse_method": None,
        "answer_parse_status": None,
        "metadata_path": None,
        "artifacts_output_dir": output_dir,
        "success": False,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script_dir,
        )

        if process.returncode != 0:
            stderr = (process.stderr or "").strip()
            stdout = (process.stdout or "").strip()
            message = stderr if stderr else stdout
            result["error"] = f"Return code {process.returncode}: {message[:500]}"
            return result

        stdout = (process.stdout or "").strip()
        model_json = parse_json_from_stdout(stdout)

        if model_json is not None:
            if not bool(model_json.get("success", True)):
                result["error"] = str(model_json.get("error", "model returned success=false"))
                return result

            raw_output_value = model_json.get("original_output")
            if raw_output_value is None:
                raw_output_value = model_json.get("response")

            model_answer, parse_method, parse_status = resolve_model_answer(
                model_json=model_json,
                stdout_text=stdout,
            )
            result["model_answer"] = model_answer
            result["answer_parse_method"] = parse_method
            result["answer_parse_status"] = parse_status
            result["correct"] = result["model_answer"] == result["correct_answer"]
            result["raw_response"] = None if raw_output_value is None else str(raw_output_value)
            result["original_output"] = None if raw_output_value is None else str(raw_output_value)
            result["thinking_content"] = model_json.get("thinking_content")
            metadata_path = model_json.get("metadata_path")
            if metadata_path:
                metadata_path = str(metadata_path)
                if not os.path.isabs(metadata_path):
                    metadata_path = os.path.abspath(os.path.join(script_dir, metadata_path))
            result["metadata_path"] = metadata_path

            inference_time = model_json.get("inference_time_seconds")
            if inference_time is None:
                inference_time = get_inference_time_from_metadata(metadata_path)
            result["inference_time_seconds"] = inference_time
            result["success"] = True
            return result

        model_answer, parse_method, parse_status = resolve_model_answer(
            model_json=None,
            stdout_text=stdout,
        )
        result["model_answer"] = model_answer
        result["answer_parse_method"] = parse_method
        result["answer_parse_status"] = parse_status
        result["correct"] = result["model_answer"] == result["correct_answer"]
        result["raw_response"] = stdout
        result["original_output"] = stdout
        think_match = re.search(r"<think>(.*?)</think>", stdout, flags=re.DOTALL)
        if think_match:
            result["thinking_content"] = think_match.group(1).strip()
        result["success"] = True
        return result

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout}s"
    except Exception as exc:
        result["error"] = str(exc)

    return result


def save_results(output_file: str, results: List[Dict[str, Any]], start_time: datetime, total_questions: int) -> None:
    """Persist result rows and model-level summary statistics."""
    stats: Dict[str, Dict[str, Any]] = {}
    for row in results:
        key = f"{row['model']}_{row['mode']}_{row.get('prompt_type', 'simple')}"
        bucket = stats.setdefault(key, {"total": 0, "correct": 0, "inference_times": []})
        if row["success"]:
            bucket["total"] += 1
            if row["correct"]:
                bucket["correct"] += 1
            if row["inference_time_seconds"] is not None:
                bucket["inference_times"].append(row["inference_time_seconds"])

    model_stats = {}
    for key, value in stats.items():
        total = value["total"]
        accuracy = (value["correct"] / total * 100) if total > 0 else 0.0
        avg_time = (
            sum(value["inference_times"]) / len(value["inference_times"])
            if value["inference_times"]
            else None
        )
        model_stats[key] = {
            "total": total,
            "correct": value["correct"],
            "accuracy_percent": round(accuracy, 2),
            "avg_inference_time_seconds": round(avg_time, 3) if avg_time is not None else None,
        }

    output_data = {
        "experiment_info": {
            "start_time": start_time.isoformat(),
            "last_update": datetime.now().isoformat(),
            "total_questions": total_questions,
            "total_results": len(results),
            "model_statistics": model_stats,
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)


def sanitize_name(value: str) -> str:
    """Sanitize value for use as one path component."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized or "unknown"


def build_artifacts_output_dir(
    artifacts_root: str,
    model_name: str,
    mode: str,
    prompt_type: str,
    question_id: str,
) -> str:
    """Build a unique artifact directory for one inference call."""
    return os.path.join(
        artifacts_root,
        sanitize_name(model_name),
        sanitize_name(mode),
        sanitize_name(prompt_type),
        sanitize_name(question_id),
    )


def resolve_video_path(question: Dict[str, Any]) -> str:
    """Resolve video paths from either public relative paths or local absolute paths."""
    raw_path = str(question.get("video_path") or "").strip()
    video_id = str(question.get("video_id") or "").strip()

    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return str(candidate)

        basename = os.path.basename(raw_path)
        if basename:
            fallback = Path(VIDEO_DIR) / basename
            if fallback.exists():
                return str(fallback)

    if video_id:
        fallback = Path(VIDEO_DIR) / f"{video_id}.mp4"
        if fallback.exists():
            return str(fallback)

    return raw_path


def normalize_prompt_type_list(prompt_types: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in prompt_types:
        value = normalize_prompt_type(item)
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def semantic_for_option(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    text = text.strip("\"'` ")
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def build_semantic_mapping(question_data: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    letter_to_semantic: dict[str, str] = {}
    semantic_to_letter: dict[str, str] = {}
    for letter, option_text in sorted(dict(question_data["options"]).items()):
        letter_text = str(letter).strip().upper()
        semantic = semantic_for_option(option_text)
        if semantic is None:
            raise ValueError(f"Cannot map option {letter_text!r} to yes/no: {option_text!r}")
        letter_to_semantic[letter_text] = semantic
        semantic_to_letter[semantic] = letter_text
    if {"yes", "no"} - set(semantic_to_letter):
        raise ValueError(f"Question does not contain both yes and no options: {question_data['id']}")
    return letter_to_semantic, semantic_to_letter


def normalize_semantic(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in {"yes", "no"} else None


def extract_semantic_yesno(text: str) -> tuple[Optional[str], str]:
    raw = str(text or "").strip()
    if not raw:
        return None, "not_found"

    tag_matches = list(YES_NO_TAG_RE.finditer(raw))
    if tag_matches:
        return tag_matches[-1].group(1).lower(), "answer_tag"

    body_matches = list(ANSWER_TAG_BODY_RE.finditer(raw))
    for match in reversed(body_matches):
        body = re.sub(r"<[^>]+>", " ", match.group(1)).strip()
        first_word = re.match(r"^(yes|no)\b", body, flags=re.I)
        if first_word:
            return first_word.group(1).lower(), "answer_tag_prefix"

    boxed_matches = list(BOXED_YESNO_RE.finditer(raw))
    if boxed_matches:
        return boxed_matches[-1].group(1).lower(), "boxed_yesno"

    final_matches = list(FINAL_YESNO_RE.finditer(raw))
    if final_matches:
        return final_matches[-1].group(1).lower(), "final_phrase"

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines[-20:]):
        line = re.sub(r"</?think>|</?answer>", "", line, flags=re.I).strip()
        match = STANDALONE_YESNO_RE.match(line)
        if match:
            return match.group(1).lower(), "standalone_line"

    tail = raw[-1200:]
    word_matches = list(YESNO_WORD_RE.finditer(tail))
    if word_matches:
        return word_matches[-1].group(1).lower(), "last_yesno_word"

    return None, "not_found"


def extract_letter_fallback(text: str, original_parser_answer: Any) -> tuple[Optional[str], str]:
    letter = str(original_parser_answer or "").strip().upper()
    if letter in {"A", "B", "C", "D"}:
        return letter, "original_letter_parser"
    raw = str(text or "")
    tag_matches = list(LETTER_TAG_RE.finditer(raw))
    if tag_matches:
        return tag_matches[-1].group(1).upper(), "letter_answer_tag"
    return None, "not_found"


def semantic_text_from_result(result: dict[str, Any]) -> str:
    parts = [
        result.get("raw_response"),
        result.get("original_output"),
        result.get("thinking_content"),
    ]
    return "\n".join(str(part) for part in parts if part)


def apply_semantic_parse(result: dict[str, Any], question_data: dict[str, Any]) -> dict[str, Any]:
    letter_to_semantic, semantic_to_letter = build_semantic_mapping(question_data)
    correct_letter = str(question_data["correct_answer"]).strip().upper()
    correct_semantic = letter_to_semantic.get(correct_letter)

    original_letter_answer = result.get("model_answer")
    raw_text = semantic_text_from_result(result)
    semantic_answer, semantic_method = extract_semantic_yesno(raw_text)
    format_compliant = semantic_answer is not None

    if semantic_answer is None:
        fallback_letter, fallback_method = extract_letter_fallback(raw_text, original_letter_answer)
        if fallback_letter in letter_to_semantic:
            semantic_answer = letter_to_semantic[fallback_letter]
            semantic_method = f"letter_fallback_{fallback_method}"

    mapped_letter = semantic_to_letter.get(semantic_answer or "")

    result["prompt_variant"] = "semantic_yesno_v1"
    result["semantic_answer"] = semantic_answer
    result["semantic_correct_answer"] = correct_semantic
    result["semantic_to_option"] = semantic_to_letter
    result["option_to_semantic"] = letter_to_semantic
    result["original_letter_parser_answer"] = original_letter_answer
    result["format_compliant_yesno"] = format_compliant
    result["model_answer"] = mapped_letter
    result["correct"] = semantic_answer == correct_semantic if semantic_answer is not None else False
    result["answer_parse_method"] = f"semantic_yesno_{semantic_method}"
    result["answer_parse_status"] = "ok" if semantic_answer is not None else "not_found"
    return result


def result_key(question_id: str, model_name: str, mode: str, prompt_type: str) -> str:
    return f"{question_id}_{model_name}_{mode}_{prompt_type}"


def load_questions(questions_file: str, video_id: Optional[str], max_questions: Optional[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(questions_file, "r", encoding="utf-8") as file:
        payload = json.load(file)
    questions = payload["questions"]
    for question in questions:
        question["video_path"] = resolve_video_path(question)
        build_semantic_mapping(question)
    if video_id:
        questions = [item for item in questions if item["video_id"] == video_id]
    if max_questions is not None:
        questions = questions[:max_questions]
    if not questions:
        raise ValueError("No questions left after filters")
    return payload, questions


def run_semantic_experiment(
    questions_file: str,
    output_file: str,
    models_to_run: Optional[Sequence[str]] = None,
    resume: bool = False,
    timeout: int = 300,
    max_questions: Optional[int] = None,
    video_id: Optional[str] = None,
    prompt_types: Optional[Sequence[str]] = None,
    video_descriptions_csv: str = DEFAULT_VIDEO_DESCRIPTIONS_CSV,
    save_every: int = 1,
    artifacts_root: Optional[str] = None,
) -> None:
    prompt_types = normalize_prompt_type_list(prompt_types or ["simple"])
    questions_payload, questions = load_questions(questions_file, video_id, max_questions)
    LOGGER.info("Loaded %d questions", len(questions))
    LOGGER.info("Prompt types: %s", ", ".join(prompt_types))

    video_descriptions: dict[str, str] = {}
    if any(is_detail_prompt_type(prompt_type) for prompt_type in prompt_types):
        video_descriptions = load_video_descriptions_from_csv(video_descriptions_csv)
        LOGGER.info("Loaded %d video descriptions", len(video_descriptions))

    configs = get_experiment_configs()
    if models_to_run:
        selected_models = set(models_to_run)
        configs = [cfg for cfg in configs if cfg[0] in selected_models]
    if not configs:
        raise ValueError("No model configurations selected")
    LOGGER.info("Selected %d model configurations", len(configs))

    existing_results: dict[str, dict[str, Any]] = {}
    if resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
        for row in existing_data.get("results", []):
            prompt_type = normalize_prompt_type(row.get("prompt_type", "simple"))
            row["prompt_type"] = prompt_type
            existing_results[result_key(row["question_id"], row["model"], row["mode"], prompt_type)] = row
        LOGGER.info("Loaded %d existing results", len(existing_results))

    results: list[dict[str, Any]] = list(existing_results.values())
    total_tasks = len(questions) * len(configs) * len(prompt_types)
    completed = len(existing_results)
    experiment_start = datetime.now()

    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if artifacts_root is None:
        artifacts_root = str(output_path.parent / f"{output_path.with_suffix('').name}_artifacts")
    artifacts_root = os.path.abspath(artifacts_root)
    os.makedirs(artifacts_root, exist_ok=True)
    LOGGER.info("Artifacts root: %s", artifacts_root)
    LOGGER.info("Total tasks: %d | Already completed: %d", total_tasks, completed)

    for prompt_type in prompt_types:
        script_prompt_type = to_script_prompt_type(prompt_type)
        for model_name, mode, config in configs:
            model_correct = 0
            model_total = 0
            LOGGER.info("Model=%s mode=%s prompt=%s", model_name, mode, prompt_type)
            for question in questions:
                key = result_key(question["id"], model_name, mode, prompt_type)
                if key in existing_results:
                    continue
                completed += 1
                LOGGER.info("[%d/%d] %s %s", completed, total_tasks, question["question_type"], question["id"][:25])
                prompt_text = build_semantic_prompt(question, prompt_type, video_descriptions)
                output_dir = build_artifacts_output_dir(
                    artifacts_root=artifacts_root,
                    model_name=model_name,
                    mode=mode,
                    prompt_type=prompt_type,
                    question_id=question["id"],
                )
                os.makedirs(output_dir, exist_ok=True)
                result = run_single_inference(
                    model_name=model_name,
                    mode=mode,
                    config=config,
                    question_data=question,
                    prompt_type=prompt_type,
                    script_prompt_type=script_prompt_type,
                    prompt_text=prompt_text,
                    output_dir=output_dir,
                    timeout=timeout,
                )
                result = apply_semantic_parse(result, question)
                results.append(result)
                existing_results[key] = result

                if result.get("success"):
                    model_total += 1
                    if result.get("correct"):
                        model_correct += 1
                    LOGGER.info(
                        "  answer=%s semantic=%s expected=%s compliant=%s",
                        result.get("model_answer"),
                        result.get("semantic_answer"),
                        result.get("semantic_correct_answer"),
                        result.get("format_compliant_yesno"),
                    )
                else:
                    LOGGER.warning("  ERROR: %s", str(result.get("error"))[:200])

                if save_every <= 1 or completed % save_every == 0:
                    save_results(
                        str(output_path),
                        results,
                        start_time=experiment_start,
                        total_questions=questions_payload.get("total_questions", len(questions)),
                    )
            if model_total:
                LOGGER.info(
                    "%s(%s,%s): %d/%d %.1f%%",
                    model_name,
                    mode,
                    prompt_type,
                    model_correct,
                    model_total,
                    model_correct / model_total * 100.0,
                )

    save_results(
        str(output_path),
        results,
        start_time=experiment_start,
        total_questions=questions_payload.get("total_questions", len(questions)),
    )
    LOGGER.info("Results written to: %s", output_path)


def selected_prompt_groups(prompt_types: Sequence[str]) -> dict[str, list[str]]:
    selected = set(normalize_prompt_type_list(prompt_types))
    groups: dict[str, list[str]] = {}
    for setting, setting_prompts in SETTING_PROMPTS.items():
        prompts = [prompt for prompt in setting_prompts if prompt in selected]
        if prompts:
            groups[setting] = prompts
    if not groups:
        raise ValueError("No supported prompt types selected")
    return groups


def run_main_experiment(
    run_root: Path,
    questions_file: str,
    models_to_run: Optional[Sequence[str]],
    prompt_types: Sequence[str],
    resume: bool,
    timeout: int,
    max_questions: Optional[int],
    video_id: Optional[str],
    video_descriptions_csv: str,
    save_every: int,
) -> None:
    """Run the released Yes/No experiment and write final results by setting."""
    run_root = run_root.resolve()
    prompt_groups = selected_prompt_groups(prompt_types)

    for setting, setting_prompt_types in prompt_groups.items():
        output_dir = run_root / setting
        output_file = output_dir / OUTPUT_FILES[setting]
        artifacts_root = output_dir / ARTIFACT_DIRS[setting]
        LOGGER.info("Running setting=%s prompts=%s", setting, ",".join(setting_prompt_types))
        run_semantic_experiment(
            questions_file=questions_file,
            output_file=str(output_file),
            models_to_run=models_to_run,
            resume=resume,
            timeout=timeout,
            max_questions=max_questions,
            video_id=video_id,
            prompt_types=setting_prompt_types,
            video_descriptions_csv=video_descriptions_csv,
            save_every=save_every,
            artifacts_root=str(artifacts_root),
        )

    LOGGER.info("Main experiment complete. Results root: %s", run_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the semantic Yes/No main experiment.")
    parser.add_argument("--run-root", type=Path, default=Path(EXPERIMENT_ROOT) / "runs" / "semantic_yesno")
    parser.add_argument("--questions", default=str(Path(EXPERIMENT_ROOT) / "questions.json"))
    parser.add_argument("--models", nargs="+", help="Optional subset of model names to run.")
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=DEFAULT_PROMPT_TYPES,
        help="Prompt types to run: simple detail embodied_simple embodied_detail.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming existing outputs.")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per model call, in seconds.")
    parser.add_argument("--max-questions", type=int, help="Optional question limit for quick validation.")
    parser.add_argument("--video-id", help="Run only one video_id.")
    parser.add_argument("--video-descriptions-csv", default=DEFAULT_VIDEO_DESCRIPTIONS_CSV)
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_main_experiment(
        run_root=args.run_root,
        questions_file=args.questions,
        models_to_run=args.models,
        prompt_types=args.prompt_types,
        resume=not args.no_resume,
        timeout=args.timeout,
        max_questions=args.max_questions,
        video_id=args.video_id,
        video_descriptions_csv=args.video_descriptions_csv,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
