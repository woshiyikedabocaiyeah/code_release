#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for extracting stable final answers from model outputs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


ANSWER_MENTION_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("answer_is", re.compile(r"answer\s+(?:is|would be|should be)\s+([A-D])", re.IGNORECASE)),
    ("final_answer", re.compile(r"final answer\s*[:：]?\s*([A-D])", re.IGNORECASE)),
    ("go_with", re.compile(r"go with\s+([A-D])", re.IGNORECASE)),
    ("choose", re.compile(r"choose\s+(?:option\s+)?([A-D])\b", re.IGNORECASE)),
    ("pick", re.compile(r"i\s+pick\s+([A-D])", re.IGNORECASE)),
    ("select", re.compile(r"i\s+select\s+([A-D])", re.IGNORECASE)),
    (
        "therefore_answer",
        re.compile(r"therefore[, ]+(?:the )?answer\s+(?:is\s+)?([A-D])", re.IGNORECASE),
    ),
)

ANSWER_TAG_PATTERN = re.compile(
    r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
BOXED_PATTERN = re.compile(
    r"<\|begin_of_box\|>\s*([A-D])\s*<\|end_of_box\|>",
    flags=re.IGNORECASE,
)
TRAILING_LINE_PATTERN = re.compile(r"(?im)^\s*([A-D])\s*[\.\)]?\s*$")


def remove_think_sections(text: str) -> str:
    """Remove common thinking sections to avoid matching option letters in reasoning."""
    cleaned = text or ""
    patterns = [
        r"<think>.*?</think>",
        r"<\|begin_of_think\|>.*?<\|end_of_think\|>",
        r"\[thinking\].*?\[/thinking\]",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned


def strip_input_echo(decoded_output: str, decoded_input: str) -> str:
    """Remove echoed input prompt from decoded full output when present."""
    output_text = (decoded_output or "").strip()
    input_text = (decoded_input or "").strip()
    if not output_text:
        return ""
    if input_text:
        index = output_text.find(input_text)
        if index != -1:
            output_text = output_text[index + len(input_text):]
    return output_text.strip()


def clean_response_text(text: str) -> str:
    """Normalize leading assistant prefixes and trim whitespace."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    # The Chinese alternatives are functional: several of the evaluated
    # checkpoints are Chinese-origin and can prefix their reply with the
    # Chinese word for "assistant" and a full-width colon.
    cleaned = re.sub(
        r"^\s*(assistant|助手|assistant_response)\s*[:：]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _extract_letter_from_fragment(fragment: str) -> Optional[str]:
    text = re.sub(r"<\|[^>]+\|>", " ", fragment or "")
    text = text.strip()
    if not text:
        return None

    line_matches = list(re.finditer(r"(?im)^\s*([A-Da-d])\s*[\.\)]?\s*$", text))
    if line_matches:
        return line_matches[-1].group(1).upper()

    tail_match = re.search(r"(?:^|[\s:：>\]\)\}])([A-Da-d])\s*[\.\)]?\s*$", text)
    if tail_match:
        return tail_match.group(1).upper()

    return None


def extract_final_answer(text: str) -> tuple[Optional[str], str]:
    """
    Extract final answer letter from text.
    Returns: (answer, method)
    """
    raw = clean_response_text(text or "")
    if not raw:
        return None, "empty"

    normalized = remove_think_sections(raw)

    tag_matches = list(
        re.finditer(
            r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if tag_matches:
        letter = _extract_letter_from_fragment(tag_matches[-1].group(1))
        if letter is not None:
            return letter, "answer_tag"

    boxed_matches = list(
        re.finditer(
            r"<\|begin_of_box\|>\s*([A-Da-d])\s*<\|end_of_box\|>",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if boxed_matches:
        return boxed_matches[-1].group(1).upper(), "boxed_token"

    latex_boxed = list(
        re.finditer(
            r"\\boxed\s*\{\s*([A-Da-d])\s*\}",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if latex_boxed:
        return latex_boxed[-1].group(1).upper(), "latex_boxed"

    parse_text = re.sub(r"<\|[^>]+\|>", " ", normalized)

    trailing_line_matches = list(re.finditer(r"(?im)^\s*([A-Da-d])\s*[\.\)]?\s*$", parse_text))
    if trailing_line_matches:
        return trailing_line_matches[-1].group(1).upper(), "trailing_line"

    if len(parse_text) <= 500:
        option_prefix_matches = list(
            re.finditer(r"(?im)^\s*([A-Da-d])\s*[:：\-]\s*\S+", parse_text)
        )
        if option_prefix_matches:
            return option_prefix_matches[-1].group(1).upper(), "option_prefix"

    phrase_matches = list(
        re.finditer(
            (
                r"(?:final\s+answer|answer\s+is|my\s+answer\s+is|"
                r"i\s+choose|i\s+pick|i\s+select)\s*[:：]?\s*([A-Da-d])\b"
            ),
            parse_text[-260:],
            flags=re.IGNORECASE,
        )
    )
    if phrase_matches:
        return phrase_matches[-1].group(1).upper(), "answer_phrase"

    trailing_match = re.search(r"(?:^|[\s:：>\]\)\}])([A-Da-d])\s*[\.\)]?\s*$", parse_text)
    if trailing_match:
        return trailing_match.group(1).upper(), "trailing_char"

    return None, "not_found"


def build_answer_payload(primary_text: str, *fallback_texts: str) -> Dict[str, Optional[str]]:
    """
    Build stable answer fields for JSON output.

    Returns keys:
    - response
    - final_answer
    - answer_parse_method
    - answer_parse_status
    """
    candidates = [primary_text, *fallback_texts]
    for idx, candidate in enumerate(candidates):
        answer, method = extract_final_answer(candidate or "")
        if answer is not None:
            if idx > 0:
                method = f"{method}_fallback_{idx}"
            return {
                "response": answer,
                "final_answer": answer,
                "answer_parse_method": method,
                "answer_parse_status": "ok",
            }

    return {
        "response": "?",
        "final_answer": None,
        "answer_parse_method": "not_found",
        "answer_parse_status": "not_found",
    }


def normalize_letter(value: Any) -> Optional[str]:
    """Normalize a value to a valid option letter."""
    if value is None:
        return None
    letter = str(value).strip().upper()
    return letter if letter in {"A", "B", "C", "D"} else None


def extract_answer_mentions(text: str) -> List[Dict[str, Any]]:
    """Find explicit answer-letter mentions in a generated response."""
    mentions: List[Dict[str, Any]] = []
    for pattern_name, pattern in ANSWER_MENTION_PATTERNS:
        for match in pattern.finditer(text or ""):
            letter = normalize_letter(match.group(1))
            if letter is None:
                continue
            mentions.append(
                {
                    "pattern": pattern_name,
                    "letter": letter,
                    "start": match.start(),
                }
            )
    mentions.sort(key=lambda item: item["start"])
    return mentions


def extract_last_answer_tag_letter(text: str) -> Optional[str]:
    """Read the final option letter from the last answer tag, if present."""
    matches = list(ANSWER_TAG_PATTERN.finditer(text or ""))
    if not matches:
        return None
    fragment = matches[-1].group(1)
    boxed = BOXED_PATTERN.search(fragment)
    if boxed:
        return normalize_letter(boxed.group(1))
    trailing_line = list(TRAILING_LINE_PATTERN.finditer(fragment))
    if trailing_line:
        return normalize_letter(trailing_line[-1].group(1))
    fallback = re.search(r"([A-D])", fragment, flags=re.IGNORECASE)
    if fallback:
        return normalize_letter(fallback.group(1))
    return None


def assess_natural_completion(
    raw_generated_output: str,
    normalized_output: str,
) -> Dict[str, Any]:
    """Check whether the raw output ends cleanly at the final answer tag."""
    raw_clean = (raw_generated_output or "").strip()
    normalized_clean = (normalized_output or "").strip()

    if not raw_clean:
        return {
            "is_natural": False,
            "reason": "empty_raw_output",
        }

    raw_for_compare = re.sub(r"^\s*assistant\s*", "", raw_clean, count=1, flags=re.IGNORECASE).strip()
    matches = list(ANSWER_TAG_PATTERN.finditer(raw_for_compare))
    if not matches:
        return {
            "is_natural": False,
            "reason": "missing_full_answer_tag_in_raw",
        }

    last_match = matches[-1]
    trailing_after_answer = raw_for_compare[last_match.end():].strip()
    if trailing_after_answer:
        return {
            "is_natural": False,
            "reason": "extra_text_after_answer_tag",
        }

    natural_segment = raw_for_compare[: last_match.end()].strip()
    if normalized_clean and natural_segment != normalized_clean:
        return {
            "is_natural": False,
            "reason": "normalized_output_differs_from_raw_completion",
        }

    return {
        "is_natural": True,
        "reason": None,
    }


def detect_likely_truncated(text: str, *, answer_parse_method: str = "") -> bool:
    """Flag long outputs that appear to stop mid-sentence before a final answer."""
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    if normalize_letter(stripped) is not None:
        return False
    if TRAILING_LINE_PATTERN.search(stripped[-20:]):
        return False
    if BOXED_PATTERN.search(stripped):
        return False
    if ANSWER_TAG_PATTERN.search(stripped):
        return False
    if len(stripped) < 500:
        return False
    safe_suffixes = (
        "</answer>",
        "<|end_of_box|></answer>",
        ".",
        "!",
        "?",
        "\nA",
        "\nB",
        "\nC",
        "\nD",
    )
    if stripped.endswith(safe_suffixes):
        return False
    if answer_parse_method in {"trailing_line", "answer_tag", "boxed_token", "option_prefix"} and len(stripped) < 1200:
        return False
    if re.search(r"[A-Za-z0-9,;:]$", stripped) and not stripped.endswith((" A", " B", " C", " D")):
        return True
    return False


def assess_answer_quality(
    raw_response: str,
    final_answer: Optional[str],
    answer_parse_method: str,
    *,
    long_threshold: int,
    tail_window: int = 2000,
    raw_generated_output: Optional[str] = None,
    normalized_output: Optional[str] = None,
    require_natural_completion: bool = False,
    require_closed_think: bool = False,
) -> Dict[str, Any]:
    """Assess whether a generated answer is complete and internally consistent."""
    normalized_answer = normalize_letter(final_answer)
    mentions = extract_answer_mentions(raw_response)
    tail_mentions = extract_answer_mentions((raw_response or "")[-tail_window:])
    tail_letters = sorted({item["letter"] for item in tail_mentions})
    answer_tag_letter = extract_last_answer_tag_letter(raw_response)
    natural_completion = assess_natural_completion(
        raw_generated_output if raw_generated_output is not None else raw_response,
        normalized_output if normalized_output is not None else raw_response,
    )

    reasons: List[str] = []
    if normalized_answer is None:
        reasons.append("missing_final_answer")
    if normalized_answer is not None and answer_tag_letter is None:
        reasons.append("missing_answer_tag")
    if answer_parse_method in {"answer_phrase", "trailing_char"}:
        reasons.append(f"fragile_parse_method:{answer_parse_method}")
    if len(raw_response or "") > long_threshold:
        reasons.append(f"long_raw_response:{len(raw_response or '')}")
    if detect_likely_truncated(raw_response, answer_parse_method=answer_parse_method):
        reasons.append("likely_truncated_output")
    if normalized_answer and re.search(rf"\b{normalized_answer}\b\s+is\s+wrong\b", raw_response or "", re.IGNORECASE):
        reasons.append(f"self_contradiction:{normalized_answer}_is_wrong")
    if mentions and normalized_answer and mentions[-1]["letter"] != normalized_answer:
        reasons.append(f"last_answer_mention_mismatch:{mentions[-1]['letter']}->{normalized_answer}")
    if answer_tag_letter and normalized_answer and answer_tag_letter != normalized_answer:
        reasons.append(f"answer_tag_mismatch:{answer_tag_letter}->{normalized_answer}")
    if len(tail_letters) >= 2:
        reasons.append("multiple_tail_answer_mentions")
    if require_closed_think:
        raw_text = raw_generated_output if raw_generated_output is not None else raw_response
        raw_lower = (raw_text or "").lower()
        if "<think>" not in raw_lower:
            reasons.append("missing_open_think_tag")
        elif "</think>" not in raw_lower:
            reasons.append("missing_close_think_tag")
    if require_natural_completion and not natural_completion["is_natural"]:
        reasons.append(f"non_natural_completion:{natural_completion['reason']}")

    return {
        "should_retry": bool(reasons),
        "risk_reasons": reasons,
        "explicit_answer_mentions": mentions,
        "tail_answer_letters": tail_letters,
        "answer_tag_letter": answer_tag_letter,
        "natural_completion": natural_completion,
    }


def score_quality(quality: Dict[str, Any]) -> Tuple[int, int]:
    """Rank an answer assessment by severe issues first, then total issues."""
    reasons = quality.get("risk_reasons") or []
    severe = sum(
        1
        for reason in reasons
        if (
            reason == "missing_final_answer"
            or reason == "multiple_tail_answer_mentions"
            or reason == "likely_truncated_output"
            or reason.startswith("non_natural_completion:")
            or "mismatch" in reason
        )
    )
    return severe, len(reasons)
