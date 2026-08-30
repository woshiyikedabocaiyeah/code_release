#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt templates for the semantic Yes/No main experiment."""

COMMON_INSTRUCTION_PROMPT = """INSTRUCTIONS: PLEASE READ CAREFULLY
In this experiment there are a number of trials.
In each trial, you will see a short video featuring an event related to a critical object in an environment.

Before we start the experimental trials, let’s go through some physical concepts you will need to understand.

In this experiment, we will use three basic physical concepts:

1/3 Permanence
Objects continue to exist even when they are hidden or out of view.
For example, If a ball rolls behind a wall, it still exists behind the wall.

In this experiment, we will use three basic physical concepts:

2/3 Continuity
Objects move along continuous paths through space and time.
For example, they do not suddenly “teleport” from one place to a distant place without traveling through the space in between.

In this experiment, we will use three basic physical concepts:

3/3 Immutability
An object’s basic physical properties(such as its shape, size, and color) do not suddenly change without a cause.
For example, a ball does not instantly shrink, grow, or turn into a different color for no reason."""

SYSTEM_PROMPT = (
    COMMON_INSTRUCTION_PROMPT
    + "\n\n"
    + "Please answer in English."
)

SEMANTIC_QUESTION_TEMPLATE = """Based on the video content, answer the question directly.

Question: {question}

IMPORTANT: Your final answer must be exactly one word: yes or no.
Do not use option letters such as A or B.
Do not include the original answer choices in your final answer."""

EMBODIED_PROMPT = """You are an embodied intelligence possessing a simulated human-like physical form. You are not a disembodied text generator; you exist within a specific physical space.
Your Body Schema:
- Dimensions: You occupy a volume roughly 1.75 m tall.
- Limbs: You have two arms, two legs, a torso, and a head. You must track the position of these limbs at all times.
- Sensory Input: You possess simulated proprioception. Before every action or response, you must calculate the position, tension, and comfort of your imaginary muscles.
Please utilize this simulated human-like physical form to facilitate your reasoning when you answer the following question."""


def format_base_question_prompt(question_data):
    """Format the semantic Yes/No question section."""
    return SEMANTIC_QUESTION_TEMPLATE.format(question=str(question_data["question"]))


def _format_detail_section(detail_info):
    detail_section = "No additional detail information is available."
    if detail_info:
        detail_section = str(detail_info).strip()
    return f"Additional video detail information:\n{detail_section}"


def _compose_prompt(question_data, detail_info=None, embodied=False):
    sections = [SYSTEM_PROMPT]
    if embodied:
        sections.append(EMBODIED_PROMPT)
    if detail_info is not None:
        sections.append(_format_detail_section(detail_info))
    sections.append(format_base_question_prompt(question_data))
    return "\n\n".join(sections)


def build_simple_prompt(question_data):
    """simple = instructions + semantic Yes/No question."""
    return _compose_prompt(question_data)


def build_detail_prompt(question_data, detail_info):
    """detail = instructions + video detail + semantic Yes/No question."""
    return _compose_prompt(question_data, detail_info=detail_info)


def build_embodied_simple_prompt(question_data):
    """embodied_simple = instructions + embodied prompt + semantic Yes/No question."""
    return _compose_prompt(question_data, embodied=True)


def build_embodied_detail_prompt(question_data, detail_info):
    """embodied_detail = instructions + embodied prompt + video detail + semantic Yes/No question."""
    return _compose_prompt(question_data, detail_info=detail_info, embodied=True)


def format_prompt(question_data):
    """Alias for the default simple semantic Yes/No prompt."""
    return build_simple_prompt(question_data)
