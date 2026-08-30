#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public model registry for the semantic Yes/No main experiment.

The script paths intentionally use a flat ``Models/<model>.py`` layout so the
release package does not require one source-code subfolder per model. Model
weight paths are configurable through ``EMBODIED_MODEL_ROOT`` or by editing the
``model_path`` fields below.
"""

from __future__ import annotations

import os
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = str(SCRIPT_ROOT)
VIDEO_DIR = str(SCRIPT_ROOT / "experiment_videos")
MODEL_ROOT = Path(os.environ.get("EMBODIED_MODEL_ROOT", SCRIPT_ROOT / "model_weights")).expanduser()


def model_script(name: str) -> str:
    return str(SCRIPT_ROOT / "Models" / f"{name}.py")


def model_weights(*parts: str) -> str:
    return str(MODEL_ROOT.joinpath(*parts))


MODELS = {
    "InternVL3.5": {
        "script_path": model_script("InternVL3.5"),
        "model_path": model_weights("InternVL3.5", "InternVL3_5-8B"),
        "supports_think_tag": True,
        "is_thinking_model": False,
        "modes": ["base", "think"],
        "mode_arg_name": "mode",
        "mode_arg_values": {"base": "base", "think": "thinking"},
    },
    "GLM-4.1V-base": {
        "script_path": model_script("GLM-4.1V-base"),
        "model_path": model_weights("GLM-4.1V-base", "GLM-4.1V-9B-Base"),
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "GLM-4.1V-thinking": {
        "script_path": model_script("GLM-4.1V-thinking"),
        "model_path": model_weights("GLM-4.1V-thinking", "GLM-4.1V-9B-Thinking"),
        "supports_think_tag": False,
        "is_thinking_model": True,
        "modes": ["think"],
    },
    "Qwen": {
        "script_path": model_script("Qwen"),
        "model_path": model_weights("Qwen", "Qwen3-VL-8B-Instruct"),
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "Qwen-Thinking": {
        "script_path": model_script("Qwen-Thinking"),
        "model_path": model_weights("Qwen-Thinking", "Qwen3-VL-8B-Thinking"),
        "supports_think_tag": False,
        "is_thinking_model": True,
        "modes": ["think"],
    },
    "RynnBrain-8B": {
        "script_path": model_script("RynnBrain-8B"),
        "model_path": model_weights("RynnBrain-8B", "RynnBrain-8B"),
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "RynnBrain-CoP": {
        "script_path": model_script("RynnBrain-CoP"),
        "model_path": model_weights("RynnBrain-CoP", "RynnBrain-CoP-8B"),
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "RoboBrain2.5": {
        "script_path": model_script("RoboBrain2.5"),
        "model_path": model_weights("RoboBrain2.5", "RoboBrain2.5-8B-NV"),
        "supports_think_tag": False,
        "is_thinking_model": False,
        "modes": ["base"],
    },
    "MiMo-Embodied": {
        "script_path": model_script("MiMo-Embodied"),
        "model_path": model_weights("MiMo-Embodied", "MiMo-Embodied-7B-hf"),
        "supports_think_tag": True,
        "is_thinking_model": False,
        "modes": ["base", "think"],
        "mode_arg_name": "mode",
        "mode_arg_values": {"base": "base", "think": "thinking"},
    },
}


def get_experiment_configs():
    """Generate all experiment configurations as (model_name, mode, config)."""
    configs = []
    for model_name, config in MODELS.items():
        for mode in config["modes"]:
            configs.append((model_name, mode, config))
    return configs


if __name__ == "__main__":
    configs = get_experiment_configs()
    print("Experiment configurations:")
    for model_name, mode, _ in configs:
        print(f"  - {model_name} ({mode})")
    print(f"\nTotal experiment settings: {len(configs)}")
