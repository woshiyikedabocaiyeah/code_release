"""
Canonical model naming for the manuscript — one definition, used by every
figure script and every derived table (RSA, PCA, probing).

The problem this solves
-----------------------
The analysis package identifies a model by a checkpoint string plus a mode flag
(`GLM-4.1V-9B-Base / base`, `InternVL3.5-8B / think`, `Qwen3-VL-8B-Instruct /
base`, ...). Three things are tangled together in those strings:

  * the model family and version        -> GLM-4.1V, Qwen3-VL, InternVL3.5, ...
  * the parameter count                 -> 9B, 8B, 7B
  * the inference regime                -> direct answer vs explicit reasoning

and the vendors are inconsistent about which of them lives in the checkpoint
name. GLM ships two checkpoints whose names carry the regime (`-Base`,
`-Thinking`); Qwen calls its direct-answer checkpoint `-Instruct`; InternVL3.5
and MiMo-Embodied expose both regimes from one checkpoint, so the regime is not
in the name at all.

The convention
--------------
A name is only a name. The regime is a separate, uniform tag.

    display = "<family>"                     single-regime models
    display = "<family> (Base|Thinking)"     dual-regime models

Parameter counts appear once, at first mention, via ``full_name``; they are
never repeated in figures or in running text. Models that expose only one
regime carry no tag at all, rather than a misleading "Base".

Usage
-----
    from model_naming import label, order_key, MODELS, canonical_order

    label("Qwen3-VL-8B-Instruct", "base")   -> "Qwen3-VL (Base)"
    label("RoboBrain2.5-8B-NV", "base")     -> "RoboBrain2.5"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariant:
    key: tuple[str, str]  # (model_name, model_mode) as they appear in the package
    family: str  # display family, no parameter count, no regime
    regime: str | None  # "Base" | "Thinking" | None for single-regime models
    full_name: str  # with parameter count, for first mention / Methods
    checkpoint: str  # the released artefact actually evaluated
    group: str  # "general-purpose" | "embodied"
    release_family: str  # the released model line, as counted in Methods

    @property
    def label(self) -> str:
        return f"{self.family} ({self.regime})" if self.regime else self.family

    @property
    def label_short(self) -> str:
        """Two-line form for cramped figure panels."""
        return f"{self.family}\n({self.regime})" if self.regime else self.family


# Order: general-purpose families first, then embodied, matching the Methods
# grouping; Base before Thinking within a family.
MODELS: tuple[ModelVariant, ...] = (
    ModelVariant(("GLM-4.1V-9B-Base", "base"), "GLM-4.1V", "Base",
                 "GLM-4.1V-9B", "GLM-4.1V-9B-Base", "general-purpose", "GLM-4.1V"),
    ModelVariant(("GLM-4.1V-9B-Thinking", "think"), "GLM-4.1V", "Thinking",
                 "GLM-4.1V-9B", "GLM-4.1V-9B-Thinking", "general-purpose", "GLM-4.1V"),
    ModelVariant(("Qwen3-VL-8B-Instruct", "base"), "Qwen3-VL", "Base",
                 "Qwen3-VL-8B", "Qwen3-VL-8B-Instruct", "general-purpose", "Qwen3-VL"),
    ModelVariant(("Qwen3-VL-8B-Thinking", "think"), "Qwen3-VL", "Thinking",
                 "Qwen3-VL-8B", "Qwen3-VL-8B-Thinking", "general-purpose", "Qwen3-VL"),
    ModelVariant(("InternVL3.5-8B", "base"), "InternVL3.5", "Base",
                 "InternVL3.5-8B", "InternVL3.5-8B", "general-purpose", "InternVL3.5"),
    ModelVariant(("InternVL3.5-8B", "think"), "InternVL3.5", "Thinking",
                 "InternVL3.5-8B", "InternVL3.5-8B", "general-purpose", "InternVL3.5"),
    ModelVariant(("MiMo-Embodied-7B", "base"), "MiMo-Embodied", "Base",
                 "MiMo-Embodied-7B", "MiMo-Embodied-7B", "embodied", "MiMo-Embodied"),
    ModelVariant(("MiMo-Embodied-7B", "think"), "MiMo-Embodied", "Thinking",
                 "MiMo-Embodied-7B", "MiMo-Embodied-7B", "embodied", "MiMo-Embodied"),
    ModelVariant(("RoboBrain2.5-8B-NV", "base"), "RoboBrain2.5", None,
                 "RoboBrain2.5-8B", "RoboBrain2.5-8B-NV", "embodied", "RoboBrain2.5"),
    ModelVariant(("RynnBrain-8B", "base"), "RynnBrain", None,
                 "RynnBrain-8B", "RynnBrain-8B", "embodied", "RynnBrain"),
    ModelVariant(("RynnBrain-CoP-8B", "base"), "RynnBrain-CoP", None,
                 "RynnBrain-CoP-8B", "RynnBrain-CoP-8B", "embodied", "RynnBrain"),
)

_BY_KEY = {m.key: m for m in MODELS}
canonical_order: list[str] = [m.label for m in MODELS]

# Families that expose two regimes from a single checkpoint. Their vision
# encoder and projection representations are identical across regimes; any
# analysis of the visual stages must not treat them as independent.
SHARED_CHECKPOINT_FAMILIES: tuple[str, ...] = ("InternVL3.5", "MiMo-Embodied")

# Display names for the architectural stages. These match the running text
# ("the vision encoder, the vision-language projector, and the language model").
STAGE_LABEL = {
    "vision_encoder_last": "Vision encoder",
    "vision_projection": "Vision–language projector",
    "language_model_last": "Language model",
}
# Two-line form for figure axes, where the projector label is too wide.
STAGE_LABEL_WRAPPED = {
    "vision_encoder_last": "Vision\nencoder",
    "vision_projection": "Vision–language\nprojector",
    "language_model_last": "Language\nmodel",
}
STAGE_ORDER = list(STAGE_LABEL)

# Human behavioural measures entering RSA. RT_onset is the measure the RSA and
# PCA analyses were built on; RT_critical is a second measure requested for the
# revision and requires human data that is not in the analysis package.
HUMAN_MEASURE_LABEL = {
    "rt": "RT$_{onset}$",
    "rt_critical": "RT$_{critical}$",
    "corr": "Accuracy",
}
HUMAN_MEASURE_LABEL_TEXT = {
    "rt": "RT~onset~",
    "rt_critical": "RT~critical~",
    "corr": "accuracy",
}

# Task codes in the package vs the embodiment-spectrum names in the manuscript.
TASK_LABEL = {
    "Category": "Concept Verification",
    "VoE": "Plausibility Assessment",
    "SM": "Affordance Recognition",
}
TASK_ORDER = list(TASK_LABEL)

# Human behavioural metric codes.
METRIC_LABEL = {"rt": "RT", "corr": "ACC"}


def variant(model_name: str, model_mode: str) -> ModelVariant:
    try:
        return _BY_KEY[(model_name, model_mode)]
    except KeyError as exc:  # a new checkpoint must be added here deliberately
        raise KeyError(
            f"unknown model variant {(model_name, model_mode)!r}; "
            "add it to common/model_naming.py"
        ) from exc


def label(model_name: str, model_mode: str) -> str:
    return variant(model_name, model_mode).label


def label_short(model_name: str, model_mode: str) -> str:
    return variant(model_name, model_mode).label_short


def order_key(model_name: str, model_mode: str) -> int:
    return MODELS.index(variant(model_name, model_mode))


def family_of(model_name: str, model_mode: str) -> str:
    return variant(model_name, model_mode).family


if __name__ == "__main__":  # quick self-check / reference table
    print(f"{'package key':38s} {'display':26s} {'full name':18s} group")
    for m in MODELS:
        print(f"{m.key[0] + ' / ' + m.key[1]:38s} {m.label:26s} "
              f"{m.full_name:18s} {m.group}")
    assert len(MODELS) == 11
    assert len({m.label for m in MODELS}) == 11
    assert len({m.family for m in MODELS}) == 7
    assert len({m.release_family for m in MODELS}) == 6
    print(f"\n{len(MODELS)} variants, {len({m.family for m in MODELS})} display families, "
          f"{len({m.release_family for m in MODELS})} release families")
