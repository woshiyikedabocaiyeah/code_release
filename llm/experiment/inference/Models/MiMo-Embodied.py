#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiMo-Embodied-7B inference script for experiment pipeline.

Supports both:
- thinking mode (default prompt)
- no_think mode (append `/no_think` to prompt text)

Embeddings exported:
- vision_encoder_last
- vision_projection
- language_model_last
"""

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
DEFAULT_MAX_NEW_TOKENS_BASE = int(os.environ.get("EMBODIED_MAX_NEW_TOKENS_BASE", "4096"))
DEFAULT_MAX_NEW_TOKENS_THINK = int(os.environ.get("EMBODIED_MAX_NEW_TOKENS_THINK", "12288"))
MIMO_MAX_NEW_TOKENS_BASE = int(os.environ.get("EMBODIED_MIMO_MAX_NEW_TOKENS_BASE", "1024"))
MIMO_MAX_NEW_TOKENS_THINK = int(os.environ.get("EMBODIED_MIMO_MAX_NEW_TOKENS_THINK", "3072"))
MIMO_RETRY_MAX_NEW_TOKENS = int(os.environ.get("EMBODIED_MIMO_RETRY_MAX_NEW_TOKENS", "3072"))
MIMO_REPETITION_PENALTY = float(os.environ.get("EMBODIED_MIMO_RESPONSE_REPETITION_PENALTY", "1.03"))
MIMO_VIDEO_FPS = float(os.environ.get("EMBODIED_MIMO_VIDEO_FPS", "1.0"))
MIMO_VIDEO_MAX_PIXELS = int(os.environ.get("EMBODIED_MIMO_VIDEO_MAX_PIXELS", str(512 * 28 * 28)))
MIMO_VIDEO_TOTAL_PIXELS = int(os.environ.get("EMBODIED_MIMO_VIDEO_TOTAL_PIXELS", str(4096 * 28 * 28)))
MIMO_GPU_MAX_MEMORY_GIB = int(os.environ.get("EMBODIED_MIMO_GPU_MAX_MEMORY_GIB", "34"))
MIMO_CPU_MAX_MEMORY_GIB = int(os.environ.get("EMBODIED_MIMO_CPU_MAX_MEMORY_GIB", "200"))
MIMO_DTYPE = os.environ.get("EMBODIED_MIMO_DTYPE", "auto").lower()

MODEL_NAME = "MiMo-Embodied"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ROOT = str(PROJECT_ROOT / "embeddings")
MIMO_PROCESSOR_FALLBACK = str(
    PROJECT_ROOT / "model_weights" / "MiMo-Embodied" / "Qwen2.5-VL-7B-Instruct-processor"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from answer_parser import (
    assess_answer_quality,
    build_answer_payload,
    clean_response_text,
    score_quality,
)  # noqa: E402

SCRIPT_VARIANT = "semantic_yesno_v2"
SEMANTIC_YESNO_OUTPUT = os.environ.get("EMBODIED_OUTPUT_FORMAT", "").strip().lower() in {
    "semantic_yesno",
    "yesno",
    "yes_no",
}
if SEMANTIC_YESNO_OUTPUT:
    MIMO_SYSTEM_PROMPT = (
        "You are a concise visual reasoner. "
        "Always answer using exactly this structure:\n"
        "<think>\n"
        "1 to 6 short sentences grounded in the video.\n"
        "</think>\n"
        "<answer>yes_or_no</answer>\n"
        "Replace yes_or_no with exactly yes or no. "
        "Do not use option letters. "
        "Do not omit the closing </answer> tag. "
        "Do not output any text after </answer>."
    )
    RESPONSE_GUIDANCE = (
        "\n\nResponse requirements:\n"
        "- Start immediately with <think> on the first line.\n"
        "- Keep the reasoning concise and focused on the video.\n"
        "- Do not restate the full prompt.\n"
        "- End with exactly one final answer enclosed in <answer> and </answer>.\n"
        "- Put only yes or no inside the <answer> tag.\n"
        "- Do not output option letters such as A or B.\n"
        "- Do not add any preamble before <think>.\n"
    )
    RETRY_GUIDANCE = (
        "\n\nPlease answer again from scratch."
        "\nKeep the reasoning shorter than before and make sure the response ends cleanly."
        "\nUse no more than 6 short reasoning sentences."
        "\nThe final line must be exactly <answer>yes</answer> or <answer>no</answer>."
    )
else:
    MIMO_SYSTEM_PROMPT = (
        "You are a concise visual reasoner. "
        "Always answer using exactly this structure:\n"
        "<think>\n"
        "1 to 6 short sentences grounded in the video.\n"
        "</think>\n"
        "<answer>X</answer>\n"
        "Replace X with exactly one option letter. "
        "Do not omit the closing </answer> tag. "
        "Do not output any text after </answer>."
    )
    RESPONSE_GUIDANCE = (
        "\n\nResponse requirements:\n"
        "- Start immediately with <think> on the first line.\n"
        "- Keep the reasoning concise and focused on the video.\n"
        "- Do not restate the full prompt or all answer options.\n"
        "- End with exactly one final answer enclosed in <answer> and </answer>.\n"
        "- Put only the chosen option letter inside the <answer> tag.\n"
        "- Do not add any preamble before <think>.\n"
    )
    RETRY_GUIDANCE = (
        "\n\nPlease answer again from scratch."
        "\nKeep the reasoning shorter than before and make sure the response ends cleanly."
        "\nUse no more than 6 short reasoning sentences."
        "\nThe final line must be exactly <answer>X</answer>."
    )
MIMO_LONG_THRESHOLD = int(os.environ.get("EMBODIED_MIMO_LONG_THRESHOLD", "4500"))


def extract_thinking_and_answer(text: str):
    if not text:
        return None, ""

    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        answer_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return thinking_content, answer_text

    end_tag = text.find("</think>")
    if end_tag != -1:
        thinking_content = text[:end_tag].replace("<think>", "").strip()
        answer_text = text[end_tag + len("</think>"):].strip()
        return thinking_content or None, answer_text

    return None, text.strip()


def normalize_first_response(text: str) -> str:
    if not text:
        return ""

    cleaned = clean_response_text(re.sub(r"<\|[^>]+\|>", "", text)).strip()
    cleaned = re.sub(r"^\s*assistant\s*", "", cleaned, count=1, flags=re.IGNORECASE)
    if not cleaned:
        return ""

    if "<think>" in cleaned:
        cleaned = cleaned[cleaned.find("<think>"):].strip()

    full_answer_match = re.search(
        r"(<think>.*?</think>\s*<answer>\s*([ABCD])\s*</answer>)",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if full_answer_match:
        return full_answer_match.group(1).strip()

    partial_answer_match = re.search(
        r"(<think>.*?</think>\s*<answer>\s*([ABCD]))(?=\s*(?:assistant\b|user\b|system\b|<think>|$))",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if partial_answer_match:
        prefix = partial_answer_match.group(1).strip()
        letter = partial_answer_match.group(2).upper()
        return f"{prefix}</answer>"

    if "<answer>" in cleaned:
        before_second_think = cleaned.split("\nassistant", 1)[0].strip()
        if before_second_think:
            return before_second_think

    return cleaned


class MiMoEmbodiedInference:
    """MiMo-Embodied-7B inference with embedding extraction."""

    def __init__(self, model_path, output_dir, mode="base"):
        self.model_path = os.path.expanduser(model_path)
        self.output_dir = output_dir
        self.mode = mode
        self.model_name = MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor_source = None

        os.makedirs(self.output_dir, exist_ok=True)

        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "auto": "auto",
        }
        model_dtype = dtype_map.get(MIMO_DTYPE, "auto")
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if model_dtype == "auto":
            load_kwargs["torch_dtype"] = "auto"
        else:
            load_kwargs["torch_dtype"] = model_dtype
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            max_memory = {i: f"{MIMO_GPU_MAX_MEMORY_GIB}GiB" for i in range(gpu_count)}
            max_memory["cpu"] = f"{MIMO_CPU_MAX_MEMORY_GIB}GiB"
            load_kwargs["max_memory"] = max_memory

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_kwargs,
        ).eval()
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self.processor_source = self.model_path
        except Exception:
            self.processor = AutoProcessor.from_pretrained(
                MIMO_PROCESSOR_FALLBACK,
                trust_remote_code=True,
            )
            try:
                model_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    use_fast=False,
                )
                self.processor.tokenizer = model_tokenizer
                self.processor_source = f"{MIMO_PROCESSOR_FALLBACK}+tokenizer_from_model"
            except Exception:
                self.processor_source = MIMO_PROCESSOR_FALLBACK

        sys.stderr.close()
        sys.stderr = original_stderr

        self.hooks = []
        self.captured_embeddings = {}
        self._register_vision_hooks()
        self.stop_strings = ["</answer>", "<|im_end|>", "\nassistant", "\nassistant\n"]
        self.stop_suffix_token_ids = [
            token_ids
            for token_ids in [
                self._build_suffix_token_ids("</answer>"),
                self._build_suffix_token_ids("\nassistant"),
                self._build_suffix_token_ids("\nassistant\n"),
            ]
            if token_ids
        ]

    def _register_vision_hooks(self):
        def make_hook(name):
            def hook(_, __, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if getattr(tensor, "is_meta", False):
                    return
                self.captured_embeddings[name] = tensor.detach().cpu().clone()

            return hook

        try:
            if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
                visual = self.model.model.visual
                if hasattr(visual, "blocks") and len(visual.blocks) > 0:
                    self.hooks.append(
                        visual.blocks[-1].register_forward_hook(
                            make_hook("vision_encoder_last")
                        )
                    )
                if hasattr(visual, "merger"):
                    self.hooks.append(
                        visual.merger.register_forward_hook(
                            make_hook("vision_projection")
                        )
                    )
            if (
                hasattr(self.model, "model")
                and hasattr(self.model.model, "language_model")
                and hasattr(self.model.model.language_model, "layers")
            ):
                layers = self.model.model.language_model.layers
                if len(layers) > 0:
                    self.hooks.append(
                        layers[-1].register_forward_hook(
                            make_hook("language_model_last")
                        )
                    )
        except Exception:
            pass

    def _extract_video_frames(self, video_path, num_frames=8):
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return frames

            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
        except Exception:
            pass
        return frames

    def _pool_embedding(self, emb):
        if len(emb.shape) == 3:
            return emb.mean(dim=1)
        if len(emb.shape) == 2:
            return emb.mean(dim=0, keepdim=True)
        return emb

    def _save_embeddings(self, video_id, prompt_type):
        saved_files = {}
        embedding_stats = {}

        for name in ["vision_encoder_last", "vision_projection", "language_model_last"]:
            if name not in self.captured_embeddings:
                continue
            emb = self.captured_embeddings[name]
            emb_to_save = self._pool_embedding(emb)
            filename = f"{video_id}_{prompt_type}_{name}.npy"
            filepath = os.path.join(self.output_dir, filename)
            np.save(filepath, emb_to_save.float().numpy())
            saved_files[name] = filepath
            embedding_stats[name] = {
                "original_shape": list(emb.shape),
                "saved_shape": list(emb_to_save.shape),
                "dtype": str(emb.dtype),
                "mean": float(emb_to_save.float().mean()),
                "std": float(emb_to_save.float().std()),
                "min": float(emb_to_save.float().min()),
                "max": float(emb_to_save.float().max()),
            }

        return saved_files, embedding_stats

    def _save_metadata(
        self,
        video_id,
        prompt_type,
        prompt_text,
        effective_prompt_text,
        original_output,
        retry_output,
        response,
        final_answer,
        answer_parse_method,
        answer_parse_status,
        thinking_content,
        inference_time,
        generation_max_new_tokens,
        video_runtime_config,
        used_oom_fallback,
        embedding_files,
        embedding_stats,
        video_path,
    ):
        metadata = {
            "model": self.model_name,
            "mode": self.mode,
            "video_id": video_id,
            "video_path": video_path,
            "prompt_type": prompt_type,
            "prompt_text": prompt_text,
            "effective_prompt_text": effective_prompt_text,
            "original_output": original_output,
            "retry_output": retry_output,
            "response": response,
            "final_answer": final_answer,
            "answer_parse_method": answer_parse_method,
            "answer_parse_status": answer_parse_status,
            "thinking_content": thinking_content,
            "native_thinking_switch_supported": True,
            "thinking_switch_method": "prompt_suffix_/no_think",
            "processor_source": self.processor_source,
            "processor_pipeline": "official_qwen_vl_utils" if process_vision_info is not None else "chat_template_tokenize",
            "model_dtype": MIMO_DTYPE,
            "max_memory_gpu_gib": MIMO_GPU_MAX_MEMORY_GIB,
            "max_memory_cpu_gib": MIMO_CPU_MAX_MEMORY_GIB,
            "inference_time_seconds": inference_time,
            "processing_duration_seconds": inference_time,
            "generation_max_new_tokens": generation_max_new_tokens,
            "generation_do_sample": False,
            "video_runtime_config": video_runtime_config,
            "used_oom_fallback": used_oom_fallback,
            "timestamp": datetime.now().isoformat(),
            "embedding_files": embedding_files,
            "embedding_stats": embedding_stats,
        }

        meta_filename = f"{video_id}_{prompt_type}_metadata.json"
        meta_path = os.path.join(self.output_dir, meta_filename)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return meta_path

    def _build_effective_prompt(self, prompt_text: str) -> str:
        effective_prompt = prompt_text
        if self.mode == "base":
            if "/no_think" not in prompt_text:
                effective_prompt = f"{prompt_text} /no_think"
        return effective_prompt + RESPONSE_GUIDANCE

    def _build_retry_prompt(self, prompt_text: str) -> str:
        return self._build_effective_prompt(prompt_text) + RETRY_GUIDANCE

    def _get_generation_max_new_tokens(self):
        default_tokens = (
            DEFAULT_MAX_NEW_TOKENS_THINK if self.mode == "thinking" else DEFAULT_MAX_NEW_TOKENS_BASE
        )
        mode_cap = (
            MIMO_MAX_NEW_TOKENS_THINK if self.mode == "thinking" else MIMO_MAX_NEW_TOKENS_BASE
        )
        return min(default_tokens, mode_cap)

    def _parse_candidate(self, raw_output: str):
        raw_text = clean_response_text(re.sub(r"<\|[^>]+\|>", "", raw_output)).strip()
        cleaned_output = normalize_first_response(raw_output)
        thinking_content, response_candidate = extract_thinking_and_answer(cleaned_output)
        answer_payload = build_answer_payload(
            response_candidate,
            cleaned_output,
            raw_text,
        )
        quality = assess_answer_quality(
            raw_text,
            answer_payload["final_answer"],
            answer_payload["answer_parse_method"],
            long_threshold=MIMO_LONG_THRESHOLD,
            raw_generated_output=raw_text,
            normalized_output=cleaned_output,
            require_natural_completion=True,
        )
        return {
            "raw_generated_output": raw_text,
            "original_output": cleaned_output,
            "thinking_content": thinking_content,
            "answer_payload": answer_payload,
            "quality": quality,
        }

    def _build_video_content(self, video_path: str, reduced: bool = False):
        fps = MIMO_VIDEO_FPS
        max_pixels = MIMO_VIDEO_MAX_PIXELS
        total_pixels = MIMO_VIDEO_TOTAL_PIXELS
        if reduced:
            fps = max(0.5, fps / 2.0)
            max_pixels = max(128 * 28 * 28, max_pixels // 2)
            total_pixels = max(1024 * 28 * 28, total_pixels // 2)

        video_content = {"type": "video", "video": video_path}
        if fps > 0:
            video_content["fps"] = fps
        if max_pixels > 0:
            video_content["max_pixels"] = int(max_pixels)
        if total_pixels > 0:
            video_content["total_pixels"] = int(total_pixels)
        return video_content

    def _build_model_inputs(self, messages):
        if process_vision_info is not None:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(text, list):
                text_inputs = text
            else:
                text_inputs = [text]
            image_inputs, video_inputs = process_vision_info(messages)
            model_inputs = self.processor(
                text=text_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            return model_inputs

        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return model_inputs

    def _build_suffix_token_ids(self, text: str):
        token_ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
        return token_ids if token_ids else None

    def _build_stopping_criteria(self):
        suffix_ids_list = self.stop_suffix_token_ids
        if not suffix_ids_list:
            return None

        class _MultiSuffixStoppingCriteria(StoppingCriteria):
            def __init__(self, suffixes):
                super().__init__()
                self.suffixes = suffixes
                self.max_len = max(len(suffix) for suffix in suffixes)

            def __call__(self, input_ids, scores, **kwargs):
                if input_ids.shape[1] < 1:
                    return False
                sequence = input_ids[0].tolist()
                for suffix in self.suffixes:
                    suffix_len = len(suffix)
                    if len(sequence) >= suffix_len and sequence[-suffix_len:] == suffix:
                        return True
                return False

        return StoppingCriteriaList([_MultiSuffixStoppingCriteria(suffix_ids_list)])

    def _build_messages(self, video_content, prompt_text: str):
        return [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

    def _is_oom_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda oom" in text

    def _run_single_pass(self, video_path: str, prompt_text: str, max_new_tokens: int):
        attempts = [False, True]
        used_oom_fallback = False
        last_exc = None

        for use_reduced_video in attempts:
            video_content = self._build_video_content(video_path, reduced=use_reduced_video)
            video_runtime_config = {
                "fps": video_content.get("fps"),
                "max_pixels": video_content.get("max_pixels"),
                "total_pixels": video_content.get("total_pixels"),
            }
            messages = self._build_messages(video_content, prompt_text)
            try:
                model_inputs = self._build_model_inputs(messages)
                model_inputs = model_inputs.to(self.model.device)

                with torch.no_grad():
                    inference_start = datetime.now()
                    generate_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                        "repetition_penalty": MIMO_REPETITION_PENALTY,
                    }
                    if self.stop_strings:
                        generate_kwargs["stop_strings"] = self.stop_strings
                        generate_kwargs["tokenizer"] = self.processor.tokenizer
                    stopping_criteria = self._build_stopping_criteria()
                    if stopping_criteria is not None:
                        generate_kwargs["stopping_criteria"] = stopping_criteria
                        generate_kwargs["eos_token_id"] = None
                    pad_token_id = self.processor.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = self.processor.tokenizer.eos_token_id
                    if pad_token_id is not None:
                        generate_kwargs["pad_token_id"] = pad_token_id

                    generated_ids = self.model.generate(
                        **model_inputs,
                        **generate_kwargs,
                    )
                    inference_end = datetime.now()
                    inference_time = (inference_end - inference_start).total_seconds()

                generated_ids_trimmed = [
                    out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
                ]
                raw_output = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )[0]
                candidate = self._parse_candidate(raw_output)
                candidate["inference_time"] = inference_time
                candidate["video_runtime_config"] = video_runtime_config
                candidate["used_oom_fallback"] = used_oom_fallback or use_reduced_video
                return candidate
            except RuntimeError as exc:
                last_exc = exc
                if self._is_oom_error(exc) and not use_reduced_video:
                    used_oom_fallback = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to generate response for MiMo-Embodied")

    def _select_candidate(self, initial_candidate, retry_candidate):
        if retry_candidate is None:
            return initial_candidate, "initial"
        initial_answer = initial_candidate["answer_payload"]["final_answer"]
        retry_answer = retry_candidate["answer_payload"]["final_answer"]
        if initial_answer is None and retry_answer is not None:
            return retry_candidate, "retry"
        if retry_answer is None:
            return initial_candidate, "initial"
        if score_quality(retry_candidate["quality"]) < score_quality(initial_candidate["quality"]):
            return retry_candidate, "retry"
        return initial_candidate, "initial"

    def run_inference(self, video_path, prompt_text, prompt_type, video_id):
        result = {
            "success": False,
            "error": None,
            "model": self.model_name,
            "mode": self.mode,
            "video_id": video_id,
            "prompt_type": prompt_type,
            "raw_prompt": prompt_text,
            "original_output": None,
            "response": None,
            "final_answer": None,
            "answer_parse_method": None,
            "answer_parse_status": None,
            "thinking_content": None,
            "inference_time_seconds": None,
            "embeddings_saved": {},
            "metadata_path": None,
        }

        try:
            self.captured_embeddings = {}
            effective_prompt = self._build_effective_prompt(prompt_text)
            retry_output = None
            raw_retry_output = None
            retry_quality = None
            max_new_tokens = self._get_generation_max_new_tokens()

            initial_candidate = self._run_single_pass(video_path, effective_prompt, max_new_tokens)
            selected_candidate = initial_candidate
            selected_pass = "initial"
            visible_initial_quality = assess_answer_quality(
                initial_candidate["original_output"],
                initial_candidate["answer_payload"]["final_answer"],
                initial_candidate["answer_payload"]["answer_parse_method"],
                long_threshold=MIMO_LONG_THRESHOLD,
                raw_generated_output=initial_candidate["original_output"],
                normalized_output=initial_candidate["original_output"],
                require_natural_completion=True,
            )
            should_retry = initial_candidate["quality"]["should_retry"] and visible_initial_quality["should_retry"]
            if should_retry:
                retry_candidate = self._run_single_pass(
                    video_path,
                    self._build_retry_prompt(prompt_text),
                    min(DEFAULT_MAX_NEW_TOKENS_THINK, MIMO_RETRY_MAX_NEW_TOKENS),
                )
                raw_retry_output = retry_candidate["raw_generated_output"]
                retry_output = retry_candidate["original_output"]
                retry_quality = retry_candidate["quality"]
                selected_candidate, selected_pass = self._select_candidate(
                    initial_candidate,
                    retry_candidate,
                )

            embedding_files, embedding_stats = self._save_embeddings(video_id, prompt_type)
            meta_path = self._save_metadata(
                video_id=video_id,
                prompt_type=prompt_type,
                prompt_text=prompt_text,
                effective_prompt_text=effective_prompt,
                original_output=selected_candidate["original_output"],
                retry_output=retry_output,
                response=selected_candidate["answer_payload"]["response"],
                final_answer=selected_candidate["answer_payload"]["final_answer"],
                answer_parse_method=selected_candidate["answer_payload"]["answer_parse_method"],
                answer_parse_status=selected_candidate["answer_payload"]["answer_parse_status"],
                thinking_content=selected_candidate["thinking_content"],
                inference_time=selected_candidate["inference_time"],
                generation_max_new_tokens=max_new_tokens,
                video_runtime_config=selected_candidate["video_runtime_config"],
                used_oom_fallback=selected_candidate["used_oom_fallback"],
                embedding_files=embedding_files,
                embedding_stats=embedding_stats,
                video_path=video_path,
            )

            result["success"] = True
            result["original_output"] = selected_candidate["original_output"]
            result["response"] = selected_candidate["answer_payload"]["response"]
            result["final_answer"] = selected_candidate["answer_payload"]["final_answer"]
            result["answer_parse_method"] = selected_candidate["answer_payload"]["answer_parse_method"]
            result["answer_parse_status"] = selected_candidate["answer_payload"]["answer_parse_status"]
            result["thinking_content"] = selected_candidate["thinking_content"]
            result["inference_time_seconds"] = selected_candidate["inference_time"]
            result["embeddings_saved"] = embedding_files
            result["metadata_path"] = meta_path
            return result

        except Exception as exc:
            result["error"] = str(exc)
            return result

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description="MiMo-Embodied-7B Inference for Experiment")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=str(PROJECT_ROOT / "model_weights" / "MiMo-Embodied" / "MiMo-Embodied-7B-hf"),
        help="Path to model directory",
    )
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to video file")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Prompt text")
    parser.add_argument(
        "--prompt-type",
        "-t",
        type=str,
        required=True,
        choices=["simple", "detailed", "embodied_simple", "embodied_detailed"],
        help="Prompt type",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="base",
        choices=["base", "thinking"],
        help="Run mode: base(no_think) or thinking",
    )
    parser.add_argument("--video-id", type=str, required=True, help="Video ID for naming output files")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=f"{EMBEDDING_ROOT}/{MODEL_NAME}/base",
        help="Output directory for embeddings",
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    try:
        runner = MiMoEmbodiedInference(args.model, args.output, mode=args.mode)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Model initialization failed: {str(exc)}",
                    "model": MODEL_NAME,
                    "mode": args.mode,
                    "video_id": args.video_id,
                    "prompt_type": args.prompt_type,
                    "raw_prompt": args.prompt,
                    "original_output": None,
                    "response": None,
                    "final_answer": None,
                    "answer_parse_method": None,
                    "answer_parse_status": None,
                    "thinking_content": None,
                    "inference_time_seconds": None,
                    "embeddings_saved": {},
                    "metadata_path": None,
                },
                ensure_ascii=False,
            )
        )
        return

    result = runner.run_inference(
        video_path=args.video,
        prompt_text=args.prompt,
        prompt_type=args.prompt_type,
        video_id=args.video_id,
    )
    print(json.dumps(result, ensure_ascii=False))
    runner.cleanup()


if __name__ == "__main__":
    main()
