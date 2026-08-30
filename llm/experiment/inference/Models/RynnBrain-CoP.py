#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RynnBrain-CoP-8B inference script for experiment pipeline.

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
from transformers import AutoModelForImageTextToText, AutoProcessor
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
DEFAULT_MAX_NEW_TOKENS_BASE = int(os.environ.get("EMBODIED_MAX_NEW_TOKENS_BASE", "4096"))

MODEL_NAME = "RynnBrain-CoP"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ROOT = str(PROJECT_ROOT / "embeddings")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from answer_parser import build_answer_payload, clean_response_text  # noqa: E402


class RynnBrainCoPInference:
    """RynnBrain-CoP-8B inference with three-layer embedding extraction."""

    def __init__(self, model_path, output_dir):
        self.model_path = os.path.expanduser(model_path)
        self.output_dir = output_dir
        self.mode = "base"
        self.model_name = MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.output_dir, exist_ok=True)

        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                dtype="auto",
                trust_remote_code=True,
                device_map="auto",
            ).eval()
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        sys.stderr.close()
        sys.stderr = original_stderr

        self.hooks = []
        self.captured_embeddings = {}
        self._register_vision_hooks()

    def _extract_thinking_content(self, response):
        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, response, flags=re.DOTALL)
        if match:
            thinking_content = match.group(1).strip()
            answer = re.sub(pattern, "", response, flags=re.DOTALL).strip()
            return thinking_content, answer
        return None, response

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

    def _extract_language_embedding(self, model_inputs):
        with torch.no_grad():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden_states = outputs.hidden_states[-1]
            language_embedding = last_hidden_states.mean(dim=1)
            return language_embedding.cpu().clone()

    def _build_model_inputs(self, messages):
        if process_vision_info is not None:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            text_inputs = text if isinstance(text, list) else [text]
            image_inputs, video_inputs = process_vision_info(messages)
            return self.processor(
                text=text_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _pool_embedding(self, emb):
        if len(emb.shape) == 3:
            return emb.mean(dim=1)
        if len(emb.shape) == 2:
            return emb.mean(dim=0, keepdim=True)
        return emb

    def _save_embeddings(self, video_id, prompt_type, language_embedding):
        saved_files = {}
        embedding_stats = {}

        for name in ["vision_encoder_last", "vision_projection"]:
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

        lm_filename = f"{video_id}_{prompt_type}_language_model_last.npy"
        lm_filepath = os.path.join(self.output_dir, lm_filename)
        np.save(lm_filepath, language_embedding.float().numpy())
        saved_files["language_model_last"] = lm_filepath
        embedding_stats["language_model_last"] = {
            "saved_shape": list(language_embedding.shape),
            "dtype": str(language_embedding.dtype),
            "mean": float(language_embedding.float().mean()),
            "std": float(language_embedding.float().std()),
            "min": float(language_embedding.float().min()),
            "max": float(language_embedding.float().max()),
        }

        return saved_files, embedding_stats

    def _save_metadata(
        self,
        video_id,
        prompt_type,
        prompt_text,
        original_output,
        response,
        final_answer,
        answer_parse_method,
        answer_parse_status,
        thinking_content,
        inference_time,
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
            "original_output": original_output,
            "response": response,
            "final_answer": final_answer,
            "answer_parse_method": answer_parse_method,
            "answer_parse_status": answer_parse_status,
            "thinking_content": thinking_content,
            "native_thinking_switch_supported": False,
            "processor_pipeline": "official_qwen_vl_utils" if process_vision_info is not None else "chat_template_tokenize",
            "inference_time_seconds": inference_time,
            "processing_duration_seconds": inference_time,
            "timestamp": datetime.now().isoformat(),
            "embedding_files": embedding_files,
            "embedding_stats": embedding_stats,
        }

        meta_filename = f"{video_id}_{prompt_type}_metadata.json"
        meta_path = os.path.join(self.output_dir, meta_filename)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return meta_path

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

            messages = [
                {
                    "role": "user",
                    "content": [
                        # Use native video input so processor applies model-default sampling.
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            model_inputs = self._build_model_inputs(messages)
            model_inputs = model_inputs.to(self.model.device)

            with torch.no_grad():
                language_embedding = self._extract_language_embedding(model_inputs)

            with torch.no_grad():
                inference_start = datetime.now()
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS_BASE,
                    do_sample=False,
                )
                inference_end = datetime.now()
                inference_time = (inference_end - inference_start).total_seconds()

            generated_ids_trimmed = [
                out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
            ]
            original_output = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            cleaned_output = clean_response_text(re.sub(r"<\|[^>]+\|>", "", original_output))
            thinking_content, response_candidate = self._extract_thinking_content(cleaned_output)
            answer_payload = build_answer_payload(
                response_candidate,
                cleaned_output,
                clean_response_text(original_output),
            )

            embedding_files, embedding_stats = self._save_embeddings(
                video_id, prompt_type, language_embedding
            )
            meta_path = self._save_metadata(
                video_id=video_id,
                prompt_type=prompt_type,
                prompt_text=prompt_text,
                original_output=original_output,
                response=answer_payload["response"],
                final_answer=answer_payload["final_answer"],
                answer_parse_method=answer_payload["answer_parse_method"],
                answer_parse_status=answer_payload["answer_parse_status"],
                thinking_content=thinking_content,
                inference_time=inference_time,
                embedding_files=embedding_files,
                embedding_stats=embedding_stats,
                video_path=video_path,
            )

            result["success"] = True
            result["original_output"] = original_output
            result["response"] = answer_payload["response"]
            result["final_answer"] = answer_payload["final_answer"]
            result["answer_parse_method"] = answer_payload["answer_parse_method"]
            result["answer_parse_status"] = answer_payload["answer_parse_status"]
            result["thinking_content"] = thinking_content
            result["inference_time_seconds"] = inference_time
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
    parser = argparse.ArgumentParser(description="RynnBrain-CoP-8B Inference for Experiment")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=str(PROJECT_ROOT / "model_weights" / "RynnBrain-CoP" / "RynnBrain-CoP-8B"),
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
        runner = RynnBrainCoPInference(args.model, args.output)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Model initialization failed: {str(exc)}",
                    "model": MODEL_NAME,
                    "mode": "base",
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
