#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM-4.1V-9B-Thinking Inference Script for Experiment
Supports structured JSON output and proper embedding extraction

GLM-4.1V-Thinking automatically uses Chain-of-Thought reasoning.
No special parameters needed - generation length is controlled by
EMBODIED_MAX_NEW_TOKENS_THINK (default: 12288).
Output format uses <think> and <answer> sections (or similar special tokens)

Embedding extraction method (following paper):
- Vision Encoder Last: Hook on model.model.visual.blocks[-1]
- Vision Projection: Hook on model.model.visual.merger
- Language Model Last: Use forward() with output_hidden_states=True,
                       then average over sequence dimension
"""

import torch
import numpy as np
from transformers import AutoProcessor, Glm4vForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
from PIL import Image
import os
import json
import re
from datetime import datetime
import warnings
import sys
import cv2
from pathlib import Path
import argparse

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
DEFAULT_MAX_NEW_TOKENS_THINK = int(os.environ.get("EMBODIED_MAX_NEW_TOKENS_THINK", "12288"))

MODEL_NAME = "GLM-4.1V"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ROOT = str(PROJECT_ROOT / "embeddings")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from answer_parser import (  # noqa: E402
    assess_answer_quality,
    build_answer_payload,
    clean_response_text,
    score_quality,
    strip_input_echo,
)

SCRIPT_VARIANT = "semantic_yesno_v2"
SEMANTIC_YESNO_OUTPUT = os.environ.get("EMBODIED_OUTPUT_FORMAT", "").strip().lower() in {
    "semantic_yesno",
    "yesno",
    "yes_no",
}
GLM_MAX_NEW_TOKENS = int(os.environ.get("EMBODIED_GLM_MAX_NEW_TOKENS", "4096"))
GLM_RETRY_MAX_NEW_TOKENS = int(os.environ.get("EMBODIED_GLM_RETRY_MAX_NEW_TOKENS", "4096"))
GLM_LONG_THRESHOLD = int(os.environ.get("EMBODIED_GLM_LONG_THRESHOLD", "5000"))
GLM_RESPONSE_REPETITION_PENALTY = float(os.environ.get("EMBODIED_GLM_RESPONSE_REPETITION_PENALTY", "1.03"))
if SEMANTIC_YESNO_OUTPUT:
    GLM_SYSTEM_PROMPT = (
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
        "- Use the normal thinking flow.\n"
        "- Keep the reasoning concise and focused on the visual evidence.\n"
        "- Do not restate the full prompt.\n"
        "- Use no more than 6 short reasoning sentences.\n"
        "- End with exactly one final answer enclosed in <answer> and </answer>.\n"
        "- Put only yes or no inside the <answer> tag.\n"
        "- Do not output option letters such as A or B.\n"
        "- Do not output anything after </answer>.\n"
    )
    RETRY_GUIDANCE = (
        "\n\nPlease answer again from scratch."
        "\nKeep the reasoning shorter than before and make sure the response ends cleanly."
        "\nDo not self-correct repeatedly."
        "\nFinish once with either <answer>yes</answer> or <answer>no</answer>."
    )
else:
    GLM_SYSTEM_PROMPT = (
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
        "- Use the normal thinking flow.\n"
        "- Keep the reasoning concise and focused on the visual evidence.\n"
        "- Do not restate the full prompt or all answer options.\n"
        "- Use no more than 6 short reasoning sentences.\n"
        "- End with exactly one final answer enclosed in <answer> and </answer>.\n"
        "- Put only the chosen option letter inside the <answer> tag.\n"
        "- Do not output anything after </answer>.\n"
    )
    RETRY_GUIDANCE = (
        "\n\nPlease answer again from scratch."
        "\nKeep the reasoning shorter than before and make sure the response ends cleanly."
        "\nDo not self-correct repeatedly."
        "\nFinish once with a single <answer>X</answer> tag."
    )


class GLM4VThinkingInference:
    """GLM-4.1V-9B-Thinking inference with embedding extraction"""
    
    def __init__(self, model_path, output_dir, mode="thinking"):
        self.model_path = os.path.expanduser(model_path)
        self.output_dir = output_dir
        self.mode = mode
        self.model_name = MODEL_NAME
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        ).eval()
        self.stop_suffix_token_ids = [
            token_ids
            for token_ids in [
                self._build_suffix_token_ids("</answer>"),
                self._build_suffix_token_ids("<|end_of_box|>"),
                self._build_suffix_token_ids("<|assistant|>"),
                self._build_suffix_token_ids("\nassistant"),
            ]
            if token_ids
        ]
        
        sys.stderr.close()
        sys.stderr = original_stderr
        
        self.hooks = []
        self.captured_embeddings = {}
        self._register_vision_hooks()
    
    def _register_vision_hooks(self):
        """Register forward hooks to capture vision embeddings only"""
        def make_hook(name):
            def hook(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if getattr(tensor, "is_meta", False):
                    return
                self.captured_embeddings[name] = tensor.detach().cpu().clone()
            return hook
        
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
                visual = self.model.model.visual
                if hasattr(visual, 'blocks') and len(visual.blocks) > 0:
                    self.hooks.append(
                        visual.blocks[-1].register_forward_hook(make_hook('vision_encoder_last'))
                    )
                if hasattr(visual, 'merger'):
                    self.hooks.append(
                        visual.merger.register_forward_hook(make_hook('vision_projection'))
                    )
        except Exception as e:
            pass
    
    def _extract_video_frames(self, video_path, fps=1.0):
        """Extract frames from video at specified FPS"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames
            
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(video_fps / fps) if fps > 0 else int(video_fps)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                frame_count += 1
            cap.release()
        except Exception:
            pass
        return frames
    
    def _extract_thinking_content(self, response):
        """
        Extract thinking content from GLM response.
        GLM-4.1V-Thinking uses special tokens or <think>/<answer> format.
        """
        patterns = [
            (r'<think>(.*?)</think>', r'</think>(.*)'),
            (r'<\|begin_of_think\|>(.*?)<\|end_of_think\|>', r'<\|end_of_think\|>(.*)'),
            (r'\[thinking\](.*?)\[/thinking\]', r'\[/thinking\](.*)'),
        ]
        
        for think_pattern, answer_pattern in patterns:
            think_match = re.search(think_pattern, response, flags=re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                answer_match = re.search(answer_pattern, response, flags=re.DOTALL)
                if answer_match:
                    clean_response = answer_match.group(1).strip()
                else:
                    clean_response = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
                return thinking_content, clean_response
        
        return None, response

    def _normalize_first_response(self, text: str) -> str:
        if not text:
            return ""

        cleaned = clean_response_text(text).strip()
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

        boxed_answer_match = re.search(
            r"(<think>.*?</think>.*?<\|begin_of_box\|>\s*([ABCD])\s*<\|end_of_box\|>)",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if boxed_answer_match:
            return boxed_answer_match.group(1).strip()

        answer_end = cleaned.find("</answer>")
        if answer_end != -1:
            return cleaned[: answer_end + len("</answer>")].strip()

        box_end = cleaned.find("<|end_of_box|>")
        if box_end != -1:
            return cleaned[: box_end + len("<|end_of_box|>")].strip()

        assistant_restart = re.search(r"\n\s*assistant\b", cleaned, flags=re.IGNORECASE)
        if assistant_restart:
            return cleaned[: assistant_restart.start()].strip()

        return cleaned
    
    def _extract_language_embedding(self, inputs):
        """Extract language model embedding using forward pass."""
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_states = outputs.hidden_states[-1]
            language_embedding = last_hidden_states.mean(dim=1)
            return language_embedding.cpu().clone(), list(last_hidden_states.shape)

    def _build_effective_prompt(self, prompt_text):
        return prompt_text + RESPONSE_GUIDANCE

    def _build_retry_prompt(self, prompt_text):
        return self._build_effective_prompt(prompt_text) + RETRY_GUIDANCE

    def _build_messages(self, video_path, prompt_text):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": GLM_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

    def _build_suffix_token_ids(self, text: str):
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return token_ids if token_ids else None

    def _build_stopping_criteria(self):
        suffix_ids_list = self.stop_suffix_token_ids
        if not suffix_ids_list:
            return None

        class _MultiSuffixStoppingCriteria(StoppingCriteria):
            def __init__(self, suffixes):
                super().__init__()
                self.suffixes = suffixes

            def __call__(self, input_ids, scores, **kwargs):
                sequence = input_ids[0].tolist()
                for suffix in self.suffixes:
                    suffix_len = len(suffix)
                    if len(sequence) >= suffix_len and sequence[-suffix_len:] == suffix:
                        return True
                return False

        return StoppingCriteriaList([_MultiSuffixStoppingCriteria(suffix_ids_list)])

    def _generate_candidate(self, video_path, prompt_text, max_new_tokens):
        messages = self._build_messages(video_path, prompt_text)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            inference_start = datetime.now()
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "repetition_penalty": GLM_RESPONSE_REPETITION_PENALTY,
            }
            stopping_criteria = self._build_stopping_criteria()
            if stopping_criteria is not None:
                generate_kwargs["stopping_criteria"] = stopping_criteria
            output = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            inference_end = datetime.now()
            inference_time = (inference_end - inference_start).total_seconds()

        full_output_text = self.processor.decode(output[0], skip_special_tokens=False)
        generated_ids = output[:, inputs['input_ids'].shape[1]:]
        generated_text_with_special = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        generated_text_no_special = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        input_text = self.processor.decode(inputs['input_ids'][0], skip_special_tokens=False)
        stripped_full = clean_response_text(strip_input_echo(full_output_text, input_text))
        response_with_special = clean_response_text(generated_text_with_special) or stripped_full
        if not response_with_special:
            response_with_special = clean_response_text(full_output_text)
        raw_generated_output = clean_response_text(generated_text_no_special) or response_with_special
        raw_generated_output = re.sub(r"<\|(?:user|im_end)\|>\s*$", "", raw_generated_output).strip()
        original_output = self._normalize_first_response(raw_generated_output)

        normalized_for_parse = original_output or response_with_special
        thinking_content, clean_response = self._extract_thinking_content(normalized_for_parse)
        answer_payload = build_answer_payload(
            clean_response,
            normalized_for_parse,
            clean_response_text(generated_text_no_special),
            stripped_full,
        )
        quality = assess_answer_quality(
            raw_generated_output,
            answer_payload['final_answer'],
            answer_payload['answer_parse_method'],
            long_threshold=GLM_LONG_THRESHOLD,
            raw_generated_output=raw_generated_output,
            normalized_output=original_output,
            require_natural_completion=True,
        )
        return {
            'inputs': inputs,
            'inference_time': inference_time,
            'raw_generated_output': raw_generated_output,
            'original_output': original_output,
            'thinking_content': thinking_content,
            'answer_payload': answer_payload,
            'quality': quality,
        }

    def _select_candidate(self, initial_candidate, retry_candidate):
        if retry_candidate is None:
            return initial_candidate, "initial"
        initial_answer = initial_candidate['answer_payload']['final_answer']
        retry_answer = retry_candidate['answer_payload']['final_answer']
        if initial_answer is None and retry_answer is not None:
            return retry_candidate, "retry"
        if retry_answer is None:
            return initial_candidate, "initial"
        if score_quality(retry_candidate['quality']) < score_quality(initial_candidate['quality']):
            return retry_candidate, "retry"
        return initial_candidate, "initial"
    
    def _save_embeddings(self, video_id, prompt_type, language_embedding, language_original_shape):
        """Save embeddings to files and return paths"""
        saved_files = {}
        embedding_stats = {}
        
        for name, emb in self.captured_embeddings.items():
            filename = f"{video_id}_{prompt_type}_{name}.npy"
            filepath = os.path.join(self.output_dir, filename)
            
            original_shape = list(emb.shape)
            if len(emb.shape) == 3:
                emb_to_save = emb.mean(dim=1)
            elif len(emb.shape) == 2:
                emb_to_save = emb.mean(dim=0, keepdim=True)
            else:
                emb_to_save = emb
            
            np.save(filepath, emb_to_save.float().numpy())
            saved_files[name] = filepath
            embedding_stats[name] = {
                'original_shape': original_shape,
                'saved_shape': list(emb_to_save.shape),
                'dtype': str(emb.dtype),
                'mean': float(emb_to_save.float().mean()),
                'std': float(emb_to_save.float().std()),
                'min': float(emb_to_save.float().min()),
                'max': float(emb_to_save.float().max())
            }
        
        filename = f"{video_id}_{prompt_type}_language_model_last.npy"
        filepath = os.path.join(self.output_dir, filename)
        np.save(filepath, language_embedding.float().numpy())
        saved_files['language_model_last'] = filepath
        embedding_stats['language_model_last'] = {
            'original_shape': language_original_shape,
            'saved_shape': list(language_embedding.shape),
            'dtype': str(language_embedding.dtype),
            'mean': float(language_embedding.float().mean()),
            'std': float(language_embedding.float().std()),
            'min': float(language_embedding.float().min()),
            'max': float(language_embedding.float().max())
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
        """Save metadata for this inference"""
        metadata = {
            'model': self.model_name,
            'mode': self.mode,
            'video_id': video_id,
            'video_path': video_path,
            'prompt_type': prompt_type,
            'prompt_text': prompt_text,
            'original_output': original_output,
            'response': response,
            'final_answer': final_answer,
            'answer_parse_method': answer_parse_method,
            'answer_parse_status': answer_parse_status,
            'thinking_content': thinking_content,
            'inference_time_seconds': inference_time,
            'timestamp': datetime.now().isoformat(),
            'embedding_files': embedding_files,
            'embedding_stats': embedding_stats
        }
        
        meta_filename = f"{video_id}_{prompt_type}_metadata.json"
        meta_path = os.path.join(self.output_dir, meta_filename)
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return meta_path
    
    def run_inference(self, video_path, prompt_text, prompt_type, video_id):
        """Run inference and return structured result"""
        result = {
            'success': False,
            'error': None,
            'model': self.model_name,
            'mode': self.mode,
            'video_id': video_id,
            'prompt_type': prompt_type,
            'raw_prompt': prompt_text,
            'original_output': None,
            'response': None,
            'final_answer': None,
            'answer_parse_method': None,
            'answer_parse_status': None,
            'thinking_content': None,
            'inference_time_seconds': None,
            'embeddings_saved': {},
            'metadata_path': None
        }
        
        try:
            self.captured_embeddings = {}
            retry_output = None
            retry_quality = None
            max_new_tokens = min(DEFAULT_MAX_NEW_TOKENS_THINK, GLM_MAX_NEW_TOKENS)
            effective_prompt = self._build_effective_prompt(prompt_text)

            initial_candidate = self._generate_candidate(video_path, effective_prompt, max_new_tokens)
            with torch.no_grad():
                language_embedding, lang_original_shape = self._extract_language_embedding(
                    initial_candidate['inputs']
                )

            selected_candidate = initial_candidate
            selected_pass = "initial"
            if initial_candidate['quality']['should_retry']:
                retry_candidate = self._generate_candidate(
                    video_path,
                    self._build_retry_prompt(prompt_text),
                    min(DEFAULT_MAX_NEW_TOKENS_THINK, GLM_RETRY_MAX_NEW_TOKENS),
                )
                retry_output = retry_candidate['original_output']
                retry_quality = retry_candidate['quality']
                selected_candidate, selected_pass = self._select_candidate(
                    initial_candidate,
                    retry_candidate,
                )
            
            embedding_files, embedding_stats = self._save_embeddings(
                video_id, prompt_type, language_embedding, lang_original_shape
            )
            
            meta_path = self._save_metadata(
                video_id=video_id,
                prompt_type=prompt_type,
                prompt_text=prompt_text,
                original_output=selected_candidate['original_output'],
                response=selected_candidate['answer_payload']['response'],
                final_answer=selected_candidate['answer_payload']['final_answer'],
                answer_parse_method=selected_candidate['answer_payload']['answer_parse_method'],
                answer_parse_status=selected_candidate['answer_payload']['answer_parse_status'],
                thinking_content=selected_candidate['thinking_content'],
                inference_time=selected_candidate['inference_time'],
                embedding_files=embedding_files,
                embedding_stats=embedding_stats,
                video_path=video_path,
            )
            
            result['success'] = True
            result['original_output'] = selected_candidate['original_output']
            result['response'] = selected_candidate['answer_payload']['response']
            result['final_answer'] = selected_candidate['answer_payload']['final_answer']
            result['answer_parse_method'] = selected_candidate['answer_payload']['answer_parse_method']
            result['answer_parse_status'] = selected_candidate['answer_payload']['answer_parse_status']
            result['thinking_content'] = selected_candidate['thinking_content']
            result['inference_time_seconds'] = selected_candidate['inference_time']
            result['embeddings_saved'] = embedding_files
            result['metadata_path'] = meta_path
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description='GLM-4.1V-9B-Thinking Inference for Experiment')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default=str(
            PROJECT_ROOT
            / 'model_weights'
            / 'GLM-4.1V-thinking'
            / 'GLM-4.1V-9B-Thinking'
        ),
        help='Path to model directory',
    )
    parser.add_argument('--video', '-v', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--prompt', '-p', type=str, required=True,
                        help='Prompt text')
    parser.add_argument('--prompt-type', '-t', type=str, required=True,
                        choices=['simple', 'detailed', 'embodied_simple', 'embodied_detailed'],
                        help='Prompt type')
    parser.add_argument('--video-id', type=str, required=True,
                        help='Video ID for naming output files')
    parser.add_argument('--output', '-o', type=str,
                        default=f'{EMBEDDING_ROOT}/{MODEL_NAME}/thinking',
                        help='Output directory for embeddings')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    try:
        model = GLM4VThinkingInference(
            model_path=args.model,
            output_dir=args.output,
            mode="thinking"
        )
    except Exception as e:
        result = {
            'success': False,
            'error': f"Model initialization failed: {str(e)}",
            'model': MODEL_NAME,
            'mode': 'thinking',
            'video_id': args.video_id,
            'prompt_type': args.prompt_type,
            'raw_prompt': args.prompt,
            'original_output': None,
            'response': None,
            'final_answer': None,
            'answer_parse_method': None,
            'answer_parse_status': None,
            'thinking_content': None,
            'inference_time_seconds': None,
            'embeddings_saved': {},
            'metadata_path': None
        }
        print(json.dumps(result, ensure_ascii=False))
        return
    
    result = model.run_inference(
        video_path=args.video,
        prompt_text=args.prompt,
        prompt_type=args.prompt_type,
        video_id=args.video_id
    )
    
    print(json.dumps(result, ensure_ascii=False))
    model.cleanup()


if __name__ == "__main__":
    main()
