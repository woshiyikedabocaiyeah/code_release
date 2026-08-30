#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Thinking Inference Script

Qwen-Thinking model outputs thinking content that may:
1. Start with <think> and end with </think>
2. Start directly with thinking content and end with </think>
3. Have thinking content followed by </think> then answer

Output fields:
- original_output: Raw model output (complete, unmodified)
- thinking_content: Extracted reasoning (content before </think>)
- response: Final answer only (content after </think>)
"""

import torch
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, StoppingCriteria, StoppingCriteriaList
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
RETRY_MAX_NEW_TOKENS = int(os.environ.get("EMBODIED_QWEN_RETRY_MAX_NEW_TOKENS", "4096"))
QWEN_THINK_MAX_NEW_TOKENS = int(os.environ.get("EMBODIED_QWEN_MAX_NEW_TOKENS", "4096"))
THINK_DO_SAMPLE = os.environ.get("EMBODIED_QWEN_THINK_DO_SAMPLE", "true").lower() in {
    "1", "true", "yes", "y", "on"
}
THINK_TEMPERATURE = float(os.environ.get("EMBODIED_QWEN_THINK_TEMPERATURE", "0.0"))
THINK_TOP_P = float(os.environ.get("EMBODIED_QWEN_THINK_TOP_P", "0.95"))
THINK_TOP_K = int(os.environ.get("EMBODIED_QWEN_THINK_TOP_K", "20"))
THINK_REPETITION_PENALTY = float(os.environ.get("EMBODIED_QWEN_THINK_REPETITION_PENALTY", "1.05"))
THINK_DO_SAMPLE_EFFECTIVE = THINK_DO_SAMPLE and THINK_TEMPERATURE > 0.0

MODEL_NAME = "Qwen"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ROOT = str(PROJECT_ROOT / "embeddings")
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
    QWEN_SYSTEM_PROMPT = (
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
        "\nUse the normal thinking flow and finish once with <answer>yes</answer> or <answer>no</answer>."
    )
else:
    QWEN_SYSTEM_PROMPT = (
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
        "\nUse the normal thinking flow and finish once with <answer>X</answer>."
    )
QUALITY_LONG_THRESHOLD = int(os.environ.get("EMBODIED_QWEN_LONG_THRESHOLD", "3500"))


def extract_thinking_and_answer_qwen(response):
    """
    Extract thinking content and answer from Qwen-Thinking response.
    
    Qwen-Thinking has multiple output formats:
    1. <think>thinking...</think>answer
    2. thinking...</think>answer  (no opening <think> tag)
    3. <think>thinking...</think><answer>answer</answer>
    
    The key marker is </think> which separates thinking from answer.
    """
    original_output = response
    thinking_content = None
    answer = response
    
    if not response:
        return None, "", original_output
    
    think_end_pos = response.find('</think>')
    
    if think_end_pos != -1:
        thinking_part = response[:think_end_pos]
        answer_part = response[think_end_pos + len('</think>'):].strip()
        
        if thinking_part.startswith('<think>'):
            thinking_part = thinking_part[len('<think>'):]
        
        thinking_content = thinking_part.strip()
        
        answer_match = re.search(r'<answer>(.*?)</answer>', answer_part, flags=re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = answer_part.strip() if answer_part.strip() else thinking_content
        
        return thinking_content, answer, original_output
    
    if response.startswith('<think>'):
        thinking_content = response[len('<think>'):].strip()
        return thinking_content, thinking_content, original_output
    
    return None, response, original_output


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
        return f"{partial_answer_match.group(1).strip()}</answer>"

    answer_end = cleaned.find("</answer>")
    if answer_end != -1:
        return cleaned[: answer_end + len("</answer>")].strip()

    assistant_restart = re.search(r"\n\s*assistant\b", cleaned, flags=re.IGNORECASE)
    if assistant_restart:
        return cleaned[: assistant_restart.start()].strip()

    return cleaned


class Qwen3VLThinkingInference:
    def __init__(self, model_path, output_dir, mode="thinking"):
        self.model_path = os.path.expanduser(model_path)
        self.output_dir = output_dir
        self.mode = mode
        self.model_name = MODEL_NAME
        
        os.makedirs(output_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.stop_suffix_token_ids = [
            token_ids
            for token_ids in [
                self._build_suffix_token_ids("</answer>"),
                self._build_suffix_token_ids("<|im_end|>"),
                self._build_suffix_token_ids("\nassistant"),
                self._build_suffix_token_ids("\nassistant\n"),
            ]
            if token_ids
        ]
        
        sys.stderr.close()
        sys.stderr = original_stderr
        
        self.hooks = []
        self.captured_embeddings = {}
        self._register_vision_hooks()

    def _build_effective_prompt(self, prompt_text):
        return prompt_text + RESPONSE_GUIDANCE

    def _build_retry_prompt(self, prompt_text):
        return self._build_effective_prompt(prompt_text) + RETRY_GUIDANCE

    def _build_messages(self, video_path, prompt_text):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": QWEN_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
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

    def _generate_output(self, model_inputs, max_new_tokens):
        generate_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': THINK_DO_SAMPLE_EFFECTIVE,
            'repetition_penalty': THINK_REPETITION_PENALTY,
        }
        stopping_criteria = self._build_stopping_criteria()
        if stopping_criteria is not None:
            generate_kwargs['stopping_criteria'] = stopping_criteria
        if THINK_DO_SAMPLE_EFFECTIVE:
            generate_kwargs['temperature'] = THINK_TEMPERATURE
            generate_kwargs['top_p'] = THINK_TOP_P
            generate_kwargs['top_k'] = THINK_TOP_K
        return self.model.generate(
            **model_inputs,
            **generate_kwargs,
        )

    def _parse_candidate(self, raw_response):
        raw_text = clean_response_text(re.sub(r'<\|[^>]+\|>', '', raw_response)).strip()
        cleaned_response = normalize_first_response(raw_response)
        thinking_content, response_candidate, _ = extract_thinking_and_answer_qwen(cleaned_response)
        answer_payload = build_answer_payload(
            response_candidate,
            cleaned_response,
            raw_text,
        )
        quality = assess_answer_quality(
            raw_text,
            answer_payload['final_answer'],
            answer_payload['answer_parse_method'],
            long_threshold=QUALITY_LONG_THRESHOLD,
            raw_generated_output=raw_text,
            normalized_output=cleaned_response,
            require_natural_completion=True,
        )
        return {
            'raw_response': raw_response,
            'original_output': raw_response,
            'thinking_content': thinking_content,
            'answer_payload': answer_payload,
            'quality': quality,
        }

    def _run_retry(self, video_path, prompt_text):
        retry_prompt = self._build_retry_prompt(prompt_text)
        messages = self._build_messages(video_path, retry_prompt)
        retry_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            retry_ids = self._generate_output(retry_inputs, RETRY_MAX_NEW_TOKENS)
        retry_trimmed = [out[len(inp):] for inp, out in zip(retry_inputs.input_ids, retry_ids)]
        retry_raw = self.processor.batch_decode(
            retry_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        return self._parse_candidate(retry_raw)

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
    
    def _register_vision_hooks(self):
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
                    self.hooks.append(visual.blocks[-1].register_forward_hook(make_hook('vision_encoder_last')))
                if hasattr(visual, 'merger'):
                    self.hooks.append(visual.merger.register_forward_hook(make_hook('vision_projection')))
        except Exception:
            pass
    
    def _extract_video_frames(self, video_path, num_frames=8):
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
        except Exception:
            pass
        return frames
    
    def _extract_language_embedding(self, inputs):
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_states = outputs.hidden_states[-1]
            language_embedding = last_hidden_states.mean(dim=1)
            return language_embedding.cpu().clone(), list(last_hidden_states.shape)
    
    def _save_embeddings(self, video_id, prompt_type, language_embedding, language_original_shape):
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
        retry_output,
        thinking_content,
        response,
        final_answer,
        answer_parse_method,
        answer_parse_status,
        inference_time,
        embedding_files,
        embedding_stats,
        video_path,
    ):
        metadata = {
            'model': self.model_name,
            'mode': self.mode,
            'video_id': video_id,
            'video_path': video_path,
            'prompt_type': prompt_type,
            'prompt_text': prompt_text,
            'original_output': original_output,
            'retry_output': retry_output,
            'retry_max_new_tokens': RETRY_MAX_NEW_TOKENS,
            'generation_max_new_tokens': min(DEFAULT_MAX_NEW_TOKENS_THINK, QWEN_THINK_MAX_NEW_TOKENS),
            'think_do_sample': THINK_DO_SAMPLE_EFFECTIVE,
            'think_temperature': THINK_TEMPERATURE,
            'think_top_p': THINK_TOP_P if THINK_DO_SAMPLE_EFFECTIVE else None,
            'think_top_k': THINK_TOP_K if THINK_DO_SAMPLE_EFFECTIVE else None,
            'think_repetition_penalty': THINK_REPETITION_PENALTY,
            'thinking_content': thinking_content,
            'response': response,
            'final_answer': final_answer,
            'answer_parse_method': answer_parse_method,
            'answer_parse_status': answer_parse_status,
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
        result = {
            'success': False,
            'error': None,
            'model': self.model_name,
            'mode': self.mode,
            'video_id': video_id,
            'prompt_type': prompt_type,
            'raw_prompt': prompt_text,
            'original_output': None,
            'thinking_content': None,
            'response': None,
            'final_answer': None,
            'answer_parse_method': None,
            'answer_parse_status': None,
            'inference_time_seconds': None,
            'embeddings_saved': {},
            'metadata_path': None
        }
        
        try:
            self.captured_embeddings = {}
            retry_output = None
            retry_quality = None
            effective_prompt = self._build_effective_prompt(prompt_text)
            
            # Use native video input so processor applies model-default video sampling.
            messages = self._build_messages(video_path, effective_prompt)
            
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                language_embedding, lang_original_shape = self._extract_language_embedding(inputs)
            
            with torch.no_grad():
                inference_start = datetime.now()
                generated_ids = self._generate_output(
                    inputs,
                    min(DEFAULT_MAX_NEW_TOKENS_THINK, QWEN_THINK_MAX_NEW_TOKENS),
                )
                inference_end = datetime.now()
                inference_time = (inference_end - inference_start).total_seconds()
                
                generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
                raw_response = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )[0]

            initial_candidate = self._parse_candidate(raw_response)
            selected_candidate = initial_candidate
            selected_pass = "initial"
            if initial_candidate['quality']['should_retry']:
                retry_candidate = self._run_retry(video_path, prompt_text)
                retry_output = retry_candidate['original_output']
                retry_quality = retry_candidate['quality']
                selected_candidate, selected_pass = self._select_candidate(
                    initial_candidate,
                    retry_candidate,
                )

            thinking_content = selected_candidate['thinking_content']
            answer_payload = selected_candidate['answer_payload']
            selected_output = selected_candidate['original_output']
            
            embedding_files, embedding_stats = self._save_embeddings(
                video_id, prompt_type, language_embedding, lang_original_shape
            )
            
            meta_path = self._save_metadata(
                video_id, prompt_type, prompt_text, selected_output,
                retry_output,
                thinking_content, answer_payload['response'],
                answer_payload['final_answer'],
                answer_payload['answer_parse_method'],
                answer_payload['answer_parse_status'],
                inference_time, embedding_files,
                embedding_stats, video_path
            )
            
            result['success'] = True
            result['original_output'] = selected_output
            result['thinking_content'] = thinking_content
            result['response'] = answer_payload['response']
            result['final_answer'] = answer_payload['final_answer']
            result['answer_parse_method'] = answer_payload['answer_parse_method']
            result['answer_parse_status'] = answer_payload['answer_parse_status']
            result['inference_time_seconds'] = inference_time
            result['embeddings_saved'] = embedding_files
            result['metadata_path'] = meta_path
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL-8B-Thinking Inference')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default=str(PROJECT_ROOT / 'model_weights' / 'Qwen-Thinking' / 'Qwen3-VL-8B-Thinking'),
    )
    parser.add_argument('--video', '-v', type=str, required=True)
    parser.add_argument('--prompt', '-p', type=str, required=True)
    parser.add_argument('--prompt-type', '-t', type=str, required=True,
                        choices=['simple', 'detailed', 'embodied_simple', 'embodied_detailed'])
    parser.add_argument('--video-id', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default=f'{EMBEDDING_ROOT}/{MODEL_NAME}/thinking')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    try:
        model = Qwen3VLThinkingInference(model_path=args.model, output_dir=args.output, mode="thinking")
    except Exception as e:
        result = {
            'success': False, 'error': f"Model initialization failed: {str(e)}",
            'model': MODEL_NAME, 'mode': 'thinking', 'video_id': args.video_id,
            'prompt_type': args.prompt_type, 'raw_prompt': args.prompt,
            'original_output': None, 'thinking_content': None, 'response': None,
            'final_answer': None, 'answer_parse_method': None, 'answer_parse_status': None,
            'inference_time_seconds': None, 'embeddings_saved': {}, 'metadata_path': None
        }
        print(json.dumps(result, ensure_ascii=False))
        return
    
    result = model.run_inference(args.video, args.prompt, args.prompt_type, args.video_id)
    print(json.dumps(result, ensure_ascii=False))
    model.cleanup()


if __name__ == "__main__":
    main()
