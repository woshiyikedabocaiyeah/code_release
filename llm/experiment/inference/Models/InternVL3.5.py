#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternVL3.5-8B Inference Script for Experiment

Output fields:
- original_output: Raw model output (complete, unmodified)
- thinking_content: Extracted reasoning (from <think> tags)
- response: Final answer only (cleaned)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

MODEL_NAME = "InternVL3.5"
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SCRIPT_VARIANT = "semantic_yesno_v1"

R1_SYSTEM_PROMPT = """You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
DEFAULT_MAX_NEW_TOKENS_BASE = int(os.environ.get("EMBODIED_MAX_NEW_TOKENS_BASE", "4096"))
DEFAULT_MAX_NEW_TOKENS_THINK = int(os.environ.get("EMBODIED_MAX_NEW_TOKENS_THINK", "12288"))
INTERNVL_MAX_NEW_TOKENS_BASE = int(os.environ.get("EMBODIED_INTERNVL_MAX_NEW_TOKENS_BASE", "512"))
INTERNVL_MAX_NEW_TOKENS_THINK = int(os.environ.get("EMBODIED_INTERNVL_MAX_NEW_TOKENS_THINK", "4096"))
RETRY_MAX_NEW_TOKENS = int(os.environ.get("EMBODIED_INTERNVL_RETRY_MAX_NEW_TOKENS", "4096"))
QUALITY_LONG_THRESHOLD = int(os.environ.get("EMBODIED_INTERNVL_LONG_THRESHOLD", "3500"))
THINK_DO_SAMPLE = os.environ.get("EMBODIED_INTERNVL_THINK_DO_SAMPLE", "true").lower() in {
    "1", "true", "yes", "y", "on"
}
THINK_TEMPERATURE = float(os.environ.get("EMBODIED_INTERNVL_THINK_TEMPERATURE", "0.0"))
THINK_TOP_P = float(os.environ.get("EMBODIED_INTERNVL_THINK_TOP_P", "0.95"))
RESPONSE_REPETITION_PENALTY = float(
    os.environ.get("EMBODIED_INTERNVL_RESPONSE_REPETITION_PENALTY", "1.0")
)
THINK_DO_SAMPLE_EFFECTIVE = THINK_DO_SAMPLE and THINK_TEMPERATURE > 0.0
SEMANTIC_YESNO_OUTPUT = os.environ.get("EMBODIED_OUTPUT_FORMAT", "").strip().lower() in {
    "semantic_yesno",
    "yesno",
    "yes_no",
}
if SEMANTIC_YESNO_OUTPUT:
    RESPONSE_GUIDANCE = (
        "\n\nResponse requirements:\n"
        "- Keep the reasoning concise and focused on the video.\n"
        "- Do not restate the entire prompt.\n"
        "- Decide on exactly one final yes/no answer and do not reconsider it afterward.\n"
        "- End with exactly one final answer enclosed in <answer> and </answer>.\n"
        "- Put only yes or no inside the <answer> tag.\n"
        "- Do not output option letters such as A or B.\n"
        "- Stop immediately after the closing </answer> tag.\n"
    )
    RETRY_GUIDANCE = (
        "\n\nPlease answer again from scratch."
        "\nKeep the reasoning shorter than before and make sure the response ends cleanly."
        "\nUse the same <think>...</think><answer>...</answer> format."
        "\nDo not list multiple possible answers or revise the final answer once chosen."
        "\nThe response is invalid if the closing </think> tag is missing."
        "\nThe final answer must be exactly yes or no."
    )
else:
    RESPONSE_GUIDANCE = (
        "\n\nResponse requirements:\n"
        "- Keep the reasoning concise and focused on the video.\n"
        "- Do not restate the entire prompt or all options.\n"
        "- Decide on exactly one final option and do not reconsider it afterward.\n"
        "- End with exactly one final answer enclosed in <answer> and </answer>.\n"
        "- Put only the option letter inside the <answer> tag.\n"
        "- Stop immediately after the closing </answer> tag.\n"
    )
    RETRY_GUIDANCE = (
        "\n\nPlease answer again from scratch."
        "\nKeep the reasoning shorter than before and make sure the response ends cleanly."
        "\nUse the same <think>...</think><answer>...</answer> format."
        "\nDo not list multiple possible answers or revise the final answer once chosen."
        "\nThe response is invalid if the closing </think> tag is missing."
    )


def extract_thinking_and_answer(response):
    """Extract thinking content and answer from model response."""
    original_output = response
    thinking_content = None
    answer = response
    
    if not response:
        return None, "", original_output
    
    match1 = re.search(r'<think>(.*?)</think>\s*<answer>(.*?)</answer>', response, flags=re.DOTALL)
    if match1:
        return match1.group(1).strip(), match1.group(2).strip(), original_output
    
    match2 = re.search(r'<think>(.*?)<answer>(.*?)</answer>', response, flags=re.DOTALL)
    if match2:
        return match2.group(1).strip(), match2.group(2).strip(), original_output
    
    match3 = re.search(r'<think>(.*?)</think>(.*)', response, flags=re.DOTALL)
    if match3:
        thinking = match3.group(1).strip()
        after = match3.group(2).strip()
        return thinking, after if after else response, original_output
    
    match4 = re.search(r'^<think>(.*)', response, flags=re.DOTALL)
    if match4:
        content = match4.group(1)
        answer_match = re.search(r'<answer>(.*?)</answer>', content, flags=re.DOTALL)
        if answer_match:
            think_end = content.find('<answer>')
            thinking = content[:think_end].strip()
            return thinking, answer_match.group(1).strip(), original_output
        return content.strip(), content.strip(), original_output
    
    return None, response, original_output


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(img) for img in images])


class InternVL35Inference:
    def __init__(self, model_path, output_dir, mode="base"):
        self.model_path = os.path.expanduser(model_path)
        self.output_dir = output_dir
        self.mode = mode
        self.model_name = MODEL_NAME
        
        os.makedirs(output_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        
        sys.stderr.close()
        sys.stderr = original_stderr
        
        self.hooks = []
        self.captured_embeddings = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if getattr(tensor, "is_meta", False):
                    return
                self.captured_embeddings[name] = tensor.detach().cpu().clone()
            return hook
        
        try:
            if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'encoder'):
                encoder = self.model.vision_model.encoder
                if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
                    self.hooks.append(encoder.layers[-1].register_forward_hook(make_hook('vision_encoder_last')))
            
            if hasattr(self.model, 'mlp1'):
                self.hooks.append(self.model.mlp1.register_forward_hook(make_hook('vision_projection')))
            
            if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
                lm_model = self.model.language_model.model
                if hasattr(lm_model, 'layers') and len(lm_model.layers) > 0:
                    self.hooks.append(lm_model.layers[-1].register_forward_hook(make_hook('language_model_last')))
        except Exception:
            pass
    
    def _extract_video_frames(self, video_path, num_segments=8):
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total_frames - 1, num_segments, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
        except Exception:
            pass
        return frames
    
    def _load_video(self, video_path, num_segments=8, max_num=1):
        frames = self._extract_video_frames(video_path, num_segments)
        if not frames:
            return None, []
        
        pixel_values_list = []
        num_patches_list = []
        for frame in frames:
            pv = load_image(frame, max_num=max_num)
            pixel_values_list.append(pv)
            num_patches_list.append(pv.shape[0])
        
        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values, num_patches_list

    def _build_effective_question(self, question):
        return question + RESPONSE_GUIDANCE

    def _build_retry_question(self, question):
        return self._build_effective_question(question) + RETRY_GUIDANCE

    def _parse_candidate(self, raw_response, enable_thinking):
        if enable_thinking:
            thinking_content, response_candidate, original_output = extract_thinking_and_answer(raw_response)
        else:
            thinking_content = None
            response_candidate = raw_response
            original_output = raw_response

        cleaned_output = clean_response_text(original_output)
        answer_payload = build_answer_payload(
            response_candidate,
            cleaned_output,
        )
        quality = assess_answer_quality(
            cleaned_output,
            answer_payload['final_answer'],
            answer_payload['answer_parse_method'],
            long_threshold=QUALITY_LONG_THRESHOLD,
            raw_generated_output=original_output,
            normalized_output=original_output,
            require_natural_completion=enable_thinking,
            require_closed_think=enable_thinking,
        )
        return {
            'raw_response': raw_response,
            'original_output': original_output,
            'thinking_content': thinking_content,
            'answer_payload': answer_payload,
            'quality': quality,
        }

    def _run_retry(self, pixel_values, num_patches_list, question, enable_thinking):
        retry_prompt = self._build_retry_question(question)
        if enable_thinking:
            retry_prompt = R1_SYSTEM_PROMPT + "\n\n" + retry_prompt
        retry_config = {
            'max_new_tokens': RETRY_MAX_NEW_TOKENS,
            'do_sample': THINK_DO_SAMPLE_EFFECTIVE if enable_thinking else False,
        }
        if enable_thinking and THINK_DO_SAMPLE_EFFECTIVE:
            retry_config['temperature'] = THINK_TEMPERATURE
            retry_config['top_p'] = THINK_TOP_P
        if RESPONSE_REPETITION_PENALTY != 1.0:
            retry_config['repetition_penalty'] = RESPONSE_REPETITION_PENALTY
        with torch.no_grad():
            retry_raw, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                retry_prompt,
                retry_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
        return self._parse_candidate(retry_raw, enable_thinking)

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
    
    def _save_embeddings(self, video_id, prompt_type):
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
            'generation_max_new_tokens': min(
                DEFAULT_MAX_NEW_TOKENS_THINK if self.mode == "thinking" else DEFAULT_MAX_NEW_TOKENS_BASE,
                INTERNVL_MAX_NEW_TOKENS_THINK if self.mode == "thinking" else INTERNVL_MAX_NEW_TOKENS_BASE,
            ),
            'think_do_sample': THINK_DO_SAMPLE_EFFECTIVE if self.mode == "thinking" else False,
            'think_temperature': THINK_TEMPERATURE if self.mode == "thinking" else None,
            'think_top_p': THINK_TOP_P if self.mode == "thinking" and THINK_DO_SAMPLE_EFFECTIVE else None,
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
            pixel_values, num_patches_list = self._load_video(video_path, num_segments=8, max_num=1)
            if pixel_values is None:
                result['error'] = f"Failed to load video from {video_path}"
                return result
            
            self.captured_embeddings = {}
            
            question = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))]) + prompt_text
            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
            retry_output = None
            retry_quality = None
            
            enable_thinking = (self.mode == "thinking")
            effective_question = self._build_effective_question(question)
            
            if enable_thinking:
                full_question = R1_SYSTEM_PROMPT + "\n\n" + effective_question
                generation_config = {
                    'max_new_tokens': min(DEFAULT_MAX_NEW_TOKENS_THINK, INTERNVL_MAX_NEW_TOKENS_THINK),
                    'do_sample': THINK_DO_SAMPLE_EFFECTIVE,
                }
                if THINK_DO_SAMPLE_EFFECTIVE:
                    generation_config['temperature'] = THINK_TEMPERATURE
                    generation_config['top_p'] = THINK_TOP_P
                if RESPONSE_REPETITION_PENALTY != 1.0:
                    generation_config['repetition_penalty'] = RESPONSE_REPETITION_PENALTY
            else:
                full_question = question
                generation_config = {
                    'max_new_tokens': min(DEFAULT_MAX_NEW_TOKENS_BASE, INTERNVL_MAX_NEW_TOKENS_BASE),
                    'do_sample': False,
                }
            
            with torch.no_grad():
                inference_start = datetime.now()
                raw_response, _ = self.model.chat(
                    self.tokenizer, pixel_values, full_question, generation_config,
                    num_patches_list=num_patches_list, history=None, return_history=True
                )
                inference_end = datetime.now()
                inference_time = (inference_end - inference_start).total_seconds()

            initial_candidate = self._parse_candidate(raw_response, enable_thinking)
            selected_candidate = initial_candidate
            selected_pass = "initial"
            if enable_thinking and initial_candidate['quality']['should_retry']:
                retry_candidate = self._run_retry(
                    pixel_values, num_patches_list, question, enable_thinking
                )
                retry_output = retry_candidate['original_output']
                retry_quality = retry_candidate['quality']
                selected_candidate, selected_pass = self._select_candidate(
                    initial_candidate,
                    retry_candidate,
                )

            thinking_content = selected_candidate['thinking_content']
            original_output = selected_candidate['original_output']
            answer_payload = selected_candidate['answer_payload']
            
            embedding_files, embedding_stats = self._save_embeddings(video_id, prompt_type)
            
            meta_path = self._save_metadata(
                video_id, prompt_type, prompt_text, original_output,
                retry_output,
                thinking_content, answer_payload['response'],
                answer_payload['final_answer'],
                answer_payload['answer_parse_method'],
                answer_payload['answer_parse_status'],
                inference_time, embedding_files,
                embedding_stats, video_path
            )
            
            result['success'] = True
            result['original_output'] = original_output
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
    parser = argparse.ArgumentParser(description='InternVL3.5-8B Inference')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default=str(PROJECT_ROOT / 'model_weights' / 'InternVL3.5' / 'InternVL3_5-8B'),
    )
    parser.add_argument('--video', '-v', type=str, required=True)
    parser.add_argument('--prompt', '-p', type=str, required=True)
    parser.add_argument('--prompt-type', '-t', type=str, required=True,
                        choices=['simple', 'detailed', 'embodied_simple', 'embodied_detailed'])
    parser.add_argument('--video-id', type=str, required=True)
    parser.add_argument('--mode', type=str, default='base', choices=['base', 'thinking'])
    parser.add_argument('--output', '-o', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'{EMBEDDING_ROOT}/{MODEL_NAME}/{args.mode}'
    
    os.makedirs(args.output, exist_ok=True)
    
    try:
        model = InternVL35Inference(model_path=args.model, output_dir=args.output, mode=args.mode)
    except Exception as e:
        result = {
            'success': False, 'error': f"Model initialization failed: {str(e)}",
            'model': MODEL_NAME, 'mode': args.mode, 'video_id': args.video_id,
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
