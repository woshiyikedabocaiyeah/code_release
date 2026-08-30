#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Instruct Inference Script for Experiment
Supports structured JSON output and proper embedding extraction
For use with run_main_experiment.py

This is the base (non-thinking) version of Qwen3-VL

Embedding extraction method (following paper):
- Vision Encoder Last: Hook on model.model.visual.blocks[-1]
- Vision Projection: Hook on model.model.visual.merger
- Language Model Last: Use forward() with output_hidden_states=True,
                       then average over sequence dimension

Reference: "we averaged over the sequence dimension, condensing multiple 
           token-level embeddings into a single vector per video"
"""

import torch
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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
DEFAULT_MAX_NEW_TOKENS_BASE = int(os.environ.get("EMBODIED_MAX_NEW_TOKENS_BASE", "4096"))

MODEL_NAME = "Qwen"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ROOT = str(PROJECT_ROOT / "embeddings")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from answer_parser import (  # noqa: E402
    build_answer_payload,
    clean_response_text,
    strip_input_echo,
)


class Qwen3VLInstructInference:
    """Qwen3-VL-8B-Instruct inference with embedding extraction"""
    
    def __init__(self, model_path, output_dir, mode="base"):
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
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
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
    
    def _extract_video_frames(self, video_path, num_frames=8):
        """Extract frames from video using uniform sampling"""
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
        """
        Extract language model embedding using forward pass.
        Following the paper: get hidden states from last layer, then average over sequence.
        """
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_states = outputs.hidden_states[-1]
            
            language_embedding = last_hidden_states.mean(dim=1)
            
            return language_embedding.cpu().clone()
    
    def _save_embeddings(self, video_id, prompt_type, language_embedding):
        """Save embeddings to files and return paths"""
        saved_files = {}
        embedding_stats = {}
        
        for name, emb in self.captured_embeddings.items():
            filename = f"{video_id}_{prompt_type}_{name}.npy"
            filepath = os.path.join(self.output_dir, filename)
            
            if len(emb.shape) == 3:
                emb_to_save = emb.mean(dim=1)  # [batch, hidden] or [hidden]
            elif len(emb.shape) == 2:
                emb_to_save = emb.mean(dim=0, keepdim=True)  # [1, hidden]
            else:
                emb_to_save = emb
            
            np.save(filepath, emb_to_save.float().numpy())
            saved_files[name] = filepath
            embedding_stats[name] = {
                'original_shape': list(emb.shape),
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
            'thinking_content': None,
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
            
            # Use native video input so processor applies model-default video sampling.
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                language_embedding = self._extract_language_embedding(inputs)
            
            with torch.no_grad():
                inference_start = datetime.now()
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS_BASE,
                    do_sample=False,
                )
                inference_end = datetime.now()
                inference_time = (inference_end - inference_start).total_seconds()
                
                generated_ids_trimmed = [
                    out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
                ]
                original_output = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )[0]
                full_output = self.processor.decode(generated_ids[0], skip_special_tokens=False)
                input_text = self.processor.decode(inputs.input_ids[0], skip_special_tokens=False)
                stripped_full = clean_response_text(strip_input_echo(full_output, input_text))
                cleaned_output = clean_response_text(re.sub(r'<\|[^>]+\|>', '', original_output))
                answer_payload = build_answer_payload(
                    cleaned_output,
                    stripped_full,
                    clean_response_text(full_output),
                )
            
            embedding_files, embedding_stats = self._save_embeddings(
                video_id, prompt_type, language_embedding
            )
            
            meta_path = self._save_metadata(
                video_id=video_id,
                prompt_type=prompt_type,
                prompt_text=prompt_text,
                original_output=original_output,
                response=answer_payload['response'],
                final_answer=answer_payload['final_answer'],
                answer_parse_method=answer_payload['answer_parse_method'],
                answer_parse_status=answer_payload['answer_parse_status'],
                inference_time=inference_time,
                embedding_files=embedding_files,
                embedding_stats=embedding_stats,
                video_path=video_path,
            )
            
            result['success'] = True
            result['original_output'] = original_output
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
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL-8B-Instruct Inference for Experiment')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default=str(PROJECT_ROOT / 'model_weights' / 'Qwen' / 'Qwen3-VL-8B-Instruct'),
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
                        default=f'{EMBEDDING_ROOT}/{MODEL_NAME}/base',
                        help='Output directory for embeddings')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    try:
        model = Qwen3VLInstructInference(
            model_path=args.model,
            output_dir=args.output,
            mode="base"
        )
    except Exception as e:
        result = {
            'success': False,
            'error': f"Model initialization failed: {str(e)}",
            'model': MODEL_NAME,
            'mode': 'base',
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
