import argparse
from vllm import SamplingParams
import pdb
import warnings
warnings.filterwarnings("ignore")
import os
import json

from models import (
    model_mllama,
    model_qwen,
    model_intern,
    model_pixtral,
    model_llava_onevision,
    model_gemma,
    model_phi,
)
from models import (
    format_mllama,
    format_qwen,
    format_intern,
    format_pixtral,
    format_llava_onevision,
    format_gemma,
    format_phi,
)
from tasks import load_tasks
from utils import vllm_format, parse_outputs


def generate(cfg, model, prompts, images, stop_token_ids):
    
    inputs = vllm_format(prompts, images, cfg)
    
    sampling_params = SamplingParams(temperature=cfg.temperature,
                                     top_p=cfg.top_p,
                                     max_tokens=cfg.max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     frequency_penalty=cfg.frequency_penalty,
                                     presence_penalty=cfg.presence_penalty,
                                     seed=cfg.seed,
                                     )
    
    print("\033[32mExample prompt:", inputs[0]["prompt"], "\033[0m", flush=True)
    outputs = model.generate(inputs, sampling_params)
    outputs = parse_outputs(outputs)
    print("\033[34mExample output:", outputs[0], "\033[0m", flush=True)
    
    return outputs


def run(cfg):

    data_tasks = load_tasks(cfg)
    
    if cfg.model_name in ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-72B-Instruct"]:
        model = model_qwen(cfg)
    elif cfg.model_name in ["InternVL2_5-8B", "InternVL2_5-78B"]:
        model = model_intern(cfg)
    elif cfg.model_name in ["Llama-3.2-11B-Vision-Instruct", "Llama-3.2-90B-Vision-Instruct"]:
        model = model_mllama(cfg)
    elif cfg.model_name in ["pixtral-12b"]:
        model = model_pixtral(cfg)
    elif cfg.model_name in ["llava-onevision-qwen2-7b-ov-hf", "llava-onevision-qwen2-72b-ov-hf"]:
        model = model_llava_onevision(cfg)
    elif cfg.model_name in ["gemma-3-12b-it", "gemma-3-27b-it"]:
        model = model_gemma(cfg)
    elif cfg.model_name in ["Phi-4-multimodal-instruct"]:
        model = model_phi(cfg)
    else:
        raise ValueError("Invalid model name")
    
    save_root = os.path.join(cfg.save_dir, f"{cfg.model_name}")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    for task in data_tasks:
        print(f"[Started]: Task: {task}, Model: {cfg.model_name}, Mode: {cfg.mode}", flush=True)
        with open(os.path.join(save_root, f"{task}_{cfg.mode}.jsonl"), "w") as f:
            questions = data_tasks[task]["questions"]
            answers = data_tasks[task]["answers"]
            images = data_tasks[task]["images"]
            
            if cfg.model_name in ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-72B-Instruct"]:
                prompts, stop_token_ids = format_qwen(cfg, questions)
            elif cfg.model_name in ["InternVL2_5-8B", "InternVL2_5-78B"]:
                prompts, stop_token_ids = format_intern(cfg, questions)
            elif cfg.model_name in ["Llama-3.2-11B-Vision-Instruct", "Llama-3.2-90B-Vision-Instruct"]:
                prompts, stop_token_ids = format_mllama(cfg, questions)
            elif cfg.model_name in ["pixtral-12b"]:
                prompts, stop_token_ids = format_pixtral(cfg, questions)
            elif cfg.model_name in ["llava-onevision-qwen2-7b-ov-hf", "llava-onevision-qwen2-72b-ov-hf"]:
                prompts, stop_token_ids = format_llava_onevision(cfg, questions)
            elif cfg.model_name in ["gemma-3-12b-it", "gemma-3-27b-it"]:
                prompts, stop_token_ids = format_gemma(cfg, questions)
            elif cfg.model_name in ["Phi-4-multimodal-instruct"]:
                prompts, stop_token_ids = format_phi(cfg, questions)
            else:
                raise ValueError("Invalid model name")
            
            if cfg.debug_samples > 0:
                prompts = prompts[:cfg.debug_samples]
            
            outputs = generate(cfg, model, prompts, images, stop_token_ids)
            print(f"[Finished]: Task: {task}, Model: {cfg.model_name}, Mode: {cfg.mode}", flush=True)
            
            for i in range(len(outputs)):
                ret = {"question": questions[i], 
                        "answer": answers[i], 
                        "output": outputs[i]}
                f.write(json.dumps(ret) + "\n")


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_root", default="/data/models/benchmark_models", type=str)
    parser.add_argument("--model_name", default="Qwen2.5-VL-3B-Instruct", type=str)
    parser.add_argument("--benchmark_root", default="../data/benchmark/", type=str)
    parser.add_argument("--tasks", default="puzzle", type=str)
    parser.add_argument("--mode", default="v", type=str)
    parser.add_argument("--save_dir", default="../outputs/", type=str)

    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--num_gpus", default=2, type=int)
    parser.add_argument("--gpu_memory_utilization", default=0.95, type=float)
    parser.add_argument("--disable_mm_preprocessor_cache", default=True, type=bool)
    parser.add_argument("--max_num_seqs", default=16, type=int)
    parser.add_argument("--max_model_len", default=4096, type=int)
    parser.add_argument("--frequency_penalty", default=0, type=float)
    parser.add_argument("--presence_penalty", default=0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug_samples", default=0, type=int)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)
        
    run(cfg)
