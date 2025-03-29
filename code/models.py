from transformers import AutoTokenizer
from vllm import LLM
import pdb
import os

def model_mllama(cfg):

    llm = LLM(
        model=os.path.join(cfg.model_root, cfg.model_name),
        tensor_parallel_size=cfg.num_gpus,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
        enforce_eager=True,
        disable_mm_preprocessor_cache=cfg.disable_mm_preprocessor_cache
    )
    
    return llm


def format_mllama(cfg, questions):
    
    if cfg.mode == "v" or cfg.mode == "vl":
        messages = [[
            {"role": "system", "content": "You are a helpful assistant"}, 
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{question}"}]}
                    ] for question in questions]
        
    elif cfg.mode == "l":
        messages = [[
            {"role": "system", "content": "You are a helpful assistant"}, 
            {"role": "user", "content": f"{question}"}] for question in questions]
    
    else:
        raise ValueError("Invalid mode")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(cfg.model_root, cfg.model_name))
    prompts = tokenizer.apply_chat_template(messages, 
                                            add_generation_prompt=True, 
                                            tokenize=False)
    stop_token_ids = None
    
    return prompts, stop_token_ids


def model_qwen(cfg):

    llm = LLM(
        model=os.path.join(cfg.model_root, cfg.model_name),
        tensor_parallel_size=cfg.num_gpus,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
        disable_mm_preprocessor_cache=cfg.disable_mm_preprocessor_cache,
    )
    
    return llm


def format_qwen(cfg, questions):

    if cfg.mode == "v" or cfg.mode == "vl":
        prompts = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n" for question in questions]
    
    elif cfg.mode == "l":
        prompts = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                "<|im_start|>assistant\n" for question in questions]

    else:
        raise ValueError("Invalid mode")
    
    stop_token_ids = None
    
    return prompts, stop_token_ids


def model_intern(cfg):
    
    llm = LLM(
        model=os.path.join(cfg.model_root, cfg.model_name),
        trust_remote_code=True,
        tensor_parallel_size=cfg.num_gpus,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
        disable_mm_preprocessor_cache=cfg.disable_mm_preprocessor_cache,
    )
    
    return llm


def format_intern(cfg, questions):
    
    if cfg.mode == "v" or cfg.mode == "vl":
        messages = [[{'role': 'system', 'content': 'You are a helpful assistant'},
                    {'role': 'user', 'content': f"<image>\n{question}"}] for question in questions]
      
    elif cfg.mode == "l":
        messages = [[{'role': 'system', 'content': 'You are a helpful assistant'},
                    {'role': 'user', 'content': question}] for question in questions]

    else:
        raise ValueError("Invalid mode")
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(cfg.model_root, cfg.model_name), trust_remote_code=True)
    
    prompts = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    
    return prompts, stop_token_ids

