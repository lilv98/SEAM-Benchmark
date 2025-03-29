import os
import json
import argparse
import pdb
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def format_prompt(prediction):
    
    prompt = f"Here is the complete predicted answer for a multiple-choice question\n\n***{prediction}***\n\n"
    prompt += "Your task: Extract the final answer (the best option) from the text above.\n"
    prompt += "Ignore the reasoning process and any inconsistence in the above complete predicted answer.\n"
    prompt += "It is usually in the format 'The best option is [letter]' at the end of the complete predicted answer.\n"
    prompt += "If found, reply with the letter 'A', 'B', 'C', or 'D'.\n"
    prompt += "Otherwise, reply with 'Z'."
    
    return prompt

def format_qwen_text(cfg, questions):
    
    messages = [[{'role': 'system', 'content': 'You are a helpful final answer extractor.'},
                {'role': 'user', 'content': format_prompt(question)}] for question in questions]
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(cfg.model_root, cfg.extraction_model_name))
    prompts = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
    
    return prompts


def compute_accuracy(answers, preds):
    
    correct = 0
    total = len(answers)
    for i in range(total):
        if answers[i] == preds[i]:
            correct += 1
    
    return correct / total


def compute_invalid(preds):
    
    invalid = 0
    total = len(preds)
    for pred in preds:
        if pred == "Z":
            invalid += 1
    
    return invalid / total


def get_final_answers(cfg, tasks, modes):
    
    for mode in modes:
        for task in tasks:
            root = cfg.outputs_root + cfg.model_name
            log = f"{task}_{mode}.jsonl"
            answers = []
            outputs = []
            questions = []
            pred_letters = []
            with open(os.path.join(root, log), "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    answers.append(data["answer"])
                    outputs.append(data["output"])
                    questions.append(data["question"])
            
            prompts = format_qwen_text(cfg, outputs)
            
            preds = llm.generate(prompts, sampling_params)
            for pred in preds:
                generated_text = pred.outputs[0].text
                if generated_text not in ["A", "B", "C", "D", "Z"]:
                    # pdb.set_trace()
                    print(generated_text)
                    generated_text = "Z"
                pred_letters.append(generated_text)
                # if generated_text == "Z":
                #     print(pred.prompt)
                #     print("\n\n")
            
            acc = compute_accuracy(answers, pred_letters)
            inv = compute_invalid(pred_letters)
            print(f"Task: {task}, Mode: {mode}, Accuracy: {acc}, Invalid: {inv}")
            
            with open(os.path.join(root, log), "w") as f:
                for i in range(len(outputs)):
                    ret = {"question": questions[i], 
                            "answer": answers[i], 
                            "output": outputs[i],
                            "final_answer": pred_letters[i]}
                    f.write(json.dumps(ret) + "\n")


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_root", default="/data/models/benchmark_models", type=str)
    parser.add_argument("--model_name", default="claude-3-5-sonnet-20241022", type=str)
    parser.add_argument("--extraction_model_name", default="Qwen2.5-VL-7B-Instruct", type=str)
    parser.add_argument("--outputs_root", default="../outputs/", type=str)
    parser.add_argument("--tasks", default="bench", type=str)
    parser.add_argument("--modes", default="l,v,vl", type=str)
    parser.add_argument("--save_dir", default="../outputs/", type=str)

    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens", default=4, type=int)
    parser.add_argument("--num_gpus", default=4, type=int)
    parser.add_argument("--gpu_memory_utilization", default=0.95, type=float)
    parser.add_argument("--max_num_seqs", default=16, type=int)
    parser.add_argument("--max_model_len", default=4096, type=int)
    parser.add_argument("--seed", default=42, type=int)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)

    llm = LLM(
        model=os.path.join(cfg.model_root, cfg.extraction_model_name),
        tensor_parallel_size=cfg.num_gpus,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        max_num_seqs=cfg.max_num_seqs,
    )
    sampling_params = SamplingParams(temperature=cfg.temperature,
                                    top_p=cfg.top_p,
                                    max_tokens=cfg.max_tokens,
                                    seed=cfg.seed)

    modes = cfg.modes.split(",")
    
    chess_tasks = ["fork", "legal", "puzzle", "eval"]
    chem_tasks = ["carbon", "hydrogen", "weight", "caption"]
    music_tasks = ["notes", "measures", "forms", "rhythm"]
    graph_tasks = ["path_counting", "path_existence", "shortest_path", "bfs_traversal"]
    
    chess_res_tasks = ["fork_res", "legal_res", "puzzle_res", "eval_res"]
    chess_bw_tasks = ["fork_bw", "legal_bw", "puzzle_bw", "eval_bw"]
    chem_rot_tasks = ["carbon_rot", "hydrogen_rot", "weight_rot", "caption_rot"]

    tasks_dict = {
        "chess": chess_tasks,
        "chem": chem_tasks,
        "music": music_tasks,
        "graph": graph_tasks,
        "chess_res": chess_res_tasks,
        "chess_bw": chess_bw_tasks,
        "chem_rot": chem_rot_tasks
    }
    
    if cfg.tasks == "bench":
        tasks = chess_tasks + chem_tasks + music_tasks + graph_tasks
    elif cfg.tasks == "variant":
        tasks = chess_res_tasks + chess_bw_tasks + chem_rot_tasks
    elif cfg.tasks in tasks_dict:
        tasks = tasks_dict[cfg.tasks]
    else:
        tasks = cfg.tasks.split(",")
    
    print(tasks)

    get_final_answers(cfg, tasks, modes)
        
