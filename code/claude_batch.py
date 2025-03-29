import os
import argparse
import pdb
import json
import tqdm
from utils import base64_encoding
from tasks import load_tasks


def postprocess_batch(data_tasks, cfg):
    
    save_root = os.path.join(cfg.save_root, f"{cfg.model}_{cfg.mode}")
    output_path = os.path.join(save_root, "output.jsonl")
    
    print(f"Model: {cfg.model}, Mode: {cfg.mode}", flush=True)
    
    ret = {}
    with open(output_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            
            output = item.get("output")
            idx = int(item.get("id").split("_")[-1])
            task = "_".join(item.get("id").split("_")[:-2])
            answer = data_tasks[task]["answers"][idx]
            question = data_tasks[task]["questions"][idx]
            
            if task not in ret:
                ret[task] = {}
            ret[task][idx] = {"question": question, "answer": answer, "output": output}
    
    final_output_root = os.path.join(cfg.final_output_root, f"{cfg.model}")
    if not os.path.exists(final_output_root):
        os.makedirs(final_output_root)
    
    for task in ret:
        results = []
        for i in range(cfg.N_samples):
            try:
                results.append(ret[task][i])
            except:
                print(f"Missing: Task: {task}, Index: {i}")
                results.append({"question": "", "answer": "A", "output": "placeholder, no answer"})
        # if len(ret[task]) != cfg.N_samples:
        #     print(f"Task: {task}, Expected: {cfg.N_samples}, Actual: {len(ret[task])}")
        #     for i in range(cfg.N_samples - len(ret[task])):
        #         ret[task].append({"question": "", "answer": "A", "output": "placeholder, no answer"})
            
        task_output_path = os.path.join(final_output_root, f"{task}_{cfg.mode}.jsonl")
        with open(task_output_path, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")


def parse_args():
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="claude-3-5-haiku-20241022", type=str)
    # parser.add_argument("--model", default="claude-3-5-sonnet-20241022", type=str)
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219", type=str)
    parser.add_argument("--mode", default="l", type=str)
    parser.add_argument("--tasks", default="bench", type=str)
    parser.add_argument("--benchmark_root", default="../data/benchmark/", type=str)
    parser.add_argument("--save_root", default="../data/claude/", type=str)
    parser.add_argument("--final_output_root", default="../outputs/", type=str)
    parser.add_argument("--N_samples", default=200, type=int)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)
        
    data_tasks = load_tasks(cfg)
    postprocess_batch(data_tasks, cfg)
        
    