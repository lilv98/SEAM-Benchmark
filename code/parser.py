import os
import json
import argparse
import pdb
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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


def compute_agreement(answers, preds):
    
    l_v = 0
    l_vl = 0
    v_vl = 0
    l_v_vl = 0
    
    for i in range(len(answers[0])):
        
        answer_l = answers[0][i]
        answer_v = answers[1][i]
        answer_vl = answers[2][i]
        
        pred_l = preds[0][i]
        pred_v = preds[1][i]
        pred_vl = preds[2][i]
        
        if (pred_l == pred_v == pred_vl) and pred_l != "Z":
            l_v_vl += 1
        
        if (pred_l == pred_v) and pred_l != "Z":
            l_v += 1
        
        if (pred_l == pred_vl) and pred_l != "Z":
            l_vl += 1
        
        if (pred_v == pred_vl) and pred_v != "Z":
            v_vl += 1
    
    total = len(answers[0])
    
    return l_v / total, l_vl / total, v_vl / total, l_v_vl / total


def get_metrics(cfg, tasks, model_name):
    
    avg_acc_l = 0
    avg_acc_v = 0
    avg_acc_vl = 0
    
    avg_inv_l = 0
    avg_inv_v = 0
    avg_inv_vl = 0
    
    avg_agree_l_v = 0
    avg_agree_l_vl = 0
    avg_agree_v_vl = 0
    avg_agree_l_v_vl = 0
    
    for task in tasks:
        
        all_answers = []
        all_preds = []
        
        for mode in ["l", "v", "vl"]:
            root = cfg.outputs_root + model_name
            log = f"{task}_{mode}.jsonl"
            answers = []
            preds = []
            with open(os.path.join(root, log), "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    answers.append(data["answer"])
                    preds.append(data["final_answer"])
            
            all_answers.append(answers)
            all_preds.append(preds)
            
            acc = compute_accuracy(answers, preds)
            inv = compute_invalid(preds)
            if cfg.verbose:
                print(f"Model: {model_name}, Task: {task}, Mode: {mode}, Accuracy: {acc}, Invalid: {inv}")
            
            avg_acc_l += acc if mode == "l" else 0
            avg_acc_v += acc if mode == "v" else 0
            avg_acc_vl += acc if mode == "vl" else 0
            
            avg_inv_l += inv if mode == "l" else 0
            avg_inv_v += inv if mode == "v" else 0
            avg_inv_vl += inv if mode == "vl" else 0
            

        agreements = compute_agreement(all_answers, all_preds)
        if cfg.verbose:
            print(f"Agreement between L and V: {agreements[0]}")
            print(f"Agreement between L and VL: {agreements[1]}")
            print(f"Agreement between V and VL: {agreements[2]}")
            print(f"Agreement between L, V, and VL: {agreements[3]}")
        
        avg_agree_l_v += agreements[0]
        avg_agree_l_vl += agreements[1]
        avg_agree_v_vl += agreements[2]
        avg_agree_l_v_vl += agreements[3]
    
    total_tasks = len(tasks)
    
    print(f"Model: {model_name}, Average Accuracy L: {round(avg_acc_l / total_tasks, 3)}")
    print(f"Model: {model_name}, Average Accuracy V: {round(avg_acc_v / total_tasks, 3)}")
    print(f"Model: {model_name}, Average Accuracy VL: {round(avg_acc_vl / total_tasks, 3)}")
    
    # print(f"Model: {model_name}, Average Invalid L: {round(avg_inv_l / total_tasks, 3)}")
    # print(f"Model: {model_name}, Average Invalid V: {round(avg_inv_v / total_tasks, 3)}")
    # print(f"Model: {model_name}, Average Invalid VL: {round(avg_inv_vl / total_tasks, 3)}")
    
    # print(f"Model: {model_name}, Average Agreement L and V: {round(avg_agree_l_v / total_tasks, 3)}")
    # print(f"Model: {model_name}, Average Agreement L and VL: {round(avg_agree_l_vl / total_tasks, 3)}")
    # print(f"Model: {model_name}, Average Agreement V and VL: {round(avg_agree_v_vl / total_tasks, 3)}")
    # print(f"Model: {model_name}, Average Agreement L, V, and VL: {round(avg_agree_l_v_vl / total_tasks, 3)}")

def get_metrics_variant(cfg, tasks, model_name):
    
    avg_acc_v = 0
    avg_acc_vl = 0
    
    avg_inv_v = 0
    avg_inv_vl = 0
    
    for task in tasks:
        
        all_answers = []
        all_preds = []
        
        for mode in ["v", "vl"]:
            root = cfg.outputs_root + model_name
            log = f"{task}_{mode}.jsonl"
            answers = []
            preds = []
            with open(os.path.join(root, log), "r") as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    answers.append(data["answer"])
                    preds.append(data["final_answer"])
            
            all_answers.append(answers)
            all_preds.append(preds)
            
            acc = compute_accuracy(answers, preds)
            inv = compute_invalid(preds)
            if cfg.verbose:
                print(f"Model: {model_name}, Task: {task}, Mode: {mode}, Accuracy: {acc}, Invalid: {inv}")
            
            avg_acc_v += acc if mode == "v" else 0
            avg_acc_vl += acc if mode == "vl" else 0

            avg_inv_v += inv if mode == "v" else 0
            avg_inv_vl += inv if mode == "vl" else 0
            

    total_tasks = len(tasks)
    
    print(f"Model: {model_name}, Average Accuracy V: {round(avg_acc_v / total_tasks, 3)}")
    print(f"Model: {model_name}, Average Accuracy VL: {round(avg_acc_vl / total_tasks, 3)}")
    
    # print(f"Model: {model_name}, Average Invalid V: {round(avg_inv_v / total_tasks, 3)}")
    # print(f"Model: {model_name}, Average Invalid VL: {round(avg_inv_vl / total_tasks, 3)}")
    


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_root", default="/data/models/benchmark_models", type=str)
    parser.add_argument("--outputs_root", default="../outputs/", type=str)
    parser.add_argument("--tasks", default="graph", type=str)
    parser.add_argument("--save_dir", default="../outputs/", type=str)
    parser.add_argument("--verbose", default=False, type=bool)
    parser.add_argument("--variant", default=False, type=bool)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)

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
    
    if cfg.variant:
        all_models = [
            "Qwen2.5-VL-7B-Instruct", 
            "Qwen2.5-VL-72B-Instruct", 
        ]
        for model_name in all_models:
            get_metrics_variant(cfg, tasks, model_name)
        
    else:
        all_models = [
                    'claude-3-5-haiku-20241022',
                    'claude-3-5-sonnet-20241022',
                    'claude-3-7-sonnet-20250219',
                    # "gpt-4o-2",
                    # "gpt-4o-2-extract",
                    # "gpt-4o-mini-2",
                    # "Qwen2.5-VL-7B-Instruct", 
                    # "Qwen2.5-VL-72B-Instruct",
                    # "InternVL2_5-8B", 
                    # "InternVL2_5-78B",
                    # "Llama-3.2-11B-Vision-Instruct", 
                    # "Llama-3.2-90B-Vision-Instruct",
                    # "pixtral-12b",
                    # "llava-onevision-qwen2-7b-ov-hf", 
                    # "llava-onevision-qwen2-72b-ov-hf",
                    # "gemma-3-12b-it",
                    # "gemma-3-27b-it"
                    ]
        
        for model_name in all_models:
            get_metrics(cfg, tasks, model_name)
        
