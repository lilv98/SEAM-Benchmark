import os
import argparse
import pdb
import json
import tqdm
from utils import base64_encoding
from tasks import load_tasks
from openai import AzureOpenAI
import pandas as pd

def setup_env():
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-10-21",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    
    return client


def file_upload(client, cfg):
    
    save_root = os.path.join(cfg.save_root, f"{cfg.model}_{cfg.mode}")
    ids_path = os.path.join(save_root, "ids.jsonl")
    
    uploaded_files = []
    files = os.listdir(save_root)
    for file in files:
        if file.startswith("chunk_"):
            file = client.files.create(
                file=open(os.path.join(save_root, file), "rb"),
                purpose="batch"
            )
            print(f"File Uploaded: {file.id}")
            uploaded_files.append(file)
    
    with open(ids_path, "w") as f:
        for file in uploaded_files:
            ret = {"file_id": file.id, "file_name": file.filename}
            f.write(json.dumps(ret) + "\n")


def create_batch(client, fileid):
    
    batch = client.batches.create(
        input_file_id=fileid,
        endpoint="/chat/completions",
        completion_window="24h",
    )
    
    return batch


def submit_batch(client, cfg):
    
    save_root = os.path.join(cfg.save_root, f"{cfg.model}_{cfg.mode}")
    ids_path = os.path.join(save_root, "ids.jsonl")
    
    batches = []
    with open(ids_path, "r") as f:
        for line in f:
            file = json.loads(line.strip())
            file_id = file.get("file_id")
            batch = create_batch(client, file_id)
            
            assert batch.input_file_id == file_id
            ret = {"batch_id": batch.id, "file_id": file_id, "file_name": file.get("file_name")}
            batches.append(ret)

            print(f"Batch Submitted: {batch.id}")
    
    # overwrite the ids
    with open(ids_path, "w") as f:
        for batch in batches:
            f.write(json.dumps(batch) + "\n")


def retrieve_batch(client, cfg):
    
    save_root = os.path.join(cfg.save_root, f"{cfg.model}_{cfg.mode}")
    ids_path = os.path.join(save_root, "ids.jsonl")
    output_path = os.path.join(save_root, "output.jsonl")
    
    ret = []
    with open(ids_path, "r") as f:
        for line in f:
            batch = json.loads(line.strip())
            batch_id = batch.get("batch_id")
            
            print(f"Retrieving Batch: {batch_id}")
            response = client.batches.retrieve(batch_id)
            status = response.status
            
            if status == "completed":
                print(f"Batch Completed: {batch_id}")
                output_file_id = response.output_file_id
                print(f"Output File ID: {output_file_id}")
                file_response = client.files.content(output_file_id)
                raw_responses = file_response.text.strip().split('\n')
                
                for raw_response in tqdm.tqdm(raw_responses):
                    json_response = json.loads(raw_response) 
                    try:
                        output = json_response['response']['body']['choices'][0]['message']['content']
                    except:
                        print(f"Error: {json_response}")
                    custom_id = json_response['custom_id']
                    ret.append({"id": custom_id, "output": output})
            else:
                print(f"Batch Status: {status}")
    
    with open(output_path, "w") as f:
        for item in ret:
            f.write(json.dumps(item) + "\n")


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


def formatting_prompt_text_only(idx, question, model, task, mode):
    
    prompt = {"custom_id": f"{task}_{mode}_{idx}",
              "method": "POST", 
              "url": "/chat/completions", 
              "body": {"model": model, 
                       "messages": [
                                    {"role": "user", 
                                     "content": question}],
                       "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "max_completion_tokens": cfg.max_new_tokens,
                        "frequency_penalty": cfg.frequency_penalty,
                        "presence_penalty": cfg.presence_penalty,
                        "seed": cfg.seed
                        }}
    
    return prompt


def formatting_prompt_with_vision(idx, question, img, model, task, mode):
    
    prompt = {"custom_id": f"{task}_{mode}_{idx}",
              "method": "POST", 
              "url": "/chat/completions", 
              "body": {"model": model, 
                       "messages": [
                                    {"role": "user", 
                                     "content": [
                                                 {"type":"text",
                                                  "text":question},
                                                 {"type":"image_url",
                                                  "image_url":{"url":f"data:image/jpeg;base64,{base64_encoding(img)}"}}
                                                ]
                                    }],
                       "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "max_completion_tokens": cfg.max_new_tokens,
                        "frequency_penalty": cfg.frequency_penalty,
                        "presence_penalty": cfg.presence_penalty,
                        "seed": cfg.seed
                       }}
    
    return prompt


def generate_text_only_each(client, question, cfg):
    
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_completion_tokens=cfg.max_new_tokens,
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
        seed=cfg.seed
    )
    
    return response.choices[0].message.content


def generate_with_vision_each(client, question, img, cfg):
    
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "user", 
             "content": [
                        {"type":"image_url",
                          "image_url":{"url":f"data:image/jpeg;base64,{base64_encoding(img)}"}},
                        {"type":"text",
                          "text":question},
                        ]
            }
        ],
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_completion_tokens=cfg.max_new_tokens,
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
        seed=cfg.seed
    )
    
    return response.choices[0].message.content


def load_data(data_tasks, cfg):

    ret = []
    for task in data_tasks:
        print(f"[Started]: Task: {task}, Model: {cfg.model}, Mode: {cfg.mode}", flush=True)
        
        questions = data_tasks[task]["questions"]
        images = data_tasks[task]["images"]
        
        for i in tqdm.tqdm(range(len(questions))):
            question = questions[i]
            img = images[i]
            
            if cfg.mode == "v" or cfg.mode == "vl":
                prompt = formatting_prompt_with_vision(i, question, img, cfg.model, task, cfg.mode)
            elif cfg.mode == "l":
                prompt = formatting_prompt_text_only(i, question, cfg.model, task, cfg.mode)
            else:
                raise ValueError("Invalid mode")
            ret.append(prompt)
    
    return ret


def run_each(client, data_tasks, cfg):
    
    for task in data_tasks:
        print(f"[Started]: Task: {task}, Model: {cfg.model}, Mode: {cfg.mode}", flush=True)
        data_task = data_tasks[task]
        questions = data_task["questions"]
        images = data_task["images"]
        answers = data_task["answers"][:cfg.debug_samples]
        
        outputs = []
        for i in tqdm.tqdm(range(len(questions))):
            
            question = questions[i]
            img = images[i]
            
            if cfg.mode == "v" or cfg.mode == "vl":
                output = generate_with_vision_each(client, question, img, cfg)
            elif cfg.mode == "l":
                output = generate_text_only_each(client, question, cfg)
            else:
                raise ValueError("Invalid mode")
            outputs.append(output)
            print(output, flush=True)
            
            if len(outputs) == cfg.debug_samples:
                break


def parse_args():
    
    parser = argparse.ArgumentParser()
    # Batch: gpt-4o-mini-2 or gpt-4o-2
    # Each: gpt-4o-mini or gpt-4o
    parser.add_argument("--model", default="gpt-4o-mini-2", type=str)
    parser.add_argument("--mode", default="l", type=str)
    parser.add_argument("--tasks", default="bench", type=str)
    parser.add_argument("--benchmark_root", default="../data/benchmark/", type=str)
    parser.add_argument("--save_root", default="../data/openai/", type=str)
    parser.add_argument("--final_output_root", default="../outputs/", type=str)
    parser.add_argument("--stage", default="data", type=str)
    parser.add_argument("--chunk_size", default=5000, type=int)
    parser.add_argument("--debug_samples", default=5, type=int)
    parser.add_argument("--N_samples", default=200, type=int)

    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_new_tokens", default=2048, type=int)
    parser.add_argument("--frequency_penalty", default=0, type=float)
    parser.add_argument("--presence_penalty", default=0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    cfg = parse_args()
    print("Configurations:", flush=True)
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}", flush=True)
        
    if cfg.stage != "debug":
        save_root = os.path.join(cfg.save_root, f"{cfg.model}_{cfg.mode}")
        if not os.path.exists(save_root):
            os.makedirs(save_root)

    client = setup_env()
    
    if cfg.stage == "data":

        data_tasks = load_tasks(cfg)
        data = load_data(data_tasks, cfg)
        
        for chunk in range(0, len(data), cfg.chunk_size):
            with open(os.path.join(save_root, f"chunk_{chunk}.jsonl"), "w") as f:
                for item in data[chunk:chunk+cfg.chunk_size]:
                    f.write(json.dumps(item) + "\n")

    elif cfg.stage == "file":
        file_upload(client, cfg)
    
    elif cfg.stage == "submit":
        submit_batch(client, cfg)
        
    elif cfg.stage == "retrieve":
        retrieve_batch(client, cfg)
        
    elif cfg.stage == "post":
        data_tasks = load_tasks(cfg)
        postprocess_batch(data_tasks, cfg)
    
    elif cfg.stage == "debug":
        data_tasks = load_tasks(cfg)
        run_each(client, data_tasks, cfg)
    
    else:
        raise ValueError("Invalid stage")
    