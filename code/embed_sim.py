import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='multimodal embeddings similarity')
    parser.add_argument('--gpu', type=int, default=0, choices=[0, 1],
                      help='GPU device index (0 or 1)')
    parser.add_argument('--model_name', type=str, 
                      default="OpenGVLab/InternVL2_5-2B",
                      help='Model name')
    parser.add_argument('--legal_json_path', type=str,
                      default="../data/benchmark/legal.jsonl",
                      help='Path to legal JSON file')
    parser.add_argument('--image_folder', type=str,
                      default="../data/benchmark/legal/",
                      help='Path to image folder')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size')
    return parser.parse_args()

class Config:
    def __init__(self, args):
        self.model_name = args.model_name
        self.legal_json_path = args.legal_json_path
        self.image_folder = args.image_folder
        self.batch_size = args.batch_size
        
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            self.device = f"cuda:{args.gpu}"
        else:
            self.device = "cpu"
            
        self.dtype = torch.bfloat16

class HookState:
    def __init__(self):
        self.last_hidden_state = None
        
    def update(self, hidden_state):
        self.last_hidden_state = hidden_state.detach().cpu().float()
        
    def get(self):
        if self.last_hidden_state is None:
            raise ValueError("No hidden state captured")
        return self.last_hidden_state

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform

def load_image(image_file, input_size=448, max_num=1):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values

def hook_fn(hook_state):
    def _hook_fn(module, input, output):
        hook_state.update(output)
    return _hook_fn

def setup_model_and_tokenizer(cfg):
    model = AutoModel.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=cfg.dtype
    ).eval().to(cfg.device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, 
        trust_remote_code=True
    )
    
    hook_state = HookState()
    model.language_model.model.norm.register_forward_hook(hook_fn(hook_state))
    
    return model, tokenizer, hook_state

def process_text_modality(model, tokenizer, hook_state, fen, device):
    inputs = tokenizer(
        f"FEN: {fen} \nSummarize this in one word: ",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        _ = model.language_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        
        hidden = hook_state.get()
        last_token_idx = inputs["attention_mask"].sum()-1
        return hidden[0, last_token_idx].numpy()

def process_vision_modality(model, tokenizer, hook_state, image_path, device, dtype):
    pixel_values = load_image(image_path, max_num=1).to(device).to(dtype)
    
    generation_config = dict(max_new_tokens=1)
    
    with torch.no_grad():
        _ = model.chat(
            tokenizer,
            pixel_values,
            f"<image>\nSummarize this in one word: ",
            generation_config
        )
        
        hidden = hook_state.get()
        return hidden[0, -1].numpy()

def process_vl_modality(model, tokenizer, hook_state, fen, image_path, device, dtype):
    pixel_values = load_image(image_path, max_num=1).to(device).to(dtype)
    
    generation_config = dict(max_new_tokens=1)

    with torch.no_grad():
        _ = model.chat(
            tokenizer, 
            pixel_values,
            f"<image>\nFEN: {fen} \nSummarize this in one word: ",
            generation_config
        )
        
        hidden = hook_state.get()
        return hidden[0, -1].numpy()

def calculate_similarities(text_repr, vision_repr, vl_repr):
    sim_tv = cosine_similarity([text_repr], [vision_repr])[0][0]
    sim_tvl = cosine_similarity([text_repr], [vl_repr])[0][0]
    sim_vvl = cosine_similarity([vision_repr], [vl_repr])[0][0]
    return sim_tv, sim_tvl, sim_vvl

def load_dataset(json_path):
    with open(json_path) as f:
        return [json.loads(line) for line in f]

def process_samples(dataset, cfg, model, tokenizer, hook_state):
    results = []
    errors = []
    
    for sample in tqdm(dataset):
        fen = sample["fen"]
        image_path = os.path.join(cfg.image_folder, 
            fen.split(" ")[0].replace("/", "_") + ".png")
        
        try:
            text_repr = process_text_modality(
                model, tokenizer, hook_state, fen, cfg.device
            )
            vision_repr = process_vision_modality(
                model, tokenizer, hook_state, image_path, cfg.device, cfg.dtype
            )
            vl_repr = process_vl_modality(
                model, tokenizer, hook_state, fen, image_path, cfg.device, cfg.dtype
            )

            similarities = calculate_similarities(text_repr, vision_repr, vl_repr)

            results.append({
                "fen": fen,
                "text_vision_sim": float(similarities[0]),
                "text_vl_sim": float(similarities[1]), 
                "vision_vl_sim": float(similarities[2])
            })
        except Exception as e:
            errors.append({"fen": fen, "error": str(e)})
            continue
            
    return results, errors

def calculate_averages(results):
    return {
        "text_vision": np.mean([r["text_vision_sim"] for r in results]),
        "text_vl": np.mean([r["text_vl_sim"] for r in results]),
        "vision_vl": np.mean([r["vision_vl_sim"] for r in results])
    }

def save_results(results, avg_sims, errors, output_file="modality_similarities_Intern26B.json"):
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "averages": avg_sims,
            "errors": errors
        }, f, indent=2)

def main():
    args = parse_args()
    cfg = Config(args)
    
    dataset = load_dataset(cfg.legal_json_path)
    model, tokenizer, hook_state = setup_model_and_tokenizer(cfg)
    results, errors = process_samples(dataset, cfg, model, tokenizer, hook_state)
    
    if results:
        avg_sims = calculate_averages(results)
        
        print(f"\nProcessed {len(results)} samples successfully. Failed: {len(errors)}")
        print("\nAverage Cosine Similarities:")
        print(f"Text-Vision: {avg_sims['text_vision']:.4f}")
        print(f"Text-VL: {avg_sims['text_vl']:.4f}")
        print(f"Vision-VL: {avg_sims['vision_vl']:.4f}")
        
        save_results(results, avg_sims, errors)
    else:
        print("No valid results to analyze")

if __name__ == "__main__":
    main()

"""
internvl2.5 + legal move:
2B:
Average Cosine Similarities:
Text-Vision: 0.3557
Text-VL: 0.4021
Vision-VL: 0.8916

4B:
Average Cosine Similarities:
Text-Vision: 0.0989
Text-VL: 0.1107
Vision-VL: 0.9749

8B:
Average Cosine Similarities:
Text-Vision: 0.5449
Text-VL: 0.5818
Vision-VL: 0.9799

26B:
Average Cosine Similarities:
Text-Vision: 0.4314
Text-VL: 0.4786
Vision-VL: 0.9619

38B:
Average Cosine Similarities:
Text-Vision: -0.1012
Text-VL: -0.0835
Vision-VL: 0.9815
"""