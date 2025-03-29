import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
import logging
import tempfile
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from matplotlib.gridspec import GridSpec
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CIFAR100_FINE_LABELS = {
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver", 5: "bed", 6: "bee",
    7: "beetle", 8: "bicycle", 9: "bottle", 10: "bowl", 11: "boy", 12: "bridge", 13: "bus",
    14: "butterfly", 15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
    20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach", 25: "couch",
    26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur", 30: "dolphin", 31: "elephant",
    32: "flatfish", 33: "forest", 34: "fox", 35: "girl", 36: "hamster", 37: "house",
    38: "kangaroo", 39: "keyboard", 40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion",
    44: "lizard", 45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
    50: "mouse", 51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid", 55: "otter",
    56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree", 60: "plain", 61: "plate",
    62: "poppy", 63: "porcupine", 64: "possum", 65: "rabbit", 66: "raccoon", 67: "ray",
    68: "road", 69: "rocket", 70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
    75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider", 80: "squirrel",
    81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table", 85: "tank", 86: "telephone",
    87: "television", 88: "tiger", 89: "tractor", 90: "train", 91: "trout", 92: "tulip",
    93: "turtle", 94: "wardrobe", 95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm"
}

class LayerHookState:
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.hidden_states = []
        
    def update(self, hidden_states):
        self.hidden_states.append(hidden_states.detach().cpu().float())
        
    def clear(self):
        self.hidden_states = []

class QwenLayerHookManager:
    def __init__(self, model):
        self.model = model
        self.layer_hooks = []
        self.hook_states = {}
        self.num_layers = len(model.model.layers)
        
        # Only register the final_norm hook since we only want the last layer
        self.hook_states["final_norm"] = LayerHookState("final_norm")
        
    def register_hooks(self):
        # Register only the final normalization layer hook
        hook = self.model.model.norm.register_forward_hook(self._create_hook_fn("final_norm"))
        self.layer_hooks.append(hook)
        
        return self
        
    def _create_hook_fn(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            if hidden_states.dim() == 3:
                self.hook_states[layer_idx].update(hidden_states)
            elif hidden_states.dim() == 2:
                self.hook_states[layer_idx].update(hidden_states.unsqueeze(0))
            else:
                logger.warning(f"Unexpected hidden state dimensions: {hidden_states.shape} at layer {layer_idx}")
                self.hook_states[layer_idx].update(hidden_states)
        return hook_fn
    
    def remove_hooks(self):
        for hook in self.layer_hooks:
            hook.remove()
        self.layer_hooks = []
    
    def get_layer_embeddings(self):
        return self.hook_states
    
    def clear_embeddings(self):
        for key in self.hook_states:
            self.hook_states[key].clear()

def hash_str(text):
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()

def get_image_path(args, sample_id, index=None):
    if args.task_name == "seam":
        return os.path.join(args.image_folder, sample_id.split(" ")[0].replace("/", "_") + ".png")
    else:
        return os.path.join(args.image_folder, f"{sample_id}.png")

def load_cifar100_dataset(num_samples=200):
    cifar100 = load_dataset("cifar100", split="train")
    total_samples = len(cifar100)
    
    if num_samples >= total_samples:
        indices = list(range(total_samples))
    else:
        indices = np.random.choice(range(total_samples), num_samples, replace=False)
    
    samples = []
    for idx in indices:
        item = cifar100[int(idx)]
        label_id = item["fine_label"]
        label_text = CIFAR100_FINE_LABELS[label_id]
        # label_text = CIFAR100_FINE_LABELS[item["fine_label"]]
        repeated_label = " ".join([label_text] * 3)
        # label_description = CIFAR100_DESCRIPTIONS[label_id]
        samples.append({
            "image": item["img"],
            "label": item["fine_label"],
            "label_text": repeated_label,  
            # "label_text": label_text,
            # "label_description": label_description,
            "id": str(idx)
        })
    
    return samples

def load_seam_dataset(json_path, num_samples=200):
    with open(json_path) as f:
        dataset = [json.loads(line) for line in f]
    
    if len(dataset) > num_samples:
        dataset = dataset[:num_samples]
    
    return dataset

def extract_embeddings(model, processor, hook_manager, dataset, args):
    main_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Only using final_norm as the target layer
    target_layer = "final_norm"
    
    embeddings = {
        'Language': {target_layer: []},
        'Vision': {target_layer: []},
        'Language+Vision': {target_layer: []}
    }
    
    labels = []
    
    for sample_idx, sample in enumerate(tqdm(dataset, desc=f"Processing {args.task_name} samples")):
        hook_manager.clear_embeddings()
        
        if args.task_name == "cifar100":
            sample_id = sample["id"]
            text = sample["label_text"]  # Using the repeated label text
            # text = sample["label_description"]
            image = sample["image"]
            label = sample["label"]
            labels.append(label)
        else:  # SEAM dataset
            sample_id = sample.get("fen", str(sample_idx))
            text = sample_id
            image_path = get_image_path(args, sample_id, sample_idx)
            image = Image.open(image_path).convert('RGB')
            label = sample_idx
            labels.append(label)
        
        try:
            # Process LANGUAGE-ONLY input
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            text_input = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            text_inputs = processor(
                text=[text_input],
                return_tensors="pt"
            ).to(main_device)
            
            with torch.no_grad():
                # Text forward pass
                _ = model.generate(**text_inputs, max_new_tokens=1)
                
                # Save text embeddings for target layer
                if hook_manager.hook_states[target_layer].hidden_states:
                    text_emb = hook_manager.hook_states[target_layer].hidden_states[0]
                    embeddings['Language'][target_layer].append(text_emb.mean(dim=1).cpu().numpy())
            
            hook_manager.clear_embeddings()
            
            # Process VISION-ONLY input
            if args.task_name == "cifar100":
                vision_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": ""}
                        ]
                    }
                ]
                
                vision_text_input = processor.apply_chat_template(
                    vision_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                vision_inputs = processor(
                    images=image,
                    text=vision_text_input,
                    return_tensors="pt",
                    padding=True
                ).to(main_device)
            else:
                vision_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": ""}
                        ]
                    }
                ]
                
                vision_text_input = processor.apply_chat_template(
                    vision_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(vision_messages)
                
                vision_inputs = processor(
                    text=[vision_text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(main_device)
            
            with torch.no_grad():
                _ = model.generate(**vision_inputs, max_new_tokens=1)
                
                if hook_manager.hook_states[target_layer].hidden_states:
                    vision_emb = hook_manager.hook_states[target_layer].hidden_states[0]
                    embeddings['Vision'][target_layer].append(vision_emb.mean(dim=1).cpu().numpy())
            
            hook_manager.clear_embeddings()

            # Process MULTIMODAL (LANGUAGE+VISION) input
            if args.task_name == "cifar100":
                mixed_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text}
                        ]
                    }
                ]
                
                mixed_text_input = processor.apply_chat_template(
                    mixed_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                mixed_inputs = processor(
                    images=image,
                    text=mixed_text_input,
                    return_tensors="pt",
                    padding=True
                ).to(main_device)
            else:
                # For SEAM dataset, process with file path
                mixed_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": text}
                        ]
                    }
                ]
                
                mixed_text_input = processor.apply_chat_template(
                    mixed_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(mixed_messages)
                
                mixed_inputs = processor(
                    text=[mixed_text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(main_device)
            
            with torch.no_grad():
                _ = model.generate(**mixed_inputs, max_new_tokens=1)

                if hook_manager.hook_states[target_layer].hidden_states:
                    mixed_emb = hook_manager.hook_states[target_layer].hidden_states[0]
                    embeddings['Language+Vision'][target_layer].append(mixed_emb.mean(dim=1).cpu().numpy())
        
        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            continue
    
    # Convert lists to numpy arrays
    for modality in embeddings:
        for layer_idx in embeddings[modality]:
            if embeddings[modality][layer_idx]:
                embeddings[modality][layer_idx] = np.vstack(embeddings[modality][layer_idx])
    
    return embeddings, labels

def create_combined_tsne_plot(embeddings_7b, embeddings_72b, args):
    plt.figure(figsize=(18, 5))
    
    colors = {
        'Language': 'orange',
        'Vision': 'blue', 
        'Language+Vision': 'green'
    }
    
    gs = GridSpec(1, 4, figure=plt.gcf(), wspace=0.35)
    
    target_layer = "final_norm"
    
    dataset_title_fontsize = 26
    model_title_fontsize = 24
    dataset_subtitle_fontsize = 20
    legend_fontsize = 14
    
    all_handles = []
    all_labels = []
    
    # First subplot: Qwen 7B CIFAR-100
    ax = plt.subplot(gs[0, 0])
    
    combined_embeddings = []
    modality_markers = []
    
    for modality in ['Language', 'Vision', 'Language+Vision']:
        if target_layer in embeddings_7b['cifar'][modality] and len(embeddings_7b['cifar'][modality][target_layer]) > 0:
            embs = embeddings_7b['cifar'][modality][target_layer]
            if np.isnan(embs).any() or np.isinf(embs).any():
                embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
            combined_embeddings.append(embs)
            modality_markers.extend([modality] * len(embs))
    
    if combined_embeddings:
        try:
            combined_embeddings = np.vstack(combined_embeddings)
            
            if combined_embeddings.shape[1] > 100:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=100, random_state=42)
                combined_embeddings = pca.fit_transform(combined_embeddings)
            
            tsne = TSNE(n_components=2, random_state=42, 
                        perplexity=min(30, len(combined_embeddings)-1),
                        n_iter=1000)
            embeddings_2d = tsne.fit_transform(combined_embeddings)
            
            start_idx = 0
            for modality in ['Language', 'Vision', 'Language+Vision']:
                modality_count = modality_markers.count(modality)
                if modality_count > 0:
                    end_idx = start_idx + modality_count
                    scatter = ax.scatter(
                        embeddings_2d[start_idx:end_idx, 0],
                        embeddings_2d[start_idx:end_idx, 1],
                        c=colors[modality],
                        label=modality,
                        alpha=0.7,
                        s=50
                    )
                    all_handles.append(scatter)
                    all_labels.append(modality)
                    start_idx = end_idx
        except Exception as e:
            logger.error(f"Error creating T-SNE for 7B CIFAR-100: {e}")
            ax.text(0.5, 0.5, "Error in T-SNE", ha='center', va='center')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_title("CIFAR-100", fontsize=dataset_subtitle_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Second subplot: Qwen 7B SEAM
    ax = plt.subplot(gs[0, 1])
    
    combined_embeddings = []
    modality_markers = []
    
    for modality in ['Language', 'Vision', 'Language+Vision']:
        if target_layer in embeddings_7b['seam'][modality] and len(embeddings_7b['seam'][modality][target_layer]) > 0:
            embs = embeddings_7b['seam'][modality][target_layer]
            if np.isnan(embs).any() or np.isinf(embs).any():
                embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
            combined_embeddings.append(embs)
            modality_markers.extend([modality] * len(embs))
    
    if combined_embeddings:
        try:
            combined_embeddings = np.vstack(combined_embeddings)
            
            if combined_embeddings.shape[1] > 100:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=100, random_state=42)
                combined_embeddings = pca.fit_transform(combined_embeddings)
            
            tsne = TSNE(n_components=2, random_state=42, 
                        perplexity=min(30, len(combined_embeddings)-1),
                        n_iter=1000)
            embeddings_2d = tsne.fit_transform(combined_embeddings)
            
            start_idx = 0
            for modality in ['Language', 'Vision', 'Language+Vision']:
                modality_count = modality_markers.count(modality)
                if modality_count > 0:
                    end_idx = start_idx + modality_count
                    ax.scatter(
                        embeddings_2d[start_idx:end_idx, 0],
                        embeddings_2d[start_idx:end_idx, 1],
                        c=colors[modality],
                        alpha=0.7,
                        s=50
                    )
                    start_idx = end_idx
        except Exception as e:
            logger.error(f"Error creating T-SNE for 7B SEAM: {e}")
            ax.text(0.5, 0.5, "Error in T-SNE", ha='center', va='center')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_title("SEAM", fontsize=dataset_subtitle_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Third subplot: Qwen 72B CIFAR-100
    ax = plt.subplot(gs[0, 2])
    
    combined_embeddings = []
    modality_markers = []
    
    for modality in ['Language', 'Vision', 'Language+Vision']:
        if target_layer in embeddings_72b['cifar'][modality] and len(embeddings_72b['cifar'][modality][target_layer]) > 0:
            embs = embeddings_72b['cifar'][modality][target_layer]
            if np.isnan(embs).any() or np.isinf(embs).any():
                embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
            combined_embeddings.append(embs)
            modality_markers.extend([modality] * len(embs))
    
    if combined_embeddings:
        try:
            combined_embeddings = np.vstack(combined_embeddings)
            
            if combined_embeddings.shape[1] > 100:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=100, random_state=42)
                combined_embeddings = pca.fit_transform(combined_embeddings)
            
            tsne = TSNE(n_components=2, random_state=42, 
                        perplexity=min(30, len(combined_embeddings)-1),
                        n_iter=1000)
            embeddings_2d = tsne.fit_transform(combined_embeddings)
            
            start_idx = 0
            for modality in ['Language', 'Vision', 'Language+Vision']:
                modality_count = modality_markers.count(modality)
                if modality_count > 0:
                    end_idx = start_idx + modality_count
                    ax.scatter(
                        embeddings_2d[start_idx:end_idx, 0],
                        embeddings_2d[start_idx:end_idx, 1],
                        c=colors[modality],
                        alpha=0.7,
                        s=50
                    )
                    start_idx = end_idx
        except Exception as e:
            logger.error(f"Error creating T-SNE for 72B CIFAR-100: {e}")
            ax.text(0.5, 0.5, "Error in T-SNE", ha='center', va='center')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_title("CIFAR-100", fontsize=dataset_subtitle_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Fourth subplot: Qwen 72B SEAM
    ax = plt.subplot(gs[0, 3])
    
    combined_embeddings = []
    modality_markers = []
    
    for modality in ['Language', 'Vision', 'Language+Vision']:
        if target_layer in embeddings_72b['seam'][modality] and len(embeddings_72b['seam'][modality][target_layer]) > 0:
            embs = embeddings_72b['seam'][modality][target_layer]
            if np.isnan(embs).any() or np.isinf(embs).any():
                embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
            combined_embeddings.append(embs)
            modality_markers.extend([modality] * len(embs))
    
    if combined_embeddings:
        try:
            combined_embeddings = np.vstack(combined_embeddings)
            
            if combined_embeddings.shape[1] > 100:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=100, random_state=42)
                combined_embeddings = pca.fit_transform(combined_embeddings)
            
            tsne = TSNE(n_components=2, random_state=42, 
                        perplexity=min(30, len(combined_embeddings)-1),
                        n_iter=1000)
            embeddings_2d = tsne.fit_transform(combined_embeddings)
            
            start_idx = 0
            for modality in ['Language', 'Vision', 'Language+Vision']:
                modality_count = modality_markers.count(modality)
                if modality_count > 0:
                    end_idx = start_idx + modality_count
                    ax.scatter(
                        embeddings_2d[start_idx:end_idx, 0],
                        embeddings_2d[start_idx:end_idx, 1],
                        c=colors[modality],
                        alpha=0.7,
                        s=50
                    )
                    start_idx = end_idx
        except Exception as e:
            logger.error(f"Error creating T-SNE for 72B SEAM: {e}")
            ax.text(0.5, 0.5, "Error in T-SNE", ha='center', va='center')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_title("SEAM", fontsize=dataset_subtitle_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add model titles
    plt.figtext(0.25, 0.98, "Qwen2.5-VL-7B-Instruct", fontsize=model_title_fontsize, ha='center')
    plt.figtext(0.75, 0.98, "Qwen2.5-VL-72B-Instruct", fontsize=model_title_fontsize, ha='center')
    
    # Add legend
    if all_handles:
        # Only use the first three scatter objects for the legend to avoid duplicates
        unique_handles = all_handles[:3]
        unique_labels = all_labels[:3]
        
        legend = plt.figlegend(unique_handles, unique_labels, 
                             loc='upper left', 
                             bbox_to_anchor=(0.02, 0.98), 
                             fontsize=legend_fontsize, 
                             frameon=False)
        for handle in legend.legendHandles:
            handle.set_sizes([100])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_dir = os.path.join(args.output_dir, "combined_models")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tsne_7b_72b_comparison.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Combined T-SNE plot saved to {output_file}")
    
    return plt.gcf()

def create_multimodal_embeddings(args, model, processor, hook_manager, model_size):
    logger.info(f"Generating embeddings for {args.task_name} with {model_size} model")
    
    if args.task_name == "cifar100":
        dataset = load_cifar100_dataset(num_samples=args.cifar_samples)
    else:  # SEAM
        dataset = load_seam_dataset(args.seam_json, num_samples=args.seam_samples)
    
    logger.info(f"Loaded {len(dataset)} samples from {args.task_name}")
    
    embeddings, labels = extract_embeddings(model, processor, hook_manager, dataset, args)
    
    return embeddings, labels

def main():
    parser = argparse.ArgumentParser(description='Generate T-SNE plots of multimodal embeddings')
    parser.add_argument('--output_dir', type=str, default="../../outputs/tsne_plots",
                        help='Output directory for plots and embeddings')
    parser.add_argument('--seam_json', type=str, default="../../data/benchmark/legal.jsonl",
                        help='Path to SEAM dataset JSON file')
    parser.add_argument('--seam_image_folder', type=str, default="../../data/benchmark/legal/",
                        help='Path to SEAM dataset image folder')
    parser.add_argument('--cifar_samples', type=int, default=200,
                        help='Number of CIFAR-100 samples to use')
    parser.add_argument('--seam_samples', type=int, default=200,
                        help='Number of SEAM dataset samples to use')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Save the extracted embeddings for later analysis')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    all_embeddings = {
        '7b': {'cifar': None, 'seam': None},
        '72b': {'cifar': None, 'seam': None}
    }
    
    model_7b_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    logger.info(f"Loading model {model_7b_name}")
    model_7b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_7b_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    processor_7b = AutoProcessor.from_pretrained(
        model_7b_name, 
        trust_remote_code=True
    )
    
    hook_manager_7b = QwenLayerHookManager(model_7b).register_hooks()

    try:
        logger.info("Processing CIFAR-100 dataset with 7B model")
        args.task_name = "cifar100"
        args.image_folder = None
        cifar_embeddings_7b, cifar_labels_7b = create_multimodal_embeddings(args, model_7b, processor_7b, hook_manager_7b, "7B")
        all_embeddings['7b']['cifar'] = cifar_embeddings_7b
        
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error processing CIFAR-100 dataset with 7B model: {e}")
    
    try:
        logger.info(f"Processing SEAM dataset from {args.seam_json} with 7B model")
        args.task_name = "seam"
        args.image_folder = args.seam_image_folder
        seam_embeddings_7b, seam_labels_7b = create_multimodal_embeddings(args, model_7b, processor_7b, hook_manager_7b, "7B")
        all_embeddings['7b']['seam'] = seam_embeddings_7b
        
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error processing SEAM dataset with 7B model: {e}")
    
    hook_manager_7b.remove_hooks()
    del model_7b
    del processor_7b
    del hook_manager_7b
    gc.collect()
    torch.cuda.empty_cache()
    
    model_72b_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    logger.info(f"Loading model {model_72b_name}")
    model_72b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_72b_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    processor_72b = AutoProcessor.from_pretrained(
        model_72b_name, 
        trust_remote_code=True
    )
    
    hook_manager_72b = QwenLayerHookManager(model_72b).register_hooks()
    
    try:
        logger.info("Processing CIFAR-100 dataset with 72B model")
        args.task_name = "cifar100"
        args.image_folder = None
        cifar_embeddings_72b, cifar_labels_72b = create_multimodal_embeddings(args, model_72b, processor_72b, hook_manager_72b, "72B")
        all_embeddings['72b']['cifar'] = cifar_embeddings_72b
        
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error processing CIFAR-100 dataset with 72B model: {e}")
    
    try:
        logger.info(f"Processing SEAM dataset from {args.seam_json} with 72B model")
        args.task_name = "seam"
        args.image_folder = args.seam_image_folder
        seam_embeddings_72b, seam_labels_72b = create_multimodal_embeddings(args, model_72b, processor_72b, hook_manager_72b, "72B")
        all_embeddings['72b']['seam'] = seam_embeddings_72b
        
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error processing SEAM dataset with 72B model: {e}")

    hook_manager_72b.remove_hooks()
    
    try:
        if all_embeddings['7b']['cifar'] is not None and all_embeddings['7b']['seam'] is not None and \
           all_embeddings['72b']['cifar'] is not None and all_embeddings['72b']['seam'] is not None:
            logger.info("Creating combined T-SNE visualization for both models")
            create_combined_tsne_plot(
                embeddings_7b={'cifar': all_embeddings['7b']['cifar'], 'seam': all_embeddings['7b']['seam']},
                embeddings_72b={'cifar': all_embeddings['72b']['cifar'], 'seam': all_embeddings['72b']['seam']},
                args=args
            )
        else:
            logger.warning("Could not create combined plot, some embeddings are missing")
    except Exception as e:
        logger.error(f"Error creating combined visualization: {e}")
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()