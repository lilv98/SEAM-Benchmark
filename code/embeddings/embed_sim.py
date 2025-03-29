import os
import json
import torch
import numpy as np
import hashlib
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def hash_str(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_image_path(args, sample_id, index=None):
    if args.task_name in ["pin", "fork", "legal", "puzzle", "eval"] or "chess" in args.task_name:
        return os.path.join(args.image_folder, sample_id.split(" ")[0].replace("/", "_") + ".png")
    elif args.task_name in ["carbon", "hydrogen", "weight", "react", "caption"] or "chem" in args.task_name:
        return os.path.join(args.image_folder, hash_str(sample_id) + ".png")
    elif args.task_name in ["notes", "measures", "predict", "forms", "rhythm"] or "music" in args.task_name:
        idx = index if index is not None else len(args.processed_ids)
        return os.path.join(args.image_folder, f"{idx}.png")
    elif args.task_name in ["cycle_detection", "path_counting", "path_existence", "shortest_path", "bfs_traversal"] or "graph" in args.task_name:
        idx = index if index is not None else len(args.processed_ids)
        return os.path.join(args.image_folder, f"{idx}.png")
    else:
        return os.path.join(args.image_folder, f"{sample_id}.png")

class LayerHookState:
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.hidden_states = []
        
    def update(self, hidden_states):
        self.hidden_states.append(hidden_states.detach().cpu().float())
        
    def get(self):
        if not self.hidden_states:
            raise ValueError(f"No hidden states captured for layer {self.layer_idx}")
        return self.hidden_states
    
    def clear(self):
        self.hidden_states = []

class QwenLayerHookManager:
    def __init__(self, model):
        self.model = model
        self.layer_hooks = []
        self.hook_states = {}
        self.num_layers = len(model.model.layers)
        
        for i in range(self.num_layers):
            self.hook_states[i] = LayerHookState(i)
        
        self.hook_states["final_norm"] = LayerHookState("final_norm")
        
    def register_hooks(self):
        for i, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(self._create_hook_fn(i))
            self.layer_hooks.append(hook)
        
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

class LlamaLayerHookManager:
    def __init__(self, model):
        self.model = model
        self.layer_hooks = []
        self.hook_states = {}

        self.num_layers = len(model.language_model.model.layers)
        
        for i in range(self.num_layers):
            self.hook_states[i] = LayerHookState(i)
        
        self.hook_states["final_norm"] = LayerHookState("final_norm")
        
    def register_hooks(self):
        for i, layer in enumerate(self.model.language_model.model.layers):
            hook = layer.register_forward_hook(self._create_hook_fn(i))
            self.layer_hooks.append(hook)

        hook = self.model.language_model.model.norm.register_forward_hook(self._create_hook_fn("final_norm"))
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

class InternVLLayerHookManager:
    def __init__(self, model):
        self.model = model
        self.layer_hooks = []
        self.hook_states = {}
        
        # Encoder layers (transformer decoder)
        self.num_layers = len(model.language_model.model.layers)
        
        for i in range(self.num_layers):
            self.hook_states[i] = LayerHookState(i)
        
        # Final norm
        self.hook_states["final_norm"] = LayerHookState("final_norm")
        
    def register_hooks(self):
        # Register hooks for each language model decoder layer
        for i, layer in enumerate(self.model.language_model.model.layers):
            hook = layer.register_forward_hook(self._create_hook_fn(i))
            self.layer_hooks.append(hook)
        
        # Register hook for the final norm layer
        hook = self.model.language_model.model.norm.register_forward_hook(self._create_hook_fn("final_norm"))
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

class ModelProcessor:
    def __init__(self, args):
        self.args = args
        
    def setup_model_and_tokenizer(self):
        raise NotImplementedError("Each model processor must implement this method")
        
    def process_modalities_layerwise(self, model, tokenizer, text, image_path):
        raise NotImplementedError("Each model processor must implement this method")

class InternVLEmbeddingProcessor(ModelProcessor):
    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
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

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def setup_model_and_tokenizer(self):
        logger.info(f"Loading model {self.args.model_name}")
        
        self.main_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        model = AutoModel.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            torch_dtype=self.args.dtype,
            device_map="auto",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name, 
            trust_remote_code=True
        )
        
        # Initialize layer-wise hook manager
        self.layer_hook_manager = InternVLLayerHookManager(model).register_hooks()
        
        return model, tokenizer
        
    def process_modalities_layerwise(self, model, tokenizer, text, image_path):
        self.layer_hook_manager.clear_embeddings()
        
        # Text processing
        prompt = text
        # prompt = "chess board with FEN: " + text

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.main_device)
        
        # Vision processing
        pixel_values = self.load_image(image_path).to(self.args.dtype).to(self.main_device)
        vision_prompt = "<image>\n"
        
        with torch.no_grad():
            try:
                # Text forward pass
                response_text = model.chat(
                    tokenizer, 
                    None, 
                    prompt, 
                    {"max_new_tokens": 1}, 
                    history=None, 
                    return_history=False
                )
                
                # Save text embeddings
                text_layer_embeddings = {}
                for layer_idx, hook_state in self.layer_hook_manager.hook_states.items():
                    if hook_state.hidden_states:
                        text_emb = hook_state.hidden_states[0]
                        text_layer_embeddings[layer_idx] = text_emb.mean(dim=1)
                
                # Clear embeddings before vision pass
                self.layer_hook_manager.clear_embeddings()
                
                # Vision forward pass
                response_vision = model.chat(
                    tokenizer,
                    pixel_values,
                    vision_prompt,
                    {"max_new_tokens": 1}
                )
                
                # Calculate similarities for each layer
                similarities = {}
                
                # For each layer, get the text and vision embeddings and calculate similarity
                for layer_idx, hook_state in self.layer_hook_manager.hook_states.items():
                    if hook_state.hidden_states and layer_idx in text_layer_embeddings:
                        vision_emb = hook_state.hidden_states[0].mean(dim=1)
                        text_emb = text_layer_embeddings[layer_idx]
                        
                        sim = cosine_similarity(
                            text_emb.numpy(), 
                            vision_emb.numpy()
                        )[0][0]
                        
                        similarities[str(layer_idx)] = float(sim)
                
                return similarities
                
            except Exception as e:
                logger.error(f"Error in processing modalities for InternVL: {e}")
                raise

class LlamaVisionProcessor(ModelProcessor):
    def setup_model_and_tokenizer(self):
        logger.info(f"Loading model {self.args.model_name}")
        
        self.main_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Import specific Llama model class
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        
        # Load the model
        model = MllamaForConditionalGeneration.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            torch_dtype=self.args.dtype,
            device_map="auto",
        )
        
        # Load the processor (handles both tokenization and image processing)
        processor = AutoProcessor.from_pretrained(
            self.args.model_name,
            trust_remote_code=True
        )
        
        # Initialize layer-wise hook manager
        self.layer_hook_manager = LlamaLayerHookManager(model).register_hooks()
        
        return model, processor
    
    def process_modalities_layerwise(self, model, processor, text, image_path):
        self.layer_hook_manager.clear_embeddings()
        
        # Prepare text-only input
        text_messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        text_input = processor.apply_chat_template(
            text_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        text_inputs = processor(
            text=[text_input],
            return_tensors="pt"
        ).to(self.main_device)
        
        image = Image.open(image_path).convert('RGB')
        vision_messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ""}
                ]
            }
        ]
        
        vision_text_input = processor.apply_chat_template(
            vision_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process the inputs using the processor
        vision_inputs = processor(
            image,
            text=vision_text_input,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.main_device)
        
        generation_config = dict(max_new_tokens=1)
        
        with torch.no_grad():
            try:
                text_outputs = model.generate(**text_inputs, **generation_config)

                text_layer_embeddings = {}
                for layer_idx, hook_state in self.layer_hook_manager.hook_states.items():
                    if hook_state.hidden_states:
                        text_emb = hook_state.hidden_states[0]
                        text_layer_embeddings[layer_idx] = text_emb.mean(dim=1)

                self.layer_hook_manager.clear_embeddings()

                vision_outputs = model.generate(**vision_inputs, **generation_config)

                similarities = {}
                
                for layer_idx, hook_state in self.layer_hook_manager.hook_states.items():
                    if hook_state.hidden_states and layer_idx in text_layer_embeddings:
                        vision_emb = hook_state.hidden_states[0].mean(dim=1)
                        text_emb = text_layer_embeddings[layer_idx]
                        
                        sim = cosine_similarity(
                            text_emb.numpy(), 
                            vision_emb.numpy()
                        )[0][0]
                        
                        similarities[str(layer_idx)] = float(sim)
                
                return similarities
                
            except Exception as e:
                logger.error(f"Error in processing modalities for Llama: {e}")
                raise

class QwenVLProcessor(ModelProcessor):
    def setup_model_and_tokenizer(self):
        logger.info(f"Loading model {self.args.model_name} with device_map='auto'")
        
        self.main_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.main_device} as the main device for inputs")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            torch_dtype=self.args.dtype,
            device_map="auto",
        )
        
        processor = AutoProcessor.from_pretrained(
            self.args.model_name, 
            trust_remote_code=True
        )
        
        # Initialize layer-wise hook manager
        self.layer_hook_manager = QwenLayerHookManager(model).register_hooks()
        
        return model, processor
            
    def process_modalities_layerwise(self, model, processor, text, image_path):
        self.layer_hook_manager.clear_embeddings()
        
        # Text processing
        # text = "Hello World!"
        # text = "chess board with FEN: " + text
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        
        text_input = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        text_inputs = processor(
            text=[text_input],
            return_tensors="pt"
        ).to(self.main_device)
        
        # Vision processing
        vision_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": ""}
                ]
            }
        ]
        
        try:
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
            ).to(self.main_device)
            
            generation_config = dict(max_new_tokens=1)
            
            with torch.no_grad():
                # Text forward pass
                text_outputs = model.generate(**text_inputs, **generation_config)
                
                # Save text embeddings
                text_layer_embeddings = {}
                for layer_idx, hook_state in self.layer_hook_manager.hook_states.items():
                    if hook_state.hidden_states:
                        text_emb = hook_state.hidden_states[0]
                        text_layer_embeddings[layer_idx] = text_emb.mean(dim=1)
                
                # Clear embeddings before vision pass
                self.layer_hook_manager.clear_embeddings()
                
                # Vision forward pass
                vision_outputs = model.generate(**vision_inputs, **generation_config)
                
                # Calculate similarities for each layer
                similarities = {}
                
                for layer_idx, hook_state in self.layer_hook_manager.hook_states.items():
                    if hook_state.hidden_states and layer_idx in text_layer_embeddings:
                        vision_emb = hook_state.hidden_states[0].mean(dim=1)
                        text_emb = text_layer_embeddings[layer_idx]
                        
                        sim = cosine_similarity(
                            text_emb.numpy(), 
                            vision_emb.numpy()
                        )[0][0]
                        
                        similarities[str(layer_idx)] = float(sim)
                
                return similarities
                
        except Exception as e:
            logger.error(f"Error in processing modalities for Qwen: {e}")
            raise

def get_processor_for_model(args):
    if "InternVL" in args.model_name:
        return InternVLEmbeddingProcessor(args)
    elif "Qwen" in args.model_name or "qwen" in args.model_name:
        return QwenVLProcessor(args)
    elif "Llama" in args.model_name or "llama" in args.model_name:
        return LlamaVisionProcessor(args)
    else:
        logger.error(f"Model architecture {args.model_name} not currently supported")
        raise NotImplementedError(f"Model {args.model_name} not supported yet")

def load_dataset(json_path):
    with open(json_path) as f:
        return [json.loads(line) for line in f]

def process_samples(dataset, args, model, tokenizer_or_processor, processor):
    results = []
    errors = []
    layerwise_results = []
    
    args.processed_ids = []
    
    if args.max_samples and args.max_samples > 0:
        dataset = dataset[:args.max_samples]
        logger.info(f"Processing limited set of {len(dataset)} samples")
    
    for sample_idx, sample in enumerate(tqdm(dataset, desc=f"Processing {args.task_name}")):
        sample_id = sample.get("fen", sample.get("smiles", sample.get("abc_notation", sample.get("matrix", str(sample_idx)))))
        image_path = get_image_path(args, sample_id, sample_idx)
        
        try:
            layer_similarities = processor.process_modalities_layerwise(model, tokenizer_or_processor, sample_id, image_path)
            
            if layer_similarities:
                # Calculate average similarity across all layers
                avg_similarity = sum(layer_similarities.values()) / len(layer_similarities)
                logger.info(f"  Average similarity across all layers: {avg_similarity:.6f}")
                
                # Store the results
                layerwise_result = {
                    "id": sample_id,
                    "similarities": layer_similarities,
                    "average_similarity": float(avg_similarity)
                }
                layerwise_results.append(layerwise_result)
                args.processed_ids.append(sample_id)
                
                # Add to overall results
                result = {
                    "id": sample_id,
                    "average_similarity": float(avg_similarity)
                }
                results.append(result)
            
        except Exception as e:
            errors.append({"id": sample_id, "error": str(e)})
            logger.error(f"Error processing sample {sample_id}: {e}")
            continue
            
    return results, layerwise_results, errors

def calculate_layerwise_averages(layerwise_results):
    if not layerwise_results:
        return {}
    
    all_layers = set()
    for result in layerwise_results:
        all_layers.update(result["similarities"].keys())
    
    avg_similarities = {}
    for layer_key in all_layers:
        layer_sims = []
        for result in layerwise_results:
            if layer_key in result["similarities"]:
                layer_sims.append(result["similarities"][layer_key])
        
        if layer_sims:
            avg_similarities[layer_key] = sum(layer_sims) / len(layer_sims)
    
    return avg_similarities

def save_results(results, layerwise_results, errors, args):
    model_name = args.model_name.split('/')[-1]
    output_dir = os.path.join(args.output_root, model_name, args.task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall results
    results_file = os.path.join(output_dir, "overall_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": results,
            "errors": errors
        }, f, indent=2)
    
    # Save ID mapping
    id_mapping_file = os.path.join(output_dir, "id_mapping.json")
    with open(id_mapping_file, "w") as f:
        json.dump(args.processed_ids, f, indent=2)
    
    # Calculate and save layer-wise averages
    avg_layer_similarities = calculate_layerwise_averages(layerwise_results)
    
    # Save layer-wise results
    layer_results_file = os.path.join(output_dir, "layer_similarities.json")
    with open(layer_results_file, "w") as f:
        json.dump({
            "results": layerwise_results,
            "averages": avg_layer_similarities
        }, f, indent=2)
    
    # Print average layer-wise similarities
    logger.info("Average layer-wise similarities across all samples:")
    for layer, avg_sim in sorted(avg_layer_similarities.items(), key=lambda x: (
        0 if x[0].isdigit() else 1,
        int(x[0]) if x[0].isdigit() else float('inf'),
        x[0]
    )):
        layer_display = f"Layer {layer}" if layer.isdigit() else layer
        logger.info(f"  {layer_display}: {avg_sim:.6f}")
    
    # Overall average across all layers and samples
    overall_avg = sum(avg_layer_similarities.values()) / len(avg_layer_similarities)
    logger.info(f"Overall average similarity across all layers and samples: {overall_avg:.6f}")
    
    logger.info(f"Results saved to {output_dir}")

def save_aggregated_results(results, layerwise_results, errors, args):
    model_name = args.model_name.split('/')[-1]
    output_dir = os.path.join(args.output_root, model_name, "seam")
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "overall_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": results,
            "errors": errors
        }, f, indent=2)

    avg_layer_similarities = calculate_layerwise_averages(layerwise_results)

    task_averages = {}
    for task in args.tasks:
        task_results = [r for r in layerwise_results if r.get("task") == task]
        if task_results:
            task_averages[task] = calculate_layerwise_averages(task_results)
    
    layer_results_file = os.path.join(output_dir, "layer_similarities.json")
    with open(layer_results_file, "w") as f:
        json.dump({
            "results": layerwise_results,
            "averages": avg_layer_similarities,
            "task_averages": task_averages
        }, f, indent=2)
    
    logger.info("Average layer-wise similarities across all samples and tasks:")
    for layer, avg_sim in sorted(avg_layer_similarities.items(), key=lambda x: (
        0 if x[0].isdigit() else 1,
        int(x[0]) if x[0].isdigit() else float('inf'),
        x[0]
    )):
        layer_display = f"Layer {layer}" if layer.isdigit() else layer
        logger.info(f"  {layer_display}: {avg_sim:.6f}")
    
    overall_avg = sum(avg_layer_similarities.values()) / len(avg_layer_similarities)
    logger.info(f"Overall average similarity across all layers, samples, and tasks: {overall_avg:.6f}")
    
    task_overall_avgs = {}
    for task in args.tasks:
        task_results = [r for r in results if r.get("task") == task]
        if task_results:
            task_avg = sum(r["average_similarity"] for r in task_results) / len(task_results)
            task_overall_avgs[task] = float(task_avg)
            logger.info(f"Task {task} average similarity: {task_avg:.6f}")

    task_summary_file = os.path.join(output_dir, "task_summary.json")
    with open(task_summary_file, "w") as f:
        json.dump({
            "overall_average": float(overall_avg),
            "task_averages": task_overall_avgs,
            "sample_counts": {task: len([r for r in results if r.get("task") == task]) for task in args.tasks}
        }, f, indent=2)
    
    logger.info(f"Aggregated results saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Layer-wise embedding similarity analysis')
    # parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    # parser.add_argument('--model_name', type=str, default="OpenGVLab/InternVL2_5-78B")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--tasks', nargs='+', default=["fork", "legal", "puzzle", "eval", 
                                                  "carbon", "hydrogen", "weight", "caption",
                                                  "notes", "measures", "rhythm", "forms",
                                                  "bfs_traversal", "path_counting", "path_existence", "shortest_path"], 
                      help='Tasks to process')
    # parser.add_argument('--tasks', nargs='+', default=["fork"])
    parser.add_argument('--samples_per_task', type=int, default=200, help='Number of samples per task')
    parser.add_argument('--output_root', type=str, default="../../outputs/layer_embeddings")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        logger.info(f"Found {gpu_count} GPUs: {gpu_names}")
    else:
        logger.warning("No GPUs available, running on CPU")
    
    args.dtype = torch.bfloat16
    args.processed_ids = []
    
    return args

def main():
    args = parse_args()
    logger.info(f"Processing {len(args.tasks)} tasks with model {args.model_name}")
    
    all_results = []
    all_layerwise_results = []
    all_errors = []
    
    try:
        processor = get_processor_for_model(args)
        model, tokenizer_or_processor = processor.setup_model_and_tokenizer()

        for task_idx, task_name in enumerate(args.tasks):
            logger.info(f"Processing task {task_idx+1}/{len(args.tasks)}: {task_name}")
            
            args.task_name = task_name
            args.input_json = f"../../data/benchmark/{task_name}.jsonl"
            args.image_folder = f"../../data/benchmark/{task_name}/"
            args.max_samples = args.samples_per_task
            args.processed_ids = []
            
            try:
                dataset = load_dataset(args.input_json)
                
                if args.max_samples and args.max_samples > 0:
                    dataset = dataset[:args.max_samples]
                
                logger.info(f"Loaded {len(dataset)} samples for task {task_name}")
                
                results, layerwise_results, errors = process_samples(
                    dataset, args, model, tokenizer_or_processor, processor
                )
                
                if results:
                    logger.info(f"Task {task_name}: Processed {len(results)} samples successfully. Failed: {len(errors)}")
                    
                    # Add task information to each result
                    for result in results:
                        result["task"] = task_name
                    for result in layerwise_results:
                        result["task"] = task_name
                    for error in errors:
                        error["task"] = task_name

                    all_results.extend(results)
                    all_layerwise_results.extend(layerwise_results)
                    all_errors.extend(errors)
                else:
                    logger.warning(f"No valid results for task {task_name}")
                
            except Exception as e:
                logger.error(f"Error processing task {task_name}: {e}", exc_info=True)
                continue

        if all_results:
            save_aggregated_results(all_results, all_layerwise_results, all_errors, args)
        else:
            logger.warning("No valid results to analyze across all tasks")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()