import os
import pdb
from PIL import Image
import hashlib
import base64
from io import BytesIO


def base64_encoding(pil_image):
    # Convert PIL Image to bytes using BytesIO buffer
    buffered = BytesIO()
    # Save image to buffer in PNG format
    pil_image.save(buffered, format="PNG")
    # Get the byte value from the buffer
    img_bytes = buffered.getvalue()
    # Encode to base64 and convert to string
    return base64.b64encode(img_bytes).decode("utf-8")

def read_image_chess(cfg, fen, directory):
    
    img = Image.open(os.path.join(cfg.benchmark_root, directory, fen.split(" ")[0].replace("/", "_") + ".png")).convert("RGB")
    
    return img

def read_image_chem(cfg, smiles, directory):
    
    img = Image.open(os.path.join(cfg.benchmark_root, directory, hash_str(smiles) + ".png")).convert("RGB")
    
    return img

def read_image_music(cfg, index, directory):

    img = Image.open(os.path.join(cfg.benchmark_root, directory, f"{index}.png")).convert("RGB")
    
    return img

def read_image_graph(cfg, idx, directory):

    img = Image.open(os.path.join(cfg.benchmark_root, directory, f"{idx}.png")).convert("RGB")

    return img

def hash_str(filename):
    
    return hashlib.md5(filename.encode()).hexdigest()

def vllm_format(prompts, images, cfg):
    
    if cfg.mode == "v" or cfg.mode == "vl":
        return [{"prompt": prompt, "multi_modal_data": {"image": image}} for prompt, image in zip(prompts, images)]
    
    elif cfg.mode == "l":
        return [{"prompt": prompt} for prompt in prompts]
    
    else:
        raise ValueError("Invalid mode")


def parse_outputs(outputs):
    
    ret = []
    for output in outputs:
        generated_text = output.outputs[0].text
        ret.append(generated_text)
    
    return ret


def show_results_exact_match(results):
    
    correct = 0
    wrong = 0
    invalid = 0
    
    for result in results:
        if result == 0:
            correct += 1
        elif result == 1:
            wrong += 1
        elif result == 2:
            invalid += 1
    
    correct = round(correct / len(results), 3)
    wrong = round(wrong / len(results), 3)
    invalid_rate = round(invalid / len(results), 3)
    
    print(f"Correct: {correct}, Wrong: {wrong}, Invalid Rate: {invalid_rate}, Total: {len(results)}", flush=True)
