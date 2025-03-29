# SEAM: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models

This repository contains the code for the SEAM (Semantically Equivalent Across Modalities) benchmark.

## Overview

SEAM addresses fundamental limitations in existing benchmarks through its utilization of distinct notation systems and preservation of semantic equivalence across modalities. By leveraging domain-specific standardized representations in:

- **Chess**: Board images vs. FEN strings
- **Chemistry**: Structural diagrams vs. SMILES strings
- **Music**: Staff images vs. ABC notations
- **Graph Theory**: Node-edge diagrams vs. adjacency matrices

SEAM presents both visual-spatial and textual-symbolic representations while maintaining semantic equivalence. The benchmark comprises 16 carefully calibrated tasks designed to be self-contained in both modalities with 3,200 four-way multiple-choice questions in total, enabling effective quantification of modality imbalance in VLMs.

## Dataset Generation

To generate the SEAM benchmark dataset manually, run the following scripts:

```bash
# Generate Chemistry tasks
python dataset_chem.py

# Generate Chess tasks
python dataset_chess.py

# Generate Graph Theory tasks
python dataset_graph.py

# Generate Music tasks
python dataset_music.py
```

Each script will generate task-specific data, images, and question files, which will be stored in the `data/benchmark/` directory. You can also directly download the pre-generated dataset from [this link](https://drive.google.com/drive/folders/12vruRWA56Sl4joIDH7uXF8QRmUcUoKwn?usp=sharing) and unzip it under that directory.

## Model Evaluation

### Open-Source Models

To evaluate open-source models on the SEAM benchmark, use the `main.py` script:

```bash
python main.py --model_name "Qwen2.5-VL-7B-Instruct" --tasks "bench" --mode "v" --max_num_seqs 64 --num_gpus 4
```

Parameters:
- `model_name`: Name of the model to evaluate (e.g., "Qwen2.5-VL-7B-Instruct", "Llama-3.2-11B-Vision-Instruct")
- `tasks`: Task set to evaluate on (e.g., "bench" for all tasks, or specific tasks like "fork", "carbon")
- `mode`: Input modality ("v" for vision-only, "l" for language-only, "vl" for vision+language)
- `max_num_seqs`: Maximum number of sequences to process in parallel
- `num_gpus`: Number of GPUs to use for inference

You can also use the provided shell script to evaluate multiple models:

```bash
bash run.sh
```

### Proprietary Models

For evaluating proprietary models like GPT-4o or Claude, we provide batch inference scripts:

**OpenAI models:**

```bash
bash run_openai.sh
```

**Claude models:**

```bash
python claude_batch.py --model claude-3-7-sonnet-20250219 --mode l
```

## Embedding Similarity Analysis

To analyze internal representations and embedding similarities across modalities:

```bash
# Extract and compare layer-wise embeddings
python embeddings/embed_sim.py --model_name "Qwen2.5-VL-7B-Instruct" --tasks "fork legal puzzle eval"

# Generate t-SNE visualizations for SEAM vs. CIFAR-100
python embeddings/tsne_embeddings.py
```

This analysis helps visualize how models process semantically equivalent information across different modalities.

<!-- ## Citation

If you use this benchmark in your research, please cite:

```
@inproceedings{seam2024,
  title={SEAM: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models},
  author={...},
  booktitle={...},
  year={2025}
}
``` -->

<!-- ## License -->