modes=(v l vl)

# Options for tasks:
# bench, variant
# chess, chem, music, graph
# chess_res, chess_bw, chem_rot

# fork, legal, puzzle, eval
# carbon, hydrogen, weight, caption
# notes, measures, forms, rhythm
# path_counting, path_existence, shortest_path, bfs_traversal

models_small=(
    "llava-onevision-qwen2-7b-ov-hf"
    "pixtral-12b"
    "Qwen2.5-VL-7B-Instruct"
    "InternVL2_5-8B"
    "Llama-3.2-11B-Vision-Instruct"
    "gemma-3-12b-it"
    "gemma-3-27b-it"
)

models_large=(
    "llava-onevision-qwen2-72b-ov-hf"
    "Qwen2.5-VL-72B-Instruct"
    "Llama-3.2-90B-Vision-Instruct"
    "InternVL2_5-78B"
)


for model in "${models_large[@]}"
do
    for mode in "${modes[@]}"
    do
        python main.py --model_name "$model" \
        --tasks "bench" --mode "$mode" --max_num_seqs 64 --num_gpus 8
    done
done


for model in "${models_small[@]}" 
do
    for mode in "${modes[@]}"
    do
        python main.py --model_name "$model" \
        --tasks "bench" --mode "$mode" --max_num_seqs 64 --num_gpus 4
    done
done


# python main.py --model_name "Qwen2.5-VL-7B-Instruct" \
# --tasks "variant" --mode "vl" --max_num_seqs 64 --num_gpus 4

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --model_name "Qwen2.5-VL-7B-Instruct" \
--tasks "fork" --mode "vl" --max_num_seqs 64 --num_gpus 4



