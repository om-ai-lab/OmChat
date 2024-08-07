#!/bin/bash

MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_general_full_en_n104/"
MODEL_ID="n104"
echo $MODEL_PATH
echo $MODEL_ID
# without video
python eval_seed_bench_qllama.py \
    --model-path  $MODEL_PATH \
    --question-file /data3/ljj/proj/MLLM_evals/data/seed_bench/llava-seed-bench_without_video.jsonl \
    --image-folder /data3/ljj/proj/MLLM_evals/data/seed_bench \
    --answers-file /data3/ljj/proj/MLLM_evals/outputs/seed_bench/results-$MODEL_ID.jsonl \
    --conv-mode vicuna_v1 \
    --temperature 0 

## for llava-1.5 eval
# python eval_seed_llava.py \
#     --model-path  $MODEL_PATH \
#     --question-file /data3/ljj/proj/MLLM_evals/data/seed_bench/llava-seed-bench_without_video.jsonl \
#     --image-folder /data3/ljj/proj/MLLM_evals/data/seed_bench \
#     --answers-file /data3/ljj/proj/MLLM_evals/outputs/seed_bench/results-$MODEL_ID.jsonl \
#     --conv-mode vicuna_v1 \
#     --temperature 0 

python get_seed_bench_score.py \
    --annotation-file /data3/ljj/proj/MLLM_evals/data/seed_bench/SEED-Bench.json \
    --result-file /data3/ljj/proj/MLLM_evals/outputs/seed_bench/results-$MODEL_ID.jsonl

# with video 
# python eval_seed_bench.py \
#     --model-path  $MODEL_PATH \
#     --question-file /data3/ljj/proj/MLLM_evals/data/seed_bench/llava-seed-bench.jsonl \
#     --image-folder /data3/ljj/proj/MLLM_evals/data/seed_bench \
#     --answers-file /data3/ljj/proj/MLLM_evals/outputs/seed_bench/results-$MODEL_ID.jsonl \
#     --conv-mode vicuna_v1

# python get_seed_bench_score.py \
#     --annotation-file /data3/ljj/proj/MLLM_evals/data/seed_bench/SEED-Bench.json \
#     --result-file /data3/ljj/proj/MLLM_evals/outputs/seed_bench/results-$MODEL_ID.jsonl