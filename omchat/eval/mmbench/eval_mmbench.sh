#!/bin/bash

MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_general_full_en_n104/"
MODEL_ID="n104"
echo $MODEL_PATH
echo $MODEL_ID

RAW_DIR="/data3/ljj/proj/MLLM_evals"

for ds in ["ccbench_20231003", "mmbench_dev_cn_20231003", "mmbench_dev_en_20231003"]
    python eval_mmbench_qllama.py \
        --model-path  $MODEL_PATH \
        --question-file $RAW_DIR/data/MMBench/$ds.jsonl \
        --image-folder $RAW_DIR/data/MMBench \
        --answers-file $RAW_DIR/outputs/MMBench/results-$MODEL_ID.jsonl \
        --conv-mode vicuna_v1 \
        --temperature 0 

    python get_mmbench_score.py \
        --annotation-file $RAW_DIR/data/MMBench/SEED-Bench.json \
        --result-file $RAW_DIR/outputs/MMBench/results-$MODEL_ID.jsonl
