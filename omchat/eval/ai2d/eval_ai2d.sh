#!/bin/bash

MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_general_full_en_n104/"
MODEL_ID="n104" 


echo $MODEL_PATH
echo $MODEL_ID

# python eval_ai2d_vicuna.py \
#     --model-path  $MODEL_PATH \
#     --question-file /data/MLLM_evals/data/ai2d/test.jsonl \
#     --image-folder /data/MLLM_evals/data/ai2d/ai2d \
#     --answers-file /data/MLLM_evals/outputs/ai2d/results-$MODEL_ID.json \
#     --conv-mode vicuna_v1 \
#     --temperature 0

python eval_ai2d_qllama.py \
    --model-path  $MODEL_PATH \
    --question-file /data/MLLM_evals/data/ai2d/test.jsonl \
    --image-folder /data/MLLM_evals/data/ai2d/ai2d \
    --answers-file /data/MLLM_evals/outputs/ai2d/results-$MODEL_ID.json \
    --conv-mode vicuna_v1 \
    --temperature 0
