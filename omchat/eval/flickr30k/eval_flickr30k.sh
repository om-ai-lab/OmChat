#!/bin/bash

MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_general_full_en_n104/"
MODEL_ID="n104" 

RAW_DIR="/data3/ljj/proj/MLLM_evals"

echo $MODEL_PATH 
echo $MODEL_ID

python eval_flickr30k.py \
    --model-path  $MODEL_PATH \
    --question-file $RAW_DIR/data/flickr30k/flickr30k_karpathy_test.json \
    --answers-file $RAW_DIR/outputs/flickr30k/results-$MODEL_ID.json \
    --image-folder $RAW_DIR \
    --conv-mode vicuna_v1 \
    --temperature 0
