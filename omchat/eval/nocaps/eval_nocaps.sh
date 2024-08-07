#!/bin/bash
MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_general_full_en_n104/"
MODEL_ID="n104" 

echo $MODEL_PATH $MODEL_ID

python eval_nocaps.py \
    --model-path  $MODEL_PATH \
    --question-file /data3/ljj/proj/MLLM_evals/data/nocaps/nocaps_val.json \
    --answers-file /data3/ljj/proj/MLLM_evals/outputs/nocaps/results-$MODEL_ID.json \
    --image-folder /data3/ljj/proj/MLLM_evals \
    --conv-mode vicuna_v1 \
    --temperature 0