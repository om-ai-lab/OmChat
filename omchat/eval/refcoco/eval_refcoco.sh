#!/bin/bash

MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_general_full_en_n104/"
MODEL_ID="n104" 
RAW_DIR="/data3/ljj/proj/MLLM_evals"

echo $MODEL_PATH 
echo $MODEL_ID
PRE="_normalized"
for ds in "refcoco_testA" "refcoco_testB" "refcoco_val" "refcoco+_val" "refcoco+_testA" "refcoco+_testB" "refcocog_val" "refcocog_test"
do 
    echo $ds

    ds_prefix=${ds%%_*}
    echo $ds_prefix

    python eval_refcoco.py \
        --model-path  $MODEL_PATH \
        --question-file $RAW_DIR/data/$ds_prefix/$ds$PRE.jsonl \
        --answers-file  $RAW_DIR/outputs/$ds_prefix/results-$MODEL_ID-$ds$PRE.json \
        --image-folder  $RAW_DIR \
        --conv-mode vicuna_v1 \
        --temperature 0 \
        --dataset $ds$PRE
done