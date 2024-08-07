#!/bin/bash
MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-vicuna-7b-v1.5-v1-1-finetune_336_en_n87_2"
MODEL_ID="n87_2"
RAW_DIR="/data3/ljj/proj/MLLM_evals"
echo $MODEL_PATH $MODEL_ID

for ds in "refcoco_testA" "refcoco_testB" "refcoco_val" "refcoco+_val" "refcoco+_testA" "refcoco+_testB" "refcocog_val" "refcocog_test"
do 
    echo $ds

    ds_prefix=${ds%%_*}
    echo $ds_prefix
    python eval_pointing.py \
        --model-path  $MODEL_PATH \
        --question-file $RAW_DIR/data/refcoco_pointingtask/input_files/finetune_${ds}_input.json \
        --answers-file $RAW_DIR/outputs/refcoco_pointingtask/finetune_${ds}_answer-$MODEL_ID.json \
        --annotations-file $RAW_DIR/data/refcoco_pointingtask/annotations/finetune_${ds}_annotation.json \
        --image-folder $RAW_DIR/data \
        --conv-mode vicuna_v1 \
        --temperature 0 \
        --dataset $ds 
done
