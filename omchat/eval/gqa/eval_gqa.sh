#!/bin/bash

MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_general_full_en_n104/"
MODEL_ID="n104"
output_file=/data3/ljj/proj/MLLM_evals/outputs/gqa/results-$MODEL_ID.jsonl

python eval_gqa_new.py \
    --model-path  $MODEL_PATH \
    --question-file /data3/ljj/proj/MLLM_evals/data/gqa/llava_gqa_testdev_balanced.jsonl \
    --image-folder /data3/ljj/proj/MLLM_evals/data/gqa/images \
    --answers-file $output_file \
    --conv-mode vicuna_v1

python convert_gqa_for_eval.py --src $output_file --dst /data3/ljj/proj/MLLM_evals/outputs/gqa/testdev_balanced_predictions.json

python get_gqa_score.py --tier testdev_balanced --questions /data3/ljj/proj/MLLM_evals/data/gqa/testdev_balanced_all_questions.json --predictions /data3/ljj/proj/MLLM_evals/outputs/gqa/testdev_balanced_predictions.json