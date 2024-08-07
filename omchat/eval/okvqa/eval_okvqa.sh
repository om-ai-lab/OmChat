#!/bin/bash

MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_general_full_en_n104/"
MODEL_ID="n104"

echo $MODEL_PATH
echo $MODEL_ID

python eval_okvqa_qllama.py \
    --model-path  $MODEL_PATH \
    --question-file /data3/ljj/proj/MLLM_evals/data/okvqa/okvqa_val.jsonl \
    --conv-mode vicuna_v1 \
    --image-folder /data3/ljj/proj/MLLM_evals/data/okvqa/val2014 \
    --temperature 0 \
    --question /data3/ljj/proj/MLLM_evals/data/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --annotation-file /data3/ljj/proj/MLLM_evals/data/okvqa/mscoco_val2014_annotations.json \
    --answers-file /data3/ljj/proj/MLLM_evals/outputs/okvqa/results-$MODEL_ID.json 


# python eval_okvqa_vicuna.py \
#     --model-path  $MODEL_PATH \
#     --question-file /data3/ljj/proj/MLLM_evals/data/okvqa/okvqa_val.jsonl \
#     --conv-mode vicuna_v1 \
#     --image-folder /data3/ljj/proj/MLLM_evals/data/okvqa/val2014 \
#     --temperature 0 \
#     --question /data3/ljj/proj/MLLM_evals/data/okvqa/OpenEnded_mscoco_val2014_questions.json \
#     --annotation-file /data3/ljj/proj/MLLM_evals/data/okvqa/mscoco_val2014_annotations.json \
#     --answers-file /data3/ljj/proj/MLLM_evals/outputs/okvqa/results-$MODEL_ID.json 
    
