#!/bin/bash

MODEL_PATH="/nas/data3/kyusong/omchat/checkpoints/omchat-llava-vicuna-7b-v1.5-v1-1-finetune_224_en_s3"
MODEL_ID="n104"
echo $MODEL_PATH
echo $MODEL_ID
python eval_textvqa.py \
    --model-path  $MODEL_PATH \
    --question-file /data3/ljj/proj/MLLM_evals/data/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /data3/ljj/proj/MLLM_evals/data/textvqa/train_images \
    --answers-file /data3/ljj/proj/MLLM_evals/outputs/textvqa/results-$MODEL_ID.jsonl \
    --conv-mode vicuna_v1

python get_textvqa_score.py \
    --annotation-file /data3/ljj/proj/MLLM_evals/data/textvqa/TextVQA_0.5.1_val.json \
    --result-file /data3/ljj/proj/MLLM_evals/outputs/textvqa/results-$MODEL_ID.jsonl