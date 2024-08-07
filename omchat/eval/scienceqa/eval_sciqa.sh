#!/bin/bash

MODEL_PATH=""
MODEL_ID="" 
echo $MODEL_PATH
echo $MODEL_ID

python eval_sciqa.py \
    --model-path $MODEL_PATH \
    --question-file MLLM_evals/data/scienceqa/llava_test_CQM-A.json \
    --image-folder MLLM_evals/data/scienceqa/images/test \
    --answers-file MLLM_evals/outputs/scienceqa/results-$MODEL_ID.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python get_sciqa_score.py \
    --base-dir MLLM_evals/data/scienceqa \
    --result-file MLLM_evals/outputs/scienceqa/results-$MODEL_ID.jsonl \
    --output-file MLLM_evals/outputs/scienceqa/results-$MODEL_ID-output.jsonl \
    --output-result  MLLM_evals/outputs/scienceqa/results-$MODEL_ID-result.json
