#!/bin/bash

MODEL_PATH="/data3/kyusong/llava/checkpoints/omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_general_full_en_n104/"
MODEL_ID="n104"
echo $MODEL_PATH
echo $MODEL_ID

for ds in "multiple_choice" "freeform_answering"; do
    if [ "$ds" == "multiple_choice" ]; then
        python eval_magnifierbench.py \
            --model-path "$MODEL_PATH" \
            --question-file "/data3/ljj/proj/MLLM_evals/data/magnifierbench/llava_instructions_multiple_choice.json" \
            --image-folder "/data3/ljj/proj/MLLM_evals/data/magnifierbench/images" \
            --answers-file "/data3/ljj/proj/MLLM_evals/outputs/magnifierbench/results-$MODEL_ID-$ds.jsonl" \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode vicuna_v1
    elif [ "$ds" == "freeform_answering" ]; then
        python eval_magnifierbench.py \
            --model-path "$MODEL_PATH" \
            --question-file "/data3/ljj/proj/MLLM_evals/data/magnifierbench/llava_instructions_freeform_answering.json" \
            --image-folder "/data3/ljj/proj/MLLM_evals/data/magnifierbench/images" \
            --answers-file "/data3/ljj/proj/MLLM_evals/outputs/magnifierbench/results-$MODEL_ID-$ds.jsonl" \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --freeform-answering
    fi
done

python get_magnifierbench_score.py \
    --gpt-model gpt-4-0613 \
    --answers-mc "/data3/ljj/proj/MLLM_evals/outputs/magnifierbench/results-$MODEL_ID-multiple_choice.jsonl" \
    --answers-ff "/data3/ljj/proj/MLLM_evals/outputs/magnifierbench/results-$MODEL_ID-freeform_answering.jsonl" 
