#!/bin/bash

#MODEL_PATH="checkpoints/omchat-qwen-2b-qllama-fk9"
#MODEL_PATH="checkpoints/omchat-qllama-qwen2-7bb-internvit6b-fk51"
#MODEL_PATH="checkpoints/omchat-qllama-qwen2-7bb-internvit6b-fk53_2/checkpoint-3000"
#MODEL_PATH="checkpoints/omchat-qwen2-qllama-7b-internvit6b-tokenpacker-fk56"
#MODEL_PATH="checkpoints/omchat-qllama-qwen2-7b-internvit300m-fk54"
#MODEL_PATH="checkpoints/omchat-qllama-qwen2-7bb-internvit6b-fk51"
MODEL_PATH="/data2/omchat_dev/omchat/checkpoints/omchat-qllama-qwen2-7bb-internvit6b-fk53_2"


MODEL_ID=$(basename "$MODEL_PATH")
MODEL_ID=${MODEL_ID##*_}
MODEL_ID="fk53"
RAW_DIR="/data1/MLLM_evals"
LOG_FILE="eval_logs/$MODEL_ID.txt"
mkdir "eval_logs"
mkdir "outputs"

echo "MODEL_PATH: $MODEL_PATH"
echo "MODEL_ID: $MODEL_ID"
##======== 1. scienceQA
python -m omchat.eval.scienceqa.eval_sciqa \
    --model-path $MODEL_PATH \
    --question-file $RAW_DIR/data/scienceqa/llava_test_CQM-A.json \
    --image-folder $RAW_DIR/data/scienceqa/images/test \
    --answers-file outputs/scienceqa/results-$MODEL_ID.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m omchat.eval.scienceqa.get_sciqa_score \
    --base-dir $RAW_DIR/data/scienceqa \
    --result-file outputs/scienceqa/results-$MODEL_ID.jsonl \
    --output-file outputs/scienceqa/results-$MODEL_ID-output.jsonl \
    --output-result  outputs/scienceqa/results-$MODEL_ID-result.json \
    >> $LOG_FILE
: '
#======== 2. Text VQA
python -m omchat.eval.textvqa.eval_textvqa_anyres \
        --model-path  $MODEL_PATH \
        --question-file $RAW_DIR/data/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder $RAW_DIR/data/textvqa/train_images \
        --answers-file outputs/textvqa/results-$MODEL_ID.jsonl \
        --conv-mode vicuna_v1
python -m omchat.eval.textvqa.get_textvqa_score \
    --annotation-file $RAW_DIR/data/textvqa/TextVQA_0.5.1_val.json \
    --result-file outputs/textvqa/results-$MODEL_ID.jsonl \
    >> $LOG_FILE
'
#======== 3. GQA
python -m omchat.eval.gqa.eval_gqa_anyres\
    --model-path  $MODEL_PATH \
    --question-file $RAW_DIR/data/gqa/llava_gqa_testdev_balanced.jsonl \
    --image-folder $RAW_DIR/data/gqa/images \
    --answers-file outputs/gqa/results-$MODEL_ID.jsonl \
    --conv-mode vicuna_v1

python -m omchat.eval.gqa.convert_gqa_for_eval --src outputs/gqa/results-$MODEL_ID.jsonl --dst outputs/gqa/testdev_balanced_predictions-$MODEL_ID.json

python -m omchat.eval.gqa.get_gqa_score --tier testdev_balanced --questions $RAW_DIR/data/gqa/testdev_balanced_all_questions.json --predictions outputs/gqa/testdev_balanced_predictions-$MODEL_ID.json >> $LOG_FILE

#======== 4. SEED
python -m omchat.eval.seed_bench.eval_seed_bench_anyres \
    --model-path  $MODEL_PATH \
    --question-file $RAW_DIR/data/seed_bench/llava-seed-bench_without_video.jsonl \
    --image-folder $RAW_DIR/data/seed_bench \
    --answers-file outputs/seed_bench/results-$MODEL_ID.jsonl \
    --conv-mode vicuna_v1 \
    --temperature 0 

python -m omchat.eval.seed_bench.get_seed_bench_score \
    --annotation-file $RAW_DIR/data/seed_bench/SEED-Bench.json \
    --result-file outputs/seed_bench/results-$MODEL_ID.jsonl >> $LOG_FILE

#======== 5. AI2D
python -m omchat.eval.ai2d.eval_ai2d_anyres \
        --model-path  $MODEL_PATH \
        --question-file $RAW_DIR/data/ai2d/test.jsonl \
        --image-folder $RAW_DIR/data/ai2d/ai2d \
        --answers-file outputs/ai2d/results-$MODEL_ID.json \
        --conv-mode vicuna_v1 \
        --temperature 0 >> $LOG_FILE
    
##======== 6. OKVQA
python -m omchat.eval.okvqa.eval_okvqa_anyres \
        --model-path  $MODEL_PATH \
        --question-file $RAW_DIR/data/okvqa/okvqa_val.jsonl \
        --conv-mode vicuna_v1 \
        --image-folder $RAW_DIR/data/okvqa/val2014 \
        --temperature 0 \
        --question $RAW_DIR/data/okvqa/OpenEnded_mscoco_val2014_questions.json \
        --annotation-file $RAW_DIR/data/okvqa/mscoco_val2014_annotations.json \
        --answers-file outputs/okvqa/results-$MODEL_ID.json  >> $LOG_FILE
