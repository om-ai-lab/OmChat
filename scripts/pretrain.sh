#!/bin/bash
WEIGHT_VERSION=v1
export WANDB__SERVICE_WAIT=300
export WANDB_MODE="disabled"
export NCCL_DEBUG=INFO

#--image_grid_pinpoints "[(896, 896), (448, 896), (896, 448), (448, 1344), (1344, 448), (1344, 896),(896, 1344)]" \
PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' NCCL_IB_DISABLE=0 WANDB_MODE="disabled" deepspeed  --master_addr=10.8.25.20 --master_port=25005 --num_nodes 10 --hostfile=hostfile omchat/train/train_mem.py \
    --deepspeed scripts/new_zero3.json \
    --model_name_or_path resources/Qwen2-7B \
    --version $WEIGHT_VERSION \
    --vision_tower resources/InternViT-6B-448px-V1-5 \
    --image_grid_pinpoints "[(896, 896), (448, 896), (896, 448), (448, 1344), (1344, 448)]" \
    --image_size 448 \
    --pretrain_mm_mlp_adapter checkpoints/vision-qwen-2b-siglip_kp7/mm_projector.bin \
    --version v1 \
    --big_data True \
    --image_aspect_ratio pad \
    --data_type pretrain \
    --data_path data1.jsonl:data2.jsonl \
    --image_folder image1/:image2/ \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /path/to/pretrained_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb \
