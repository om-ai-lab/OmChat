#echo "1h."
#sleep 1h

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_RETRY_CNT=10

PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' deepspeed --master_addr=172.16.10.22 --master_port=25004 --num_nodes 10 --hostfile=hostfile  omchat/train/train_mem.py \
    --deepspeed ./scripts/new_zero3.json \
    --model_name_or_path /path/to/pretrained_model \
    --version v1 \
    --data_path  llava_instruct_80k.json \
    --image_folder /path/to/coco/train2017 \
    --data_type qwen \
    --image_aspect_ratio anyres \
    --mm_projector_type mlp2x_gelu \
    --vision_tower resources/InternViT-6B-448px-V1-5 \
    --mm_vision_select_layer -1 \
    --image_size 448 \
    --image_grid_pinpoints "[(896, 896), (448, 896), (896, 448), (448, 1344), (1344, 448), (1344, 896),(896, 1344), (1344, 1344)]" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /path/to/finetune_modell \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --cache_dir "./cache_dir"
