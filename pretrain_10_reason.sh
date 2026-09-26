
#!/usr/bin/env bash
export MASTER_PORT=29501
export TORCH_DISTRIBUTED_DEFAULT_PORT=29501

# export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
export CUDA_VISIBLE_DEVICES=0,1

# 设置 Hugging Face 模型仓库的镜像地址，方便下载模型等资源
export HF_ENDPOINT=API_ENDPOINT_NOT_CONFIGURED
# 2. 强制把 HF 缓存目录切到 scratch
export HF_HOME="./external/e1374390/work/huggingface_cache"
# 可选：显式再同步一份旧变量，防止某些库只认 TRANSFORMERS_CACHE
export TRANSFORMERS_CACHE="$HF_HOME"

export TRITON_CACHE_DIR=./external/$USER/triton_cache
mkdir -p $TRITON_CACHE_DIR
# 使用 deepspeed 工具运行 simple_LLaVA_run.py 脚本
# --include localhost:0,1 表示指定在本地的 0 号和 1 号 GPU 上运行任务
# 注：localhost 代表本地机器，0 和 1 是 GPU 的编号
MODEL_NAME="dinov_qwen_10"

export TRITON_CACHE_DIR=./external/e1374390/triton_cache   
# 获取SLURM分配的GPU数量（例如1、2等）
# NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)

# # 生成逻辑GPU编号列表（0,1,...,N-1）
# LOGICAL_GPUS=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
# deepspeed --include localhost:0 pretrain.py \dinov_model_qwen2.5-1.5b
# LLaVA-CC3M-Pretrain-595K
  # --train_type tune_mm_mlp_adapter \
  # deepspeed pretrain.py \
deepspeed --master_port $MASTER_PORT --include localhost:0,1 pretrain_10_reason.py \
  --deepspeed zero2.json \
  --model_name_or_path ./external/e1374390/work/${MODEL_NAME} \
  --output_dir ./external/e1374390/work/${MODEL_NAME}/reason_trained \
  --data_path ./external/e1374390/work/dataset/FLUX-Reason/train \
  --train_type tune_mm_mlp_adapter \
  --bf16 true \
  --tf32 true \
  --dataloader_num_workers 10 \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 10000 \
  --save_total_limit 3 \
  --report_to "tensorboard" \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --gradient_checkpointing true \
  --logging_steps 20 \
  --report_to none \

# 如果 loss 波动大，可减半 LR 或增大 warmup