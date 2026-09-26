#!/usr/bin/env bash
set -euo pipefail

source ./external/miniconda3/etc/profile.d/conda.sh
conda activate bot56

export MASTER_PORT=29502
export TORCH_DISTRIBUTED_DEFAULT_PORT=29502

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export HF_ENDPOINT=API_ENDPOINT_NOT_CONFIGURED
export HF_HOME="./external/granulon_work/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

export TRITON_CACHE_DIR=./external/granulon_work/triton_cache
mkdir -p "$TRITON_CACHE_DIR"

MODEL_NAME="dinov_qwen_10_bankV5-liteV1_qwen2.5-7B-Instruct"

echo "===== ENV CHECK ====="
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
which python
python -V
python -c "import sys; print('sys.executable =', sys.executable)"
python -c "import transformers; print('transformers =', transformers.__version__, transformers.__file__)"
which deepspeed
head -n 1 "$(which deepspeed)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "====================="

deepspeed --master_port "$MASTER_PORT" --include localhost:0,1,2,3,4,5,6,7 \
  ./external/granulon/pretrain_10_reason_bankV5-fixed.py \
  --deepspeed ./external/granulon/zero2.json \
  --model_name_or_path ./external/granulon_work/${MODEL_NAME} \
  --output_dir ./external/granulon_work/${MODEL_NAME}/reason_trained \
  --data_path ./external/granulon/FLUX-Reason-6M_processed/train \
  --train_type tune_mm_mlp_adapter \
  --bf16 true \
  --tf32 true \
  --dataloader_num_workers 10 \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers true \
  --num_train_epochs 4 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 10000 \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --gradient_checkpointing true \
  --logging_steps 20 \
  --report_to none \
  --semantic_bank_path ./external/granulon/Bank/semantic_bankV5/qwen-2.5-7B-Instruct_debugV5.pt \
  --semantic_cluster_num 10 \
  --global_bank_topk 4 \
  --entity_bank_topk 6 \
  --layout_bank_topk 2 \
  --relation_bank_topk 2 \
  --layout_grid_size 4 \
  --relation_near_threshold 0.22 \
  --relation_overlap_threshold 0.10 \
  --relation_direction_margin 0.08 \
  --relation_max_pairs 8 \
  --patch_drop 5 