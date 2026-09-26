#!/usr/bin/env bash
set -euo pipefail

source ./external/miniconda3/etc/profile.d/conda.sh
conda activate bot

export MASTER_PORT=29512
export TORCH_DISTRIBUTED_DEFAULT_PORT=29512
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export HF_ENDPOINT=API_ENDPOINT_NOT_CONFIGURED
export HF_HOME="./external/granulon_work/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TRITON_CACHE_DIR=./external/granulon_work/triton_cache
mkdir -p "$TRITON_CACHE_DIR"

CODE_ROOT="./external/granulon_Codex"
MODEL_NAME="dinov_qwen_10_bankV5-liteV1"
OUTPUT_NAME="reason_trained_meta_cog_v1"

deepspeed --master_port "$MASTER_PORT" --include localhost:0,1,2,3,4,5,6,7 \
  "${CODE_ROOT}/pretrain_10_reason_bankV5-fixedV2.py" \
  --deepspeed "${CODE_ROOT}/zero2.json" \
  --model_name_or_path "./external/granulon_work/${MODEL_NAME}" \
  --output_dir "./external/granulon_work/${MODEL_NAME}/${OUTPUT_NAME}" \
  --data_path "./FLUX-Reason-6M_processed/train" \
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
  --semantic_bank_path "./Bank/semantic_bankV5/semantic_bank-liteV2.pt" \
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
  --patch_drop 5 \
  --meta_cog_enabled true \
  --meta_cog_num_steps 4 \
  --meta_cog_min_steps 2 \
  --meta_cog_stop_threshold 0.78 \
  --meta_cog_stability_threshold 0.015 \
  --meta_cog_state_tokens 14
