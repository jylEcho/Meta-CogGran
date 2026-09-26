set -euo pipefail

# 仅使用第一张 GPU（按 PCI 顺序）
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Hugging Face 镜像与缓存
export HF_ENDPOINT=${HF_ENDPOINT:-API_ENDPOINT_NOT_CONFIGURED}
export HF_HOME="${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface}"
unset TRANSFORMERS_CACHE

# 模型与数据路径
BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
CLIP_MODEL="./external/huggingface/clip-vit-base-patch32"
CANDIDATE_MODEL="${CANDIDATE_MODEL:-./external/e1374390/work/dinov_model_qwen2.5-1.5b}"

# SIGLIP="${SIGLIP:-./external/e1374390/work/siglip_qwen}"
# DINOV_5="${DINOV_5:-./external/e1374390/work/dinov_qwen_5}"
DINOV_10="./external/granulon_work/dinov_qwen_10_bankV3"
DINOV_10_llama="./external/granulon_work/dinov_llama_10"
DINOV_10_qwen257BInstruct="./external/granulon_work/dinov_qwen_10_qwen-2.5-7B-Instruct"
DINOV_10_qwen38Bbase="./external/granulon_work/dinov_qwen_10_qwen3-8B-Base"
DINOV_10_qwen257BInstruct_debugV5="./external/granulon_work/dinov_qwen_10_qwen-2.5-7B-Instruct_debugV5"
# DINOV_10_BANKV3_qwen38Bbase="./external/granulon_work/dinov_qwen_10_bankV3_qwen3-8B-base"

DINOV_10_BANKV3_llama="./external/granulon_work/dinov_llama_10_bankV3"
DINOV_10_BANKV3_qwen257BInstruct_debugV5="./external/granulon_work/dinov_qwen_10_bankV3_qwen2.5-7B-Instruct"
DINOV_10_BANKV3_qwen257BInstruct_debugV5_Med="./external/granulon_work/dinov_qwen_10_bankV3_qwen2.5-7B-Instruct_Med"
DINOV_10_BANKV3_qwen38Bbase_debugV5="./external/granulon_work/dinov_qwen_10_bankV3_qwen3-8B-Base"
DINOV_10_BANKV3_SEED="./external/granulon_work/dinov_qwen_10_bankV3"

DINOV_10_BANKV5_LITEV1="./external/granulon_work/dinov_qwen_10_bankV5-liteV1"
DINOV_10_BANKV5_LITEV2="./external/granulon_work/dinov_qwen_10_bankV5-liteV1"
DINO_QWEN3_8B_10_bankV5="./external/granulon_work/dinov_qwen3-8B_10_bankV5"
SIGLIP_10_qwen257BInstruct_debugV5="./external/granulon_work/siglip_qwen_10_qwen-2.5-7B-Instruct"
SIGLIP_10_qwen38Bbase_debugV5="./external/granulon_work/siglip_qwen_10_qwen3-8B-Base_debugV5"
# DINOV_10_bANKV3_qwen2.5-7B-Instruct="./external/granulon_work/dinov_qwen_10_bankV3_qwen2.5-7B-Instruct"
# DINOV_10_BANKV5_LITEV3="./external/granulon_work/dinov_qwen_10_bankV5-liteV1"
# DINOV_20="${DINOV_20:-./external/e1374390/work/dinov_qwen_20}"
# DINOV_30="${DINOV_30:-./external/e1374390/work/dinov_qwen_30}"
# DINOV_50="${DINOV_50:-./external/e1374390/work/dinov_qwen_50}"
# SIGLIP_QWEN="${SIGLIP_QWEN:-./external/e1374390/work/siglip_qwen}"
# DATASET_DIR="${DATASET_DIR:-./external/e1374390/work/dataset/lmms-lab/SEED-Bench}"
# DATASET_DIR="${DATASET_DIR:-./external/e1374390/work/dataset/LLaVA-CC3M-Pretrain-595K}"
# DATASET_DIR="${DATASET_DIR:-./external/e1374390/work/dataset/A-OKVQA}"
DATASET_DIR="./external/granulon/FLUX-Reason-6M_processed/test"
# 评测参数
NUM_SAMPLES="${NUM_SAMPLES:-39}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

if [ ! -d "$DATASET_DIR" ]; then
  echo "ERROR: Dataset directory not found: $DATASET_DIR"
  exit 1
fi

PYBIN="${PYBIN:-python -u}"

# echo "==> Evaluating candidate (dinov_5 in seed)"
# $PYBIN eval.py \
#   --model "$DINOV_5" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_5 \
#   --use_lora No


# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_10_bankV3 \
#   --use_lora No

# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_BANKV5_LITEV1" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_10_bankV5_liteV1 \
#   --use_lora No

# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_BANKV5_LITEV2" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_10_bankV5_liteV1 \
#   --use_lora No

# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_BANKV3_qwen257BInstruct" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_qwen_10_bankV3_qwen2.5-7B-Instruct \
#   --use_lora No

# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_qwen257BInstruct" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_qwen_10_qwen2.5-7B-Instruct \
#   --use_lora No

# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_qwen257BInstruct_debugV5" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_qwen_10_qwen2.5-7B-Instruct_debugV5 \
#   --use_lora No

# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_BANKV3_qwen257BInstruct_debugV5" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_qwen_10_bankV3_qwen2.5-7B-Instruct \
#   --use_lora No

# DINOV_10_BANKV3_qwen38Bbase_debugV5
# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_BANKV3_qwen38Bbase_debugV5" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_qwen_10_bankV3_qwen3-8B-Base \
#   --use_lora No

# DINOV_10_llama
# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_llama" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_llama_10 \
#   --use_lora No

# DINOV_10_BANKV3_llama
# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_BANKV3_llama" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_llama_10_bankV3 \
#   --use_lora No

# SIGLIP_10_qwen257BInstruct_debugV5
# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$SIGLIP_10_qwen257BInstruct_debugV5" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model siglip_qwen_10_qwen-2.5-7B-Instruct \
#   --use_lora No

# DINOV_10_BANKV3_qwen257BInstruct_debugV5_Med
# echo "==> Evaluating candidate (dinov_10 in seed)"
# $PYBIN ./external/granulon/eval.py \
#   --model "$DINOV_10_BANKV3_qwen257BInstruct_debugV5_Med" \
#   --model_type llava \
#   --dataset_dir "$DATASET_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_new_tokens "$MAX_NEW_TOKENS" \
#   --use_custom_model dinov_qwen_10_bankV3_qwen2.5-7B-Instruct_Med \
#   --use_lora No

# dinov_qwen3-8B_10_bankV5
echo "==> Evaluating candidate (dinov_10 in seed)"
$PYBIN ./external/granulon/eval.py \
  --model "$DINO_QWEN3_8B_10_bankV5" \
  --model_type llava \
  --dataset_dir "$DATASET_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --use_custom_model dinov_qwen3-8B_10_bankV5 \
  --use_lora No
