#!/usr/bin/env bash
set -e


# 临时设置Hugging Face下载路径（本次脚本运行生效）
export HF_HOME="./external/e1374390/work/raw_data"
export TRANSFORMERS_CACHE="$HF_HOME"
export DATASETS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"  # 补充huggingface_hub的缓存路径

# 默认配置
REPO_ID="lmms-lab/SEED-Bench"
RAW_DIR="./external/e1374390/work/raw_data"
OUT_DIR="./external/e1374390/work/dataset/${REPO_ID}"
NO_WRITE_IMAGES=false
SCRIPT_PATH="seed.py"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id) REPO_ID="$2"; shift 2;;
    --raw-dir) RAW_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --no-write-images) NO_WRITE_IMAGES=true; shift;;
    --script) SCRIPT_PATH="$2"; shift 2;;
    *) echo "未知参数：$1"; exit 1;;
  esac
done

# 检查脚本是否存在
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "未找到脚本：$SCRIPT_PATH"; exit 1
fi

# 组装参数并运行脚本
ARGS=( "--repo-id" "$REPO_ID" "--raw-dir" "$RAW_DIR" "--out-dir" "$OUT_DIR" )
$NO_WRITE_IMAGES && ARGS+=( "--no-write-images" )

python3 "$SCRIPT_PATH" "${ARGS[@]}"