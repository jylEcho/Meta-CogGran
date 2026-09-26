import os
import io
import time
import json
from pathlib import Path
from requests.exceptions import HTTPError

import torch
import torch.distributed as dist
from datasets import load_dataset
from PIL import Image

def main():
    # 初始化分布式环境（使用 NCCL）
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"Initialized rank {rank} / {world_size}")

    # 小延迟按 rank 错开请求，减少并发 429
    time.sleep(rank * 5)

    # HF 环境建议（可按需修改）
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['HF_XET_MAX_CONCURRENT_DOWNLOADS'] = '2'
    os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'

    # 输出目录
    output_dir = Path("./external/e01374390/work/dataset/ImageNet-Think")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集（带重试，处理网络/429）
    max_retries = 6
    for attempt in range(max_retries):
        try:
            ds = load_dataset("krishnateja95/ImageNet-Think", split="train", streaming=True)
            break
        except HTTPError as e:
            wait = 2 ** attempt
            print(f"Rank {rank}: load_dataset HTTPError, attempt {attempt+1}/{max_retries}, sleeping {wait}s")
            time.sleep(wait)
            if attempt == max_retries - 1:
                raise
    print(f"Rank {rank}: Dataset loaded.")

    out = []
    count = 0

    for i, ex in enumerate(ds):
        # 按 rank 分片处理
        if i % world_size != rank:
            continue

        try:
            img = ex.get("image")
            rid = ex.get("id", f"img_{i:08d}")
            rel_img_name = f"{rid}.jpg"

            # 支持多种 image 表示：PIL.Image / dict(bytes=...) / bytes
            saved = False
            if isinstance(img, Image.Image):
                im = img
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(str(images_dir / rel_img_name), format="JPEG", quality=95)
                saved = True
            elif isinstance(img, dict) and img.get("bytes") is not None:
                im = Image.open(io.BytesIO(img["bytes"]))
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(str(images_dir / rel_img_name), format="JPEG", quality=95)
                saved = True
            elif isinstance(img, (bytes, bytearray)):
                im = Image.open(io.BytesIO(img))
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(str(images_dir / rel_img_name), format="JPEG", quality=95)
                saved = True
            else:
                # 若 image 是 path/str/其他类型，尝试让 PIL 打开（容错）
                try:
                    im = Image.open(img)
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    im.save(str(images_dir / rel_img_name), format="JPEG", quality=95)
                    saved = True
                except Exception:
                    saved = False

            if not saved:
                print(f"Rank {rank}: skipped saving image for item {i} id={rid}")
        except Exception as e:
            print(f"Rank {rank}: failed to process image {i}: {e}")
            continue

        # 构建对话条目（与原脚本保持一致）
        a_text = ex.get("answer_2", "")
        entry = {
            "id": rid,
            "image": rel_img_name,
            "conversations": [
                {"from": "human", "value": "Please analyze this image step by step. Explain your reasoning process. Describe this image and give as much information as possible.\n<image>"},
                {"from": "gpt", "value": a_text},
            ],
        }
        out.append(entry)

        count += 1
        if count % 1000 == 0:
            print(f"Rank {rank}: Processed {count} items")

    print(f"Rank {rank}: Total items processed: {count}")

    # 写入每个 rank 的结果文件
    rank_chat_path = output_dir / f"chat_rank{rank}.json"
    with open(rank_chat_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

    # 等待所有 rank 完成
    dist.barrier()

    # rank 0 合并
    if rank == 0:
        merged = []
        for r in range(world_size):
            p = output_dir / f"chat_rank{r}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    merged += json.load(f)
        with open(output_dir / "chat.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=4)
        print("Merged all JSONs to chat.json")

if __name__ == "__main__":
    main()