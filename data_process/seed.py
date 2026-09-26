"""
SEED-Bench 数据预处理工具

功能说明：
- 将 SEED-Bench 的 parquet 文件转换为 LLaVA 格式的 chat.json
- 格式：{"id": str, "image": str, "conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
- 问题包含四个选项，以 "\n<image>" 结尾
- 答案从选项字母转换为对应的文本内容
- 图片使用相对路径而非 base64 编码
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Optional
from tqdm import tqdm


# 从 HuggingFace 下载 SEED-Bench 数据集
def download_seed_bench(dest_dir: str, repo_id: str = "lmms-lab/SEED-Bench") -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required for download; install it or skip download"
        ) from e

    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    # 下载数据集快照到本地（注意：SEED-Bench 是 dataset 类型）
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        revision="main",
        max_workers=4,
        allow_patterns=["*.parquet"],
    )
    return Path(snapshot_path)


# 将 bytes 转换为 Base64 编码字符串
def bytes_to_base64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


# 问题模板：包含四个选项
QUESTION_TEMPLATE = (
    "{question}"
    "\nOptions:"
    "\nA. {choice_a}"
    "\nB. {choice_b}"
    "\nC. {choice_c}"
    "\nD. {choice_d}"
    "\n(Answer with the option content)"
)


# 确保问题以 \n<image> 结尾
def ensure_image_marker(q: str) -> str:
    q = q.rstrip("\n")
    if not q.endswith("<image>"):
        q = f"{q}\n<image>"
    return q


# 将答案字母 (A-D) 转换为对应的选项文本
def pick_answer_text(row: Dict) -> str:
    ans = (row.get("answer") or "").strip()
    mapping = {
        "A": row.get("choice_a"),
        "B": row.get("choice_b"),
        "C": row.get("choice_c"),
        "D": row.get("choice_d"),
    }
    if ans in mapping and mapping[ans] is not None:
        return str(mapping[ans])
    # 如果未找到对应选项，返回原始答案
    return str(ans)


# 从 parquet 的 image 字段提取相对路径文件名，优先使用 'path'，默认扩展名为 .jpg
def extract_rel_image_name(image_field) -> Optional[str]:
    try:
        # image 可能是列表或数组，取第一个元素
        if image_field is None:
            return None
        if isinstance(image_field, (list, tuple)):
            elem = image_field[0] if image_field else None
        else:
            elem = image_field[0]
        if isinstance(elem, dict):
            p = elem.get("path")
            if p:
                # 只保留文件名，不包含目录
                base = os.path.basename(str(p))
                if not base.lower().endswith((".jpg", ".jpeg", ".png")):
                    base = base + ".jpg"
                return base
    except Exception:
        pass
    return None


# 将图片 bytes 写入本地文件，成功返回完整路径，失败返回 None
def maybe_write_local_image(image_field, images_dir: Path, rel_name: str) -> Optional[Path]:
    try:
        if image_field is None:
            return None
        # 取第一个元素的 'bytes'
        if isinstance(image_field, (list, tuple)):
            elem = image_field[0] if image_field else None
        else:
            elem = image_field[0]
        if isinstance(elem, dict) and elem.get("bytes"):
            images_dir.mkdir(parents=True, exist_ok=True)
            out_path = images_dir.joinpath(rel_name)
            with open(out_path, "wb") as f:
                f.write(elem["bytes"])
            return out_path
    except Exception:
        return None
    return None


# 从多个 parquet 文件读取数据，返回字典列表
def rows_from_parquets(parquet_paths: Iterable[Path], limit: Optional[int] = None) -> List[Dict]:
    import pandas as pd
    records: List[Dict] = []
    n = 0
    for p in parquet_paths:
        df = pd.read_parquet(str(p), engine="pyarrow")
        for _, row in df.iterrows():
            rec = {k: row[k] for k in df.columns}
            records.append(rec)
            n += 1
            if limit is not None and n >= limit:
                return records
    return records


# 将 SEED-Bench 原始数据转换为 LLaVA 格式的对话数据，可选择将图片 bytes 写入本地文件
def normalize_seed_records(records: List[Dict], images_subdir: str = "images", write_local_images: bool = False, out_root: Optional[Path] = None) -> List[Dict]:
    out: List[Dict] = []
    images_dir = out_root.joinpath(images_subdir) if (write_local_images and out_root) else None

    for row in tqdm(records, desc="Normalizing SEED-Bench records", unit="record"):
        # 组合问题和选项
        q = QUESTION_TEMPLATE.format(
            question=str(row.get("question", "")).strip(),
            choice_a=str(row.get("choice_a", "")),
            choice_b=str(row.get("choice_b", "")),
            choice_c=str(row.get("choice_c", "")),
            choice_d=str(row.get("choice_d", "")),
        )
        q = ensure_image_marker(q)

        # 获取答案文本
        a_text = pick_answer_text(row)

        # 图片相对路径
        rel_img_name = extract_rel_image_name(row.get("image"))
        if rel_img_name is None:
            # 使用 data_id 作为备选
            did = str(row.get("data_id", "seed_image"))
            rel_img_name = f"{os.path.basename(did)}.jpg"

        # 可选：写入本地图片文件
        if images_dir is not None:
            maybe_write_local_image(row.get("image"), images_dir, rel_img_name)

        # 生成唯一 ID
        rid = f"SEED_{row.get('data_id', '')}_{row.get('question_id', '')}"

        item = {
            "id": rid,
            "image": rel_img_name,  # 相对路径，pretrain.py 会拼接 dataset_dir/images
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": str(a_text)},
            ],
        }
        out.append(item)

    return out


# 保存数据为 JSON 格式
def save_chat_json(items: List[Dict], out_root: Path, filename: str = "chat.json") -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root.joinpath(filename)
    with open(out_path, "w", encoding="utf-8") as f:
        # 使用缩进，便于阅读，风格与 evaldata/processed_data.json 保持一致
        json.dump(items, f, ensure_ascii=False, indent=4)
    return out_path


# 主函数：解析命令行参数并执行数据转换
def _collect_parquet_paths_from_dir(root: Path) -> List[Path]:
    return sorted(root.rglob("*.parquet"))


def main():
    parser = argparse.ArgumentParser(description="下载并将 SEED-Bench 规范化为 LLaVA 风格 chat.json")
    parser.add_argument("--repo-id", type=str, default="lmms-lab/SEED-Bench", help="HuggingFace 数据集 repo id")
    parser.add_argument("--raw-dir", type=str, default="datasets/SEED-Bench", help="原始数据下载/存放目录")
    parser.add_argument("--out-dir", type=str, default="datasets/seed", help="输出数据集根目录（相对路径推荐，如 datasets/seed）")
    parser.add_argument("--no-write-images", action="store_true", help="不写出图片文件（仅保留相对路径名）")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_root = Path(args.out_dir)

    # 如果原始目录不存在或没有 parquet，则执行完整下载
    parquet_paths: List[Path] = _collect_parquet_paths_from_dir(raw_dir) if raw_dir.exists() else []
    if not parquet_paths:
        print(f"未在 {raw_dir} 发现 parquet，开始从 {args.repo_id} 下载……")
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/SEED-Bench")
        parquet_paths = _collect_parquet_paths_from_dir(raw_dir)

    if not parquet_paths:
        raise SystemExit(f"在 {raw_dir} 下仍未找到任何 parquet 文件")

    # 全量处理（可能耗时较长）
    records = rows_from_parquets(parquet_paths, limit=None)
    items = normalize_seed_records(
        records, images_subdir="images", write_local_images=(not args.no_write_images), out_root=out_root
    )
    chat_path = save_chat_json(items, out_root)

    # 输出处理结果（缩进打印，便于查看）
    print(json.dumps({
        "raw_dir": str(Path(raw_dir).resolve()),
        "out_root": str(out_root.resolve()),
        "chat_json": str(chat_path.resolve()),
        "num_items": len(items),
        "example": items[0] if items else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
