"""
OCR-VQA 数据预处理工具

功能说明：
- 将 OCR-VQA 的 parquet 文件转换为 LLaVA 格式的 chat.json
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
import os
os.environ["HF_HOME"]="./external/e1374390/HF_DIR"
os.environ["TRANSFORMERS_CACHE"]="./external/e1374390/HF_DIR/hub"
os.environ["HF_DATASETS_CACHE"]="./external/e1374390/HF_DIR/datasets"
os.environ["HF_MODULES_CACHE"]="./external/e1374390/HF_DIR/modules"
os.environ["HF_TOKEN"] = "REDACTED_TOKEN_USE_ENV"
os.environ["HF_ENDPOINT"] = "API_ENDPOINT_NOT_CONFIGURED"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 将 bytes 转换为 Base64 编码字符串
def bytes_to_base64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


# 确保问题以 \n<image> 结尾
def ensure_image_marker(q: str) -> str:
    q = q.rstrip("\n")
    if not q.endswith("<image>"):
        q = f"{q}\n<image>"
    return q


def extract_rel_image_name(image_field) -> Optional[str]:
    try:
        if not image_field:
            return None
        elem = image_field[0] if isinstance(image_field, (list, tuple)) else image_field

        # dict 常见键
        if isinstance(elem, dict):
            for key in ("path", "filename", "file_name", "name"):
                p = elem.get(key)
                if p:
                    base = os.path.basename(str(p))
                    if not base.lower().endswith((".jpg", ".jpeg", ".png")):
                        base = base + ".jpg"
                    return base
        # 如果直接是 bytes / bytearray，不能推断名字，返回 None 由调用方使用 id 作为文件名
        if isinstance(elem, (bytes, bytearray)):
            return None
    except Exception:
        pass
    return None


def maybe_write_local_image(image_field, images_dir: Path, rel_name: str) -> Optional[Path]:
    try:
        if image_field is None:
            return None
        elem = image_field[0] if isinstance(image_field, (list, tuple)) else image_field

        # 1) dict with bytes-like under common keys
        if isinstance(elem, dict):
            for key in ("bytes", "data", "image", "content"):
                b = elem.get(key)
                if isinstance(b, (bytes, bytearray)):
                    images_dir.mkdir(parents=True, exist_ok=True)
                    out_path = images_dir.joinpath(rel_name)
                    with open(out_path, "wb") as f:
                        f.write(b)
                    return out_path
            # some records may store raw bytes as first value in dict (rare)
            # try to find any bytes-like value
            for v in elem.values():
                if isinstance(v, (bytes, bytearray)):
                    images_dir.mkdir(parents=True, exist_ok=True)
                    out_path = images_dir.joinpath(rel_name)
                    with open(out_path, "wb") as f:
                        f.write(v)
                    return out_path

        # 2) element is raw bytes/bytearray
        if isinstance(elem, (bytes, bytearray)):
            images_dir.mkdir(parents=True, exist_ok=True)
            out_path = images_dir.joinpath(rel_name)
            with open(out_path, "wb") as f:
                f.write(elem)
            return out_path

        # 3) PIL Image -> save directly
        try:
            from PIL import Image as PILImage
            if isinstance(elem, PILImage.Image):
                images_dir.mkdir(parents=True, exist_ok=True)
                out_path = images_dir.joinpath(rel_name)
                # preserve format if available, fallback to JPEG
                fmt = getattr(elem, "format", None) or "JPEG"
                try:
                    elem.save(out_path, format=fmt)
                except Exception:
                    elem.save(out_path, format="JPEG")
                return out_path
        except Exception:
            pass

        # 4) numpy array -> try to save via PIL (if available)
        try:
            import numpy as np
            from PIL import Image
            if isinstance(elem, np.ndarray):
                images_dir.mkdir(parents=True, exist_ok=True)
                out_path = images_dir.joinpath(rel_name)
                img = Image.fromarray(elem)
                img.save(out_path)
                return out_path
        except Exception:
            pass

        # 无法识别的结构，打印调试信息一次
        print("maybe_write_local_image: unsupported image element type:", type(elem))
        # 如果 elem 可 repr，打印简短信息
        try:
            print("elem repr:", repr(elem)[:400])
        except Exception:
            pass

    except Exception as e:
        print("maybe_write_local_image error:", e)
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


# 将 howard-hou/OCR-VQA 原始数据转换为 LLaVA 格式的对话数据，可选择将图片 bytes 写入本地文件
def normalize_seed_records(records: List[Dict], images_subdir: str = "images", write_local_images: bool = False, out_root: Optional[Path] = None) -> List[Dict]:
    out: List[Dict] = []
    images_dir = out_root.joinpath(images_subdir) if (write_local_images and out_root) else None

    for row in tqdm(records, desc="Normalizing howard-hou/OCR-VQA records", unit="record"):

        rel_img_name = extract_rel_image_name(row.get("image"))
        if rel_img_name is None:
            did = str(row.get('image_id', ''))
            rel_img_name = f"{os.path.basename(did)}.jpg"

        # 写图片（如果需要）
        if images_dir is not None:
            maybe_write_local_image(row.get("image"), images_dir, rel_img_name)

        # 基础 id
        base_rid = str(row.get('image_id', ''))

        # 分支：如果存在 questions (list) 与 answers (list) -> 拆分
        questions_list = row.get("questions")
        answers_list = row.get("answers")
        if isinstance(questions_list, (list, tuple)) and isinstance(answers_list, (list, tuple)):
            for i, q_raw in enumerate(questions_list):
                q_text = str(q_raw).strip()
                q_text = ensure_image_marker(q_text)
                a_text = str(answers_list[i]) if i < len(answers_list) else ""
                rid = f"{base_rid}{i}"
                item = {
                    "id": rid,
                    "image": rel_img_name,
                    "conversations": [
                        {"from": "human", "value": q_text},
                        {"from": "gpt", "value": a_text},
                    ],
                }
                out.append(item)
            continue

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


# 新增：根据文件名或父目录推断 split，将 paths 分组为 train/validation/test（只返回非空组）
def group_parquet_paths_by_split(paths: Iterable[Path]) -> Dict[str, List[Path]]:
    groups = {"train": [], "validation": [], "test": []}
    for p in paths:
        name = p.name.lower()
        parent = str(p.parent).lower()
        assigned = False
        for token in ("train", "validation", "val", "test", "dev"):
            if token in name or token in parent:
                if token in ("validation", "val", "dev"):
                    groups["validation"].append(p)
                elif token == "test":
                    groups["test"].append(p)
                else:
                    groups["train"].append(p)
                assigned = True
                break
        if not assigned:
            groups["train"].append(p)
    return {k: v for k, v in groups.items() if v}


def main():
    parser = argparse.ArgumentParser(description="下载并将 howard-hou/OCR-VQA 规范化为 LLaVA 风格 chat.json")
    parser.add_argument("--repo-id", type=str, default="howard-hou/OCR-VQA", help="HuggingFace 数据集 repo id")
    parser.add_argument("--raw-dir", type=str, default="./external/e1374390/work/raw_data", help="原始数据下载/存放目录")
    parser.add_argument("--out-dir", type=str, default="./external/e1374390/work/dataset/OCR-VQA", help="输出数据集根目录（相对路径推荐，如 datasets/seed）")
    parser.add_argument("--no-write-images", action="store_true", help="不写出图片文件（仅保留相对路径名）")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_root = Path(args.out_dir)

    parquet_paths: List[Path] = _collect_parquet_paths_from_dir(raw_dir) if raw_dir.exists() else []
    summary: Dict[str, Dict] = {}

    # 优先使用本地 parquet（并按文件名/目录分 split）
    if parquet_paths:
        grouped = group_parquet_paths_by_split(parquet_paths)
        for split, paths in grouped.items():
            print(f"处理 split={split}, parquet files: {len(paths)}")
            records = rows_from_parquets(paths, limit=None)
            out_dir_for_split = out_root.joinpath(split)
            items = normalize_seed_records(
                records,
                images_subdir="images",
                write_local_images=(not args.no_write_images),
                out_root=out_dir_for_split,
            )
            chat_path = save_chat_json(items, out_dir_for_split)
            summary[split] = {"num_items": len(items), "chat_json": str(chat_path.resolve()), "example": items[0] if items else None}

    else:
        # 没有 parquet，尝试从 HuggingFace load_dataset（保留 split 信息）
        print(f"未在 {raw_dir} 发现 parquet，尝试直接从 {args.repo_id} 加载数据集（load_dataset）……")
        try:
            from datasets import load_dataset
            ds = load_dataset(args.repo_id)
            # 如果是 DatasetDict（含多个 split），按 split 保存；否则当作单一 split "all"
            if hasattr(ds, "items"):
                for split_name, d in ds.items():
                    print(f"加载 split {split_name}, size={len(d)}")
                    records = [dict(r) for r in d]
                    target_split = split_name if split_name in ("train", "validation", "test") else ("validation" if split_name == "dev" else split_name)
                    out_dir_for_split = out_root.joinpath(target_split)
                    items = normalize_seed_records(
                        records,
                        images_subdir="images",
                        write_local_images=(not args.no_write_images),
                        out_root=out_dir_for_split,
                    )
                    chat_path = save_chat_json(items, out_dir_for_split)
                    summary[target_split] = {"num_items": len(items), "chat_json": str(chat_path.resolve()), "example": items[0] if items else None}
            else:
                records = [dict(r) for r in ds]
                out_dir_for_split = out_root.joinpath("all")
                items = normalize_seed_records(
                    records,
                    images_subdir="images",
                    write_local_images=(not args.no_write_images),
                    out_root=out_dir_for_split,
                )
                chat_path = save_chat_json(items, out_dir_for_split)
                summary["all"] = {"num_items": len(items), "chat_json": str(chat_path.resolve()), "example": items[0] if items else None}
        except Exception as e:
            print("load_dataset 失败：", e)
            raise SystemExit(f"在 {raw_dir} 下未找到 parquet，且从 {args.repo_id} 加载失败")

    # 输出处理结果（缩进打印，便于查看）
    print(json.dumps({
        "raw_dir": str(Path(raw_dir).resolve()),
        "out_root": str(out_root.resolve()),
        "splits": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
