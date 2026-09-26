# ...existing code...
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any

import numpy as np
import pandas as pd


QUESTION_IMAGE_MARKER = "\n<image>"


def download_dataset(dest_dir: str, repo_id: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required for download; install it or skip download"
        ) from e

    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(repo_id=repo_id, local_dir=str(dest))
    return Path(snapshot_path)


def ensure_image_marker(q: str) -> str:
    q = (q or "").rstrip("\n")
    if not q.endswith("<image>"):
        q = f"{q}{QUESTION_IMAGE_MARKER}"
    return q


def rows_from_parquets(parquet_paths: Iterable[Path], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
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


def extract_rel_image_name_from_row(row: Dict[str, Any]) -> Optional[str]:
    # Try common fields: file_name, image path inside dict/list, data_id fallback
    if row.get("file_name"):
        return os.path.basename(str(row["file_name"]))
    img = row.get("image")
    try:
        if img is None:
            return None
        # image may be list/tuple or array-like; take first element if so
        if isinstance(img, (list, tuple)) and img:
            elem = img[0]
        else:
            # if it's dict-like with 'path' or has indexable 0
            elem = img[0] if hasattr(img, "__getitem__") else img
        if isinstance(elem, dict):
            p = elem.get("path") or elem.get("file_name")
            if p:
                base = os.path.basename(str(p))
                if not base.lower().endswith((".jpg", ".jpeg", ".png")):
                    base = base + ".jpg"
                return base
    except Exception:
        pass
    return None


def maybe_write_local_image_from_row(row: Dict[str, Any], images_dir: Path, rel_name: str) -> Optional[Path]:
    try:
        img = row.get("image")
        if img is None:
            return None
        if isinstance(img, (list, tuple)) and img:
            elem = img[0]
        else:
            elem = img[0] if hasattr(img, "__getitem__") else img
        if isinstance(elem, dict) and elem.get("bytes"):
            images_dir.mkdir(parents=True, exist_ok=True)
            out_path = images_dir.joinpath(rel_name)
            # elem["bytes"] may be numpy ndarray, bytes, or bytearray
            data = elem["bytes"]
            if isinstance(data, np.ndarray):
                # if 1D uint8, treat as raw encoded bytes
                if data.dtype == np.uint8 and data.ndim == 1:
                    b = data.tobytes()
                else:
                    # fallback: convert to bytes view
                    b = data.tobytes()
            elif isinstance(data, (bytes, bytearray)):
                b = bytes(data)
            else:
                # cannot write unknown type
                return None
            with open(out_path, "wb") as f:
                f.write(b)
            return out_path
    except Exception:
        return None
    return None


def normalize_records(records: List[Dict[str, Any]], images_subdir: str = "images", write_local_images: bool = False, out_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    images_dir = out_root.joinpath(images_subdir) if (write_local_images and out_root) else None

    for row in records:
        question = str(row.get("question", "")).strip()
        question = ensure_image_marker(question)

        answer = row.get("answer")
        # normalize answer to str; handle numpy arrays
        if isinstance(answer, np.ndarray):
            try:
                if answer.ndim == 0:
                    answer_val = str(answer.item())
                else:
                    answer_val = "; ".join(map(str, answer.tolist()))
            except Exception:
                answer_val = str(answer.tolist())
        elif isinstance(answer, (list, tuple)):
            answer_val = "; ".join(map(str, answer))
        else:
            answer_val = str(answer)

        rel_img = extract_rel_image_name_from_row(row)
        if rel_img is None:
            # fallback: use data_id or file-based id
            did = row.get("data_id") or row.get("id") or row.get("image_id") or row.get("file_name")
            rel_img = f"{os.path.basename(str(did))}.jpg" if did else "unknown.jpg"

        if images_dir is not None:
            maybe_write_local_image_from_row(row, images_dir, rel_img)

        # generate id
        if row.get("file_name"):
            rid = os.path.splitext(os.path.basename(str(row.get("file_name"))))[0]
        else:
            rid = f"{row.get('data_id','')}_{row.get('question_id','')}".strip("_")
            if not rid:
                rid = str(len(out))

        item = {
            "id": rid,
            "image": rel_img,
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer_val},
            ],
        }
        out.append(item)

    return out


def save_chat_json(items: List[Dict], out_root: Path, filename: str = "chat.json") -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root.joinpath(filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)
    return out_path


def _collect_parquet_paths_from_dir(root: Path) -> List[Path]:
    return sorted(root.rglob("*.parquet"))


def main():
    parser = argparse.ArgumentParser(description="下载并将 RefCOCO 规范化为 LLaVA 风格 chat.json")
    parser.add_argument("--repo-id", type=str, default="lmms-lab/RefCOCO", help="HuggingFace 数据集 repo id for RefCOCO")
    parser.add_argument("--raw-dir", type=str, default="datasets/RefCOCO", help="原始数据下载/存放目录")
    parser.add_argument("--out-dir", type=str, default="datasets/lmms-lab-RefCOCO", help="输出数据集根目录（相对路径推荐）")
    parser.add_argument("--no-write-images", action="store_true", help="不写出图片文件（仅保留相对路径名）")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_root = Path(args.out_dir)

    # collect parquet paths under raw_dir
    parquet_paths: List[Path] = _collect_parquet_paths_from_dir(raw_dir) if raw_dir.exists() else []

    # 如果没有发现 parquet，尝试下载 RefCOCO 到 raw_dir
    if not parquet_paths:
        print(f"未在 {raw_dir} 发现 parquet，开始从 {args.repo_id} 下载")
        raw_dir = download_dataset(dest_dir=str(raw_dir), repo_id=args.repo_id)
        parquet_paths = _collect_parquet_paths_from_dir(raw_dir)
            # 如果用户要求同时下载 RefCOCO，则在指定目录或默认位置处理

    if not parquet_paths:
        raise SystemExit(f"在 {raw_dir} 下仍未找到任何 parquet 文件")

    # 全量处理（可能耗时较长）
    records = rows_from_parquets(parquet_paths, limit=None)
    items = normalize_records(
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