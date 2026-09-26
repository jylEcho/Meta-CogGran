import os
import re
import json
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from custom_llava_10_bankV3 import CustomLlavaForConditionalGeneration, CustomLlavaProcessor
# from custom_llava_10 import CustomLlavaForConditionalGeneration, CustomLlavaProcessor


logger = logging.getLogger(__name__)


STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "image", "picture", "photo",
    "what", "which", "who", "where", "when", "why", "how", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "and", "or", "does", "do", "did", "there",
    "it", "they", "them", "he", "she", "his", "her", "their", "be", "as", "at", "from",
}


@dataclass
class BankConfig:
    # 和模型对齐：默认去掉前 5 个 patch
    patch_drop: int = 5

    # 局部聚类 token 数
    num_cluster_tokens: int = 10

    # global bank 中 prototype 的总数量（构建时用）
    num_global_prototypes: int = 32

    # 检索 global 时返回多少个（推理时用）
    global_topk: int = 4

    # 检索 entity 时返回多少个
    num_entity_tokens: int = 6

    entity_min_count: int = 8
    entity_max_entities: int = 2000
    entity_max_prototypes_per_entity: int = 4
    kmeans_iters: int = 40
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    dtype: str = "bfloat16"


class SemanticBankManager:
    """
    运行时加载和检索 bank。
    这里假设 bank 中的特征维度与训练时 CustomLLaVA 的 image_features 维度完全一致。
    """

    def __init__(self, bank_path: str, device: Optional[torch.device] = None):
        self.bank_path = bank_path
        self.device = device or torch.device("cpu")

        payload = torch.load(bank_path, map_location="cpu")
        self.config = payload.get("config", {})
        self.stats = payload.get("stats", {})
        self.global_bank = payload.get("global_bank", {})
        self.entity_bank = payload.get("entity_bank", {})

        self.global_prototypes = self.global_bank.get("prototypes")
        if isinstance(self.global_prototypes, torch.Tensor):
            self.global_prototypes = F.normalize(self.global_prototypes.float(), dim=-1)

        entity_entries = self.entity_bank.get("entries", [])
        flat_feats = []
        flat_names = []
        for item in entity_entries:
            feats = item.get("prototypes")
            if isinstance(feats, torch.Tensor) and feats.numel() > 0:
                feats = F.normalize(feats.float(), dim=-1)
                flat_feats.append(feats)
                flat_names.extend([item["name"]] * feats.shape[0])

        if flat_feats:
            self.entity_prototypes = torch.cat(flat_feats, dim=0)
            self.entity_names = flat_names
        else:
            self.entity_prototypes = None
            self.entity_names = []

        self.bank_dim = None
        if isinstance(self.global_prototypes, torch.Tensor) and self.global_prototypes.numel() > 0:
            self.bank_dim = int(self.global_prototypes.shape[-1])
        elif self.entity_prototypes is not None and self.entity_prototypes.numel() > 0:
            self.bank_dim = int(self.entity_prototypes.shape[-1])

    def retrieve_global(self, patch_features: torch.Tensor, topk: Optional[int] = None) -> Optional[torch.Tensor]:
        if self.global_prototypes is None or self.global_prototypes.numel() == 0:
            return None

        patch_features = patch_features.float()
        if self.bank_dim is not None and patch_features.shape[-1] != self.bank_dim:
            raise ValueError(
                f"global bank dim mismatch: patch_tokens dim={patch_features.shape[-1]}, bank dim={self.bank_dim}"
            )

        # 默认使用 global_topk，而不是 prototype 总数
        topk = topk or int(self.config.get("global_topk", 4))
        global_query = F.normalize(patch_features.mean(dim=0, keepdim=True), dim=-1)
        bank = self.global_prototypes.to(global_query.device)

        scores = global_query @ bank.t()
        k = min(topk, bank.shape[0])
        idx = scores.topk(k=k, dim=-1).indices[0]
        return bank[idx]

    def retrieve_entity(
        self,
        local_features: torch.Tensor,
        topk: Optional[int] = None,
        query_text: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        if self.entity_prototypes is None or self.entity_prototypes.numel() == 0:
            return None

        local_features = local_features.float()
        if self.bank_dim is not None and local_features.shape[-1] != self.bank_dim:
            raise ValueError(
                f"entity bank dim mismatch: local_features dim={local_features.shape[-1]}, bank dim={self.bank_dim}"
            )

        topk = topk or int(self.config.get("num_entity_tokens", 6))
        queries = F.normalize(local_features, dim=-1)
        bank = self.entity_prototypes.to(queries.device)

        filtered_bank = bank
        matched_entity_names = _entities_from_text(query_text or "")
        if matched_entity_names:
            keep = [i for i, name in enumerate(self.entity_names) if name in matched_entity_names]
            if keep:
                keep = torch.tensor(keep, dtype=torch.long, device=queries.device)
                filtered_bank = bank.index_select(0, keep)

        if filtered_bank.numel() == 0:
            return None

        scores = queries @ filtered_bank.t()
        best_scores, _ = scores.max(dim=0)
        k = min(topk, filtered_bank.shape[0])
        picked = best_scores.topk(k=k).indices
        selected = filtered_bank.index_select(0, picked)
        return selected


class SemanticBankBuilder:
    """
    重点：
    1. 不再单独加载 vision tower 的 AutoModel。
    2. 直接使用和 pretrain_10_reason.py 一样的 CustomLlavaForConditionalGeneration + CustomLlavaProcessor。
    3. 图像特征直接走 model.get_image_features(...)，保证维度与训练时 patch_tokens 一致。
    """

    def __init__(self, config: Optional[BankConfig] = None):
        self.config = config or BankConfig()

    def _get_dtype(self) -> torch.dtype:
        dtype_name = str(self.config.dtype).lower()
        if dtype_name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype_name in ("fp16", "float16", "half"):
            return torch.float16
        return torch.float32

    def _load_model_processor(self, model_name_or_path: str, device: str):
        model = CustomLlavaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=self._get_dtype(),
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = CustomLlavaProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        model = model.to(device)
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = False
        return model, processor

    @torch.no_grad()
    def _encode_image(self, model, processor, image, device: str) -> torch.Tensor:
        """
        返回 projected patch tokens，维度与训练时保持一致。
        兼容 processor.image_processor 可能返回 list 的情况。
        """
        image_inputs = processor.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]

        if isinstance(pixel_values, list):
            if len(pixel_values) == 0:
                raise ValueError("image_processor returned empty pixel_values list")

            if len(pixel_values) == 1:
                pixel_values = pixel_values[0]
            else:
                first_shape = tuple(pixel_values[0].shape)
                if not all(tuple(x.shape) == first_shape for x in pixel_values):
                    raise ValueError(
                        f"pixel_values is a list with inconsistent shapes: "
                        f"{[tuple(x.shape) for x in pixel_values]}"
                    )
                pixel_values = torch.stack(pixel_values, dim=0)

        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"pixel_values must be Tensor, got {type(pixel_values)}")

        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        pixel_values = pixel_values.to(device)

        vision_feature_layer = getattr(model.config, "vision_feature_layer", -2)
        vision_feature_select_strategy = getattr(model.config, "vision_feature_select_strategy", "default")

        image_features = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )

        if isinstance(image_features, list):
            if len(image_features) != 1:
                raise ValueError(f"Unexpected image_features list length: {len(image_features)}")
            image_features = image_features[0]

        if image_features.dim() != 3 or image_features.shape[0] != 1:
            raise ValueError(f"Unexpected image_features shape: {tuple(image_features.shape)}")

        patch = image_features[0].float().cpu()
        patch = F.normalize(patch, dim=-1)

        # 默认去掉前 5 个 patch，和模型对齐
        if self.config.patch_drop > 0 and patch.shape[0] > self.config.patch_drop:
            patch = patch[self.config.patch_drop:]

        return patch

    def build(
        self,
        dataset_root: str,
        output_path: str,
        model_name_or_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_samples: Optional[int] = None,
    ) -> str:
        from PIL import Image

        model, processor = self._load_model_processor(model_name_or_path, device)

        chat_path = os.path.join(dataset_root, "chat.json")
        images_dir = os.path.join(dataset_root, "images")
        samples = _load_chat_samples(chat_path)

        global_vectors: List[torch.Tensor] = []
        entity_vectors: Dict[str, List[torch.Tensor]] = defaultdict(list)

        processed = 0
        skipped = 0
        feature_dim = None
        patch_token_count_hist = defaultdict(int)

        for idx, sample in enumerate(samples):
            if max_samples is not None and processed >= max_samples:
                break

            image_path = _resolve_image_path(sample, images_dir, self.config.image_extensions)
            if image_path is None or not os.path.exists(image_path):
                skipped += 1
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.warning("open image failed: %s (%s)", image_path, e)
                skipped += 1
                continue

            try:
                patch = self._encode_image(model, processor, image, device)
            except Exception as e:
                logger.warning("encode image failed: %s (%s)", image_path, e)
                skipped += 1
                continue

            if patch.numel() == 0:
                skipped += 1
                continue

            if feature_dim is None:
                feature_dim = int(patch.shape[-1])
            elif int(patch.shape[-1]) != feature_dim:
                raise ValueError(
                    f"Inconsistent feature dim detected: current={patch.shape[-1]}, expected={feature_dim}, "
                    f"image={image_path}"
                )

            patch_token_count_hist[int(patch.shape[0])] += 1

            # 每张图一个 global 向量
            global_vectors.append(patch.mean(dim=0, keepdim=True))

            cluster_tokens = build_cluster_tokens(
                patch,
                num_clusters=self.config.num_cluster_tokens,
                max_iter=self.config.kmeans_iters,
            )

            entities = extract_entities_from_sample(sample)
            if entities:
                pooled_local = cluster_tokens.mean(dim=0, keepdim=True)
                for ent in entities:
                    entity_vectors[ent].append(pooled_local)

            processed += 1
            if processed % 100 == 0:
                logger.info("[bank] processed=%d skipped=%d dim=%s", processed, skipped, feature_dim)

        if not global_vectors:
            raise RuntimeError(f"No usable samples found under: {dataset_root}")

        global_matrix = torch.cat(global_vectors, dim=0)

        # 这里改成 num_global_prototypes，表示 global bank 总 prototype 数量
        global_prototypes = _cluster_or_pad(
            global_matrix,
            k=self.config.num_global_prototypes,
            max_iter=self.config.kmeans_iters,
        )

        scored_entities = sorted(entity_vectors.items(), key=lambda kv: len(kv[1]), reverse=True)
        scored_entities = [(k, v) for k, v in scored_entities if len(v) >= self.config.entity_min_count]
        scored_entities = scored_entities[: self.config.entity_max_entities]

        entity_entries = []
        for name, feats in scored_entities:
            matrix = torch.cat(feats, dim=0)
            prototypes = _cluster_or_pad(
                matrix,
                k=min(self.config.entity_max_prototypes_per_entity, matrix.shape[0]),
                max_iter=self.config.kmeans_iters,
            )
            entity_entries.append(
                {
                    "name": name,
                    "count": int(matrix.shape[0]),
                    "prototypes": prototypes.cpu(),
                }
            )

        payload = {
            "config": asdict(self.config),
            "stats": {
                "processed_samples": processed,
                "skipped_samples": skipped,
                "global_vectors": int(global_matrix.shape[0]),
                "global_prototype_count": int(global_prototypes.shape[0]),
                "global_topk": int(self.config.global_topk),
                "entity_count": len(entity_entries),
                "feature_dim": int(global_matrix.shape[-1]),
                "patch_drop": int(self.config.patch_drop),
                "patch_token_histogram": dict(sorted(patch_token_count_hist.items(), key=lambda x: x[0])),
                "model_name_or_path": model_name_or_path,
            },
            "global_bank": {
                "prototypes": global_prototypes.cpu(),
            },
            "entity_bank": {
                "entries": entity_entries,
            },
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(payload, output_path)
        logger.info("semantic bank saved to: %s", output_path)
        logger.info("feature_dim=%s", payload["stats"]["feature_dim"])
        logger.info(
            "global_prototype_count=%d, global_topk=%d, patch_drop=%d",
            payload["stats"]["global_prototype_count"],
            payload["stats"]["global_topk"],
            payload["stats"]["patch_drop"],
        )
        return output_path


def build_cluster_tokens(patch_features: torch.Tensor, num_clusters: int, max_iter: int = 40) -> torch.Tensor:
    patch_features = F.normalize(patch_features.float(), dim=-1)
    n = patch_features.shape[0]
    if n <= num_clusters:
        return patch_features

    centers, labels = kmeans_pytorch(patch_features, num_clusters=num_clusters, max_iter=max_iter)
    out = []
    for k in range(num_clusters):
        mask = labels == k
        if mask.any():
            out.append(patch_features[mask].mean(dim=0, keepdim=True))

    if not out:
        out = [patch_features.mean(dim=0, keepdim=True)]

    out = torch.cat(out, dim=0)
    return F.normalize(out, dim=-1)


def kmeans_pytorch(X: torch.Tensor, num_clusters: int, max_iter: int = 40, tol: float = 1e-5):
    X = X.float()
    n, d = X.shape
    if n <= num_clusters:
        labels = torch.arange(n, device=X.device)
        return X, labels

    with torch.no_grad():
        perm = torch.randperm(n, device=X.device)[:num_clusters]
        centers = X.index_select(0, perm)

        for _ in range(max_iter):
            dists = torch.cdist(X, centers)
            labels = dists.argmin(dim=1)

            new_centers = []
            for i in range(num_clusters):
                mask = labels == i
                if mask.any():
                    new_centers.append(X[mask].mean(dim=0))
                else:
                    new_centers.append(centers[i])

            new_centers = torch.stack(new_centers, dim=0)
            if torch.allclose(new_centers, centers, atol=tol, rtol=0):
                centers = new_centers
                break
            centers = new_centers

    return centers, labels


def extract_entities_from_sample(sample: Dict[str, Any]) -> List[str]:
    if isinstance(sample.get("entities"), list):
        return [_normalize_entity(x) for x in sample["entities"] if _normalize_entity(x)]

    texts = []

    for key in ("conversations", "messages", "chat", "dialog"):
        if key in sample and isinstance(sample[key], list):
            for item in sample[key]:
                if isinstance(item, dict):
                    for field in ("value", "content", "text"):
                        if field in item and isinstance(item[field], str):
                            texts.append(item[field])
                elif isinstance(item, str):
                    texts.append(item)

    if not texts:
        for _, v in sample.items():
            if isinstance(v, str):
                texts.append(v)

    merged = "\n".join(texts)
    return _entities_from_text(merged)


def _entities_from_text(text: str) -> List[str]:
    text = text or ""
    entities = set()

    for quoted in re.findall(r'"([^"]{2,40})"|“([^”]{2,40})”|\'([^\']{2,40})\'', text):
        for part in quoted:
            ent = _normalize_entity(part)
            if ent:
                entities.add(ent)

    for tok in re.findall(r"[A-Za-z][A-Za-z\-]{2,30}", text.lower()):
        ent = _normalize_entity(tok)
        if ent and ent not in STOPWORDS:
            entities.add(ent)

    for tok in re.findall(r"[\u4e00-\u9fff]{2,12}", text):
        ent = _normalize_entity(tok)
        if ent and ent not in STOPWORDS:
            entities.add(ent)

    return sorted(entities)


def _normalize_entity(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    text = re.sub(r"[^\w\-\u4e00-\u9fff ]+", "", text)
    if len(text) < 2:
        return ""
    if text in STOPWORDS:
        return ""
    return text


def _cluster_or_pad(matrix: torch.Tensor, k: int, max_iter: int) -> torch.Tensor:
    matrix = F.normalize(matrix.float(), dim=-1)
    if matrix.shape[0] == 0:
        raise ValueError("matrix is empty")

    if matrix.shape[0] <= k:
        if matrix.shape[0] < k:
            repeat_idx = torch.randint(0, matrix.shape[0], (k - matrix.shape[0],))
            repeat = matrix.index_select(0, repeat_idx)
            matrix = torch.cat([matrix, repeat], dim=0)
        return F.normalize(matrix, dim=-1)

    centers, _ = kmeans_pytorch(matrix, num_clusters=k, max_iter=max_iter)
    return F.normalize(centers, dim=-1)


def _load_chat_samples(chat_path: str) -> List[Dict[str, Any]]:
    with open(chat_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("data", "items", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]

    if isinstance(data, list):
        return data

    raise TypeError(f"Unsupported chat.json format: {type(data)}")


def _resolve_image_path(sample: Dict[str, Any], images_dir: str, exts: Tuple[str, ...]) -> Optional[str]:
    candidates = []

    for key in ("image", "image_path", "img", "file_name", "filename"):
        if key in sample and isinstance(sample[key], str):
            candidates.append(sample[key])

    if not candidates and isinstance(sample.get("images"), list) and sample["images"]:
        first = sample["images"][0]
        if isinstance(first, str):
            candidates.append(first)
        elif isinstance(first, dict):
            for key in ("path", "image", "file_name", "filename"):
                if key in first and isinstance(first[key], str):
                    candidates.append(first[key])

    for cand in candidates:
        cand = os.path.basename(cand)
        base, ext = os.path.splitext(cand)

        if ext:
            p = os.path.join(images_dir, cand)
            if os.path.exists(p):
                return p
        else:
            for e in exts:
                p = os.path.join(images_dir, base + e)
                if os.path.exists(p):
                    return p

    return None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Build semantic bank with the SAME visual pipeline as pretrain_10_reason.py"
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="数据根目录，内部需包含 chat.json 和 images/")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="CustomLLaVA 模型路径")
    parser.add_argument("--output", type=str, required=True, help="输出 pt 文件路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None)

    # 默认去掉前 5 个 patch
    parser.add_argument("--patch_drop", type=int, default=5)

    parser.add_argument("--num_cluster_tokens", type=int, default=10)

    # 拆分：global prototype 总数 vs retrieve topk
    parser.add_argument("--num_global_prototypes", type=int, default=32)
    parser.add_argument("--global_topk", type=int, default=4)

    parser.add_argument("--num_entity_tokens", type=int, default=6)
    parser.add_argument("--entity_min_count", type=int, default=8)
    parser.add_argument("--entity_max_entities", type=int, default=2000)
    parser.add_argument("--entity_max_prototypes_per_entity", type=int, default=4)
    parser.add_argument("--kmeans_iters", type=int, default=40)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    args = parser.parse_args()

    if args.num_global_prototypes < args.global_topk:
        raise ValueError(
            f"num_global_prototypes ({args.num_global_prototypes}) should be >= global_topk ({args.global_topk})"
        )

    cfg = BankConfig(
        patch_drop=args.patch_drop,
        num_cluster_tokens=args.num_cluster_tokens,
        num_global_prototypes=args.num_global_prototypes,
        global_topk=args.global_topk,
        num_entity_tokens=args.num_entity_tokens,
        entity_min_count=args.entity_min_count,
        entity_max_entities=args.entity_max_entities,
        entity_max_prototypes_per_entity=args.entity_max_prototypes_per_entity,
        kmeans_iters=args.kmeans_iters,
        dtype=args.dtype,
    )

    builder = SemanticBankBuilder(cfg)
    path = builder.build(
        dataset_root=args.dataset_root,
        output_path=args.output,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        max_samples=args.max_samples,
    )
    print(f"semantic bank saved to: {path}")