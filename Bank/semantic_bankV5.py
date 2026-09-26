import os
import re
import json
import math
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from custom_llava_10_bankV5 import CustomLlavaForConditionalGeneration, CustomLlavaProcessor


logger = logging.getLogger(__name__)


STOPWORDS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "image",
    "picture",
    "photo",
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "does",
    "do",
    "did",
    "there",
    "it",
    "they",
    "them",
    "he",
    "she",
    "his",
    "her",
    "their",
    "be",
    "as",
    "at",
    "from",
}


RELATION_TYPES = ["left_of", "right_of", "above", "below", "overlap", "near"]


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

    # ===== bank3: layout =====
    layout_grid_size: int = 4
    num_layout_tokens: int = 6
    layout_min_count_per_cell: int = 24
    layout_max_prototypes_per_cell: int = 4

    # ===== bank4: relation =====
    num_relation_tokens: int = 6
    relation_min_count_per_type: int = 24
    relation_max_prototypes_per_type: int = 8
    relation_max_pairs_per_image: int = 64
    relation_near_threshold: float = 0.22
    relation_overlap_threshold: float = 0.10
    relation_direction_margin: float = 0.08

    # misc
    kmeans_iters: int = 40
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    dtype: str = "bfloat16"


class SemanticBankManager:
    """
    运行时加载和检索 bank。
    假设所有 bank 的 prototype 维度都与训练时 CustomLLaVA 的 image_features 维度一致，
    因而可以被直接拼接回视觉 token 序列。
    """

    def __init__(self, bank_path: str, device: Optional[torch.device] = None):
        self.bank_path = bank_path
        self.device = device or torch.device("cpu")

        payload = torch.load(bank_path, map_location="cpu")
        self.config = payload.get("config", {})
        self.stats = payload.get("stats", {})
        self.global_bank = payload.get("global_bank", {})
        self.entity_bank = payload.get("entity_bank", {})
        self.layout_bank = payload.get("layout_bank", {})
        self.relation_bank = payload.get("relation_bank", {})

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

        layout_entries = self.layout_bank.get("entries", [])
        self.layout_cell_to_feats: Dict[int, torch.Tensor] = {}
        for item in layout_entries:
            feats = item.get("prototypes")
            cell_id = int(item["cell_id"])
            if isinstance(feats, torch.Tensor) and feats.numel() > 0:
                self.layout_cell_to_feats[cell_id] = F.normalize(feats.float(), dim=-1)

        relation_entries = self.relation_bank.get("entries", [])
        self.relation_type_to_feats: Dict[str, torch.Tensor] = {}
        for item in relation_entries:
            feats = item.get("prototypes")
            rel_type = str(item["relation"])
            if isinstance(feats, torch.Tensor) and feats.numel() > 0:
                self.relation_type_to_feats[rel_type] = F.normalize(
                    feats.float(), dim=-1
                )

        self.bank_dim = None
        if (
            isinstance(self.global_prototypes, torch.Tensor)
            and self.global_prototypes.numel() > 0
        ):
            self.bank_dim = int(self.global_prototypes.shape[-1])
        elif self.entity_prototypes is not None and self.entity_prototypes.numel() > 0:
            self.bank_dim = int(self.entity_prototypes.shape[-1])
        else:
            for bank_dict in (self.layout_cell_to_feats, self.relation_type_to_feats):
                for feats in bank_dict.values():
                    if isinstance(feats, torch.Tensor) and feats.numel() > 0:
                        self.bank_dim = int(feats.shape[-1])
                        break
                if self.bank_dim is not None:
                    break

    def _check_dim(self, x: torch.Tensor, name: str):
        if self.bank_dim is not None and x.shape[-1] != self.bank_dim:
            raise ValueError(
                f"{name} dim mismatch: input={x.shape[-1]}, bank dim={self.bank_dim}"
            )

    def retrieve_global(
        self, patch_features: torch.Tensor, topk: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        if self.global_prototypes is None or self.global_prototypes.numel() == 0:
            return None

        patch_features = patch_features.float()
        self._check_dim(patch_features, "global bank")

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
        self._check_dim(local_features, "entity bank")

        topk = topk or int(self.config.get("num_entity_tokens", 6))
        queries = F.normalize(local_features, dim=-1)
        bank = self.entity_prototypes.to(queries.device)

        filtered_bank = bank
        matched_entity_names = _entities_from_text(query_text or "")
        if matched_entity_names:
            keep = [
                i
                for i, name in enumerate(self.entity_names)
                if name in matched_entity_names
            ]
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

    def retrieve_layout(
        self,
        local_features: torch.Tensor,
        local_coords: torch.Tensor,
        topk_per_token: int = 1,
        unique_only: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        按 token 坐标路由到 layout cell，再在对应 cell 的 prototype 中检索。
        local_coords: [N, 2], 取值建议为 [0, 1] 范围的 (x, y)
        """
        if not self.layout_cell_to_feats:
            return None

        local_features = F.normalize(local_features.float(), dim=-1)
        local_coords = local_coords.float()
        self._check_dim(local_features, "layout bank")

        grid_size = int(self.config.get("layout_grid_size", 4))
        retrieved = []

        for feat, coord in zip(local_features, local_coords):
            cell_id = coord_to_cell_id(coord, grid_size)
            bank = self.layout_cell_to_feats.get(cell_id)
            if bank is None or bank.numel() == 0:
                continue
            bank = bank.to(local_features.device)
            score = feat.unsqueeze(0) @ bank.t()
            k = min(topk_per_token, bank.shape[0])
            idx = score.topk(k=k, dim=-1).indices[0]
            retrieved.append(bank.index_select(0, idx))

        if not retrieved:
            return None

        out = torch.cat(retrieved, dim=0)
        if unique_only:
            out = _unique_rows_by_similarity(out, sim_threshold=0.999)
        return out

    def retrieve_relation(
        self,
        local_features: torch.Tensor,
        local_coords: torch.Tensor,
        topk_per_pair: int = 1,
        max_pairs: Optional[int] = None,
        unique_only: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        基于当前 cluster token 两两配对，按几何伪标签选择 relation 子库，再检索对应 prototype。
        返回的 prototype 仍是与视觉 token 同维度的 relation prior token。
        """
        if not self.relation_type_to_feats:
            return None

        local_features = F.normalize(local_features.float(), dim=-1)
        local_coords = local_coords.float()
        self._check_dim(local_features, "relation bank")

        if local_features.shape[0] < 2:
            return None

        max_pairs = max_pairs or int(
            self.config.get("relation_max_pairs_per_image", 64)
        )
        relation_specs = build_relation_queries(
            local_features=local_features,
            local_coords=local_coords,
            near_threshold=float(self.config.get("relation_near_threshold", 0.22)),
            overlap_threshold=float(
                self.config.get("relation_overlap_threshold", 0.10)
            ),
            direction_margin=float(self.config.get("relation_direction_margin", 0.08)),
            max_pairs=max_pairs,
        )

        retrieved = []
        for rel_type, rel_query in relation_specs:
            bank = self.relation_type_to_feats.get(rel_type)
            if bank is None or bank.numel() == 0:
                continue
            bank = bank.to(local_features.device)
            score = rel_query.unsqueeze(0) @ bank.t()
            k = min(topk_per_pair, bank.shape[0])
            idx = score.topk(k=k, dim=-1).indices[0]
            retrieved.append(bank.index_select(0, idx))

        if not retrieved:
            return None

        out = torch.cat(retrieved, dim=0)
        if unique_only:
            out = _unique_rows_by_similarity(out, sim_threshold=0.999)
        return out


class SemanticBankBuilder:
    """
    重点：
    1. 不再单独加载 vision tower 的 AutoModel。
    2. 直接使用和 pretrain_10_reason.py 一样的 CustomLlavaForConditionalGeneration + CustomLlavaProcessor。
    3. 图像特征直接走 model.get_image_features(...)，保证维度与训练时 patch_tokens 一致。
    4. bank3/layout 与 bank4/relation 在构建期离线聚合成 prototype，运行时可按位置/关系检索。
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
        vision_feature_select_strategy = getattr(
            model.config, "vision_feature_select_strategy", "default"
        )

        image_features = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )

        if isinstance(image_features, list):
            if len(image_features) != 1:
                raise ValueError(
                    f"Unexpected image_features list length: {len(image_features)}"
                )
            image_features = image_features[0]

        if image_features.dim() != 3 or image_features.shape[0] != 1:
            raise ValueError(
                f"Unexpected image_features shape: {tuple(image_features.shape)}"
            )

        patch = image_features[0].float().cpu()
        patch = F.normalize(patch, dim=-1)

        # 默认去掉前 5 个 patch，和模型对齐
        if self.config.patch_drop > 0 and patch.shape[0] > self.config.patch_drop:
            # patch = patch[self.config.patch_drop:]
            # BUG 不用去掉前5个patch，因为用的模型是 custom_llava_10，反而应该去掉后10个patch
            patch = patch[: -self.config.patch_drop]

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
        layout_vectors: Dict[int, List[torch.Tensor]] = defaultdict(list)
        relation_vectors: Dict[str, List[torch.Tensor]] = defaultdict(list)

        processed = 0
        skipped = 0
        feature_dim = None
        patch_token_count_hist = defaultdict(int)

        for idx, sample in enumerate(samples):
            if max_samples is not None and processed >= max_samples:
                break

            image_path = _resolve_image_path(
                sample, images_dir, self.config.image_extensions
            )
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

            # ===== cluster tokens + coords =====
            cluster_tokens, cluster_coords = build_cluster_tokens_with_coords(
                patch_features=patch,
                num_clusters=self.config.num_cluster_tokens,
                max_iter=self.config.kmeans_iters,
            )

            # ===== entity bank =====
            entities = extract_entities_from_sample(sample)
            if entities:
                pooled_local = cluster_tokens.mean(dim=0, keepdim=True)
                for ent in entities:
                    entity_vectors[ent].append(pooled_local)

            # ===== layout bank =====
            cell_ids = coords_to_cell_ids(cluster_coords, self.config.layout_grid_size)
            for feat, cell_id in zip(cluster_tokens, cell_ids.tolist()):
                layout_vectors[int(cell_id)].append(feat.unsqueeze(0))

            # ===== relation bank =====
            relation_specs = build_relation_queries(
                local_features=cluster_tokens,
                local_coords=cluster_coords,
                near_threshold=self.config.relation_near_threshold,
                overlap_threshold=self.config.relation_overlap_threshold,
                direction_margin=self.config.relation_direction_margin,
                max_pairs=self.config.relation_max_pairs_per_image,
            )
            for rel_type, rel_feat in relation_specs:
                relation_vectors[rel_type].append(rel_feat.unsqueeze(0))

            processed += 1
            if processed % 100 == 0:
                logger.info(
                    "[bank] processed=%d skipped=%d dim=%s entity=%d layout_cells=%d relation_types=%d",
                    processed,
                    skipped,
                    feature_dim,
                    len(entity_vectors),
                    len(layout_vectors),
                    len(relation_vectors),
                )

        if not global_vectors:
            raise RuntimeError(f"No usable samples found under: {dataset_root}")

        global_matrix = F.normalize(torch.cat(global_vectors, dim=0).float(), dim=-1)

        global_prototypes = _cluster_or_pad(
            global_matrix,
            k=self.config.num_global_prototypes,
            max_iter=self.config.kmeans_iters,
        )

        scored_entities = sorted(
            entity_vectors.items(), key=lambda kv: len(kv[1]), reverse=True
        )
        scored_entities = [
            (k, v) for k, v in scored_entities if len(v) >= self.config.entity_min_count
        ]
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

        layout_entries = []
        for cell_id, feats in sorted(layout_vectors.items(), key=lambda kv: kv[0]):
            if len(feats) < self.config.layout_min_count_per_cell:
                continue
            matrix = F.normalize(torch.cat(feats, dim=0).float(), dim=-1)
            prototypes = _cluster_or_pad(
                matrix,
                k=min(self.config.layout_max_prototypes_per_cell, matrix.shape[0]),
                max_iter=self.config.kmeans_iters,
            )
            layout_entries.append(
                {
                    "cell_id": int(cell_id),
                    "count": int(matrix.shape[0]),
                    "prototypes": prototypes.cpu(),
                }
            )

        relation_entries = []
        for rel_type in RELATION_TYPES:
            feats = relation_vectors.get(rel_type, [])
            if len(feats) < self.config.relation_min_count_per_type:
                continue
            matrix = torch.cat(feats, dim=0)
            prototypes = _cluster_or_pad(
                matrix,
                k=min(self.config.relation_max_prototypes_per_type, matrix.shape[0]),
                max_iter=self.config.kmeans_iters,
            )
            relation_entries.append(
                {
                    "relation": rel_type,
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
                "layout_cell_count": len(layout_entries),
                "relation_type_count": len(relation_entries),
                "feature_dim": int(global_matrix.shape[-1]),
                "patch_drop": int(self.config.patch_drop),
                "layout_grid_size": int(self.config.layout_grid_size),
                "patch_token_histogram": dict(
                    sorted(patch_token_count_hist.items(), key=lambda x: x[0])
                ),
                "model_name_or_path": model_name_or_path,
            },
            "global_bank": {
                "prototypes": global_prototypes.cpu(),
            },
            "entity_bank": {
                "entries": entity_entries,
            },
            "layout_bank": {
                "entries": layout_entries,
            },
            "relation_bank": {
                "entries": relation_entries,
            },
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(payload, output_path)
        logger.info("semantic bank saved to: %s", output_path)
        logger.info("feature_dim=%s", payload["stats"]["feature_dim"])
        logger.info(
            "global=%d entity=%d layout=%d relation=%d",
            payload["stats"]["global_prototype_count"],
            len(entity_entries),
            len(layout_entries),
            len(relation_entries),
        )
        return output_path


def build_cluster_tokens(
    patch_features: torch.Tensor, num_clusters: int, max_iter: int = 40
) -> torch.Tensor:
    cluster_tokens, _ = build_cluster_tokens_with_coords(
        patch_features=patch_features,
        num_clusters=num_clusters,
        max_iter=max_iter,
    )
    return cluster_tokens


def build_cluster_tokens_with_coords(
    patch_features: torch.Tensor,
    num_clusters: int,
    max_iter: int = 40,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
        cluster_tokens: [K, D]
        cluster_coords: [K, 2], 为每个 cluster 在近似 2D patch 网格上的归一化中心坐标 (x, y)
    """
    patch_features = F.normalize(patch_features.float(), dim=-1)
    n = patch_features.shape[0]

    patch_coords = reconstruct_patch_grid_coords(n)

    if n <= num_clusters:
        return patch_features, patch_coords

    centers, labels = kmeans_pytorch(
        patch_features, num_clusters=num_clusters, max_iter=max_iter
    )

    out_feats = []
    out_coords = []
    for k in range(num_clusters):
        mask = labels == k
        if mask.any():
            out_feats.append(patch_features[mask].mean(dim=0, keepdim=True))
            out_coords.append(patch_coords[mask].mean(dim=0, keepdim=True))

    if not out_feats:
        out_feats = [patch_features.mean(dim=0, keepdim=True)]
        out_coords = [patch_coords.mean(dim=0, keepdim=True)]

    out_feats = torch.cat(out_feats, dim=0)
    out_coords = torch.cat(out_coords, dim=0)
    out_feats = F.normalize(out_feats, dim=-1)
    out_coords = out_coords.clamp(0.0, 1.0)
    return out_feats, out_coords


def reconstruct_patch_grid_coords(num_patches: int) -> torch.Tensor:
    """
    从 patch 数近似恢复 2D 网格坐标。
    优先找最接近 sqrt(num_patches) 的因子分解；若找不到较好分解，则退化为近似矩形。
    输出为 [num_patches, 2]，每个坐标为 [0, 1] 范围内的 cell center.
    """
    if num_patches <= 0:
        raise ValueError(f"num_patches must be positive, got {num_patches}")

    h, w = infer_grid_shape(num_patches)
    xs = []
    ys = []
    for idx in range(num_patches):
        y = idx // w
        x = idx % w
        y = min(y, h - 1)
        x = min(x, w - 1)
        xs.append((x + 0.5) / max(w, 1))
        ys.append((y + 0.5) / max(h, 1))
    return torch.tensor(list(zip(xs, ys)), dtype=torch.float32)


def infer_grid_shape(num_patches: int) -> Tuple[int, int]:
    root = int(math.sqrt(num_patches))
    best_h, best_w = 1, num_patches
    best_gap = abs(best_w - best_h)

    for h in range(1, root + 1):
        if num_patches % h == 0:
            w = num_patches // h
            gap = abs(w - h)
            if gap < best_gap:
                best_h, best_w, best_gap = h, w, gap

    if best_h == 1 and num_patches > 1:
        h = root
        w = math.ceil(num_patches / max(h, 1))
        best_h, best_w = h, w

    return best_h, best_w


def coord_to_cell_id(coord: torch.Tensor, grid_size: int) -> int:
    x = float(coord[0])
    y = float(coord[1])
    gx = min(max(int(x * grid_size), 0), grid_size - 1)
    gy = min(max(int(y * grid_size), 0), grid_size - 1)
    return gy * grid_size + gx


def coords_to_cell_ids(coords: torch.Tensor, grid_size: int) -> torch.Tensor:
    cell_ids = [coord_to_cell_id(coord, grid_size) for coord in coords]
    return torch.tensor(cell_ids, dtype=torch.long)


def build_relation_queries(
    local_features: torch.Tensor,
    local_coords: torch.Tensor,
    near_threshold: float = 0.22,
    overlap_threshold: float = 0.10,
    direction_margin: float = 0.08,
    max_pairs: int = 64,
) -> List[Tuple[str, torch.Tensor]]:
    """
    relation feature:
        仍保持视觉 token 维度，以便后续可直接作为额外 token 拼回去。
        具体做法：对 (fi, fj) 做方向敏感组合：
            rel_feat = normalize(0.5 * fi + 0.5 * fj + 0.25 * (fj - fi))
        几何关系类别通过 coords 单独决定，用于路由到不同 relation 子库。
    """
    # BUG 已经在聚类的时候做过归一化
    # local_features = F.normalize(local_features.float(), dim=-1)
    local_coords = local_coords.float()
    n = local_features.shape[0]

    if n < 2:
        return []

    pair_candidates = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ci = local_coords[i]
            cj = local_coords[j]
            rel_type = classify_relation(
                src_coord=ci,
                dst_coord=cj,
                near_threshold=near_threshold,
                overlap_threshold=overlap_threshold,
                direction_margin=direction_margin,
            )
            if rel_type is None:
                continue

            fi = local_features[i]
            fj = local_features[j]
            rel_feat = 0.5 * fi + 0.5 * fj + 0.25 * (fj - fi)
            rel_feat = F.normalize(rel_feat.unsqueeze(0), dim=-1)[0]

            dist = torch.norm(cj - ci, p=2).item()
            priority = relation_priority(rel_type, dist)
            pair_candidates.append((priority, rel_type, rel_feat))

    if not pair_candidates:
        return []

    pair_candidates.sort(key=lambda x: x[0], reverse=True)
    pair_candidates = pair_candidates[:max_pairs]
    return [(rel_type, rel_feat) for _, rel_type, rel_feat in pair_candidates]


def classify_relation(
    src_coord: torch.Tensor,
    dst_coord: torch.Tensor,
    near_threshold: float = 0.22,
    overlap_threshold: float = 0.10,
    direction_margin: float = 0.08,
) -> Optional[str]:
    dx = float(dst_coord[0] - src_coord[0])
    dy = float(dst_coord[1] - src_coord[1])

    if abs(dx) <= overlap_threshold and abs(dy) <= overlap_threshold:
        return "overlap"

    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= near_threshold:
        return "near"

    if abs(dx) >= abs(dy):
        if dx > direction_margin:
            return "right_of"
        if dx < -direction_margin:
            return "left_of"
    else:
        # y 轴向下增大，因此 dst y 更小表示 above
        if dy < -direction_margin:
            return "above"
        if dy > direction_margin:
            return "below"

    return None


def relation_priority(rel_type: str, dist: float) -> float:
    type_bias = {
        "overlap": 3.0,
        "near": 2.5,
        "left_of": 2.0,
        "right_of": 2.0,
        "above": 2.0,
        "below": 2.0,
    }.get(rel_type, 1.0)
    return type_bias - dist


def _unique_rows_by_similarity(
    x: torch.Tensor, sim_threshold: float = 0.999
) -> torch.Tensor:
    if x.shape[0] <= 1:
        return x
    x = F.normalize(x.float(), dim=-1)
    keep = []
    for i in range(x.shape[0]):
        if not keep:
            keep.append(i)
            continue
        sims = x[i].unsqueeze(0) @ x[keep].t()
        if float(sims.max()) < sim_threshold:
            keep.append(i)
    return x.index_select(0, torch.tensor(keep, dtype=torch.long, device=x.device))


def kmeans_pytorch(
    X: torch.Tensor, num_clusters: int, max_iter: int = 40, tol: float = 1e-5
):
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
        return [
            _normalize_entity(x) for x in sample["entities"] if _normalize_entity(x)
        ]

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
    # BUG 假设matrix都是归一化后传入的
    # matrix = F.normalize(matrix.float(), dim=-1)
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


def _resolve_image_path(
    sample: Dict[str, Any], images_dir: str, exts: Tuple[str, ...]
) -> Optional[str]:
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
        description="Build semantic bank with global/entity/layout/relation memory using the SAME visual pipeline as pretrain_10_reason.py"
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="数据根目录，内部需包含 chat.json 和 images/")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="CustomLLaVA 模型路径")
    parser.add_argument("--output", type=str, required=True, help="输出 pt 文件路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--patch_drop", type=int, default=10)
    parser.add_argument("--num_cluster_tokens", type=int, default=10)

    parser.add_argument("--num_global_prototypes", type=int, default=32)
    parser.add_argument("--global_topk", type=int, default=4)

    parser.add_argument("--num_entity_tokens", type=int, default=6)
    parser.add_argument("--entity_min_count", type=int, default=8)
    parser.add_argument("--entity_max_entities", type=int, default=2000)
    parser.add_argument("--entity_max_prototypes_per_entity", type=int, default=4)

    parser.add_argument("--layout_grid_size", type=int, default=4)
    parser.add_argument("--num_layout_tokens", type=int, default=6)
    parser.add_argument("--layout_min_count_per_cell", type=int, default=24)
    parser.add_argument("--layout_max_prototypes_per_cell", type=int, default=4)

    parser.add_argument("--num_relation_tokens", type=int, default=6)
    parser.add_argument("--relation_min_count_per_type", type=int, default=24)
    parser.add_argument("--relation_max_prototypes_per_type", type=int, default=8)
    parser.add_argument("--relation_max_pairs_per_image", type=int, default=64)
    parser.add_argument("--relation_near_threshold", type=float, default=0.22)
    parser.add_argument("--relation_overlap_threshold", type=float, default=0.10)
    parser.add_argument("--relation_direction_margin", type=float, default=0.08)

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
        layout_grid_size=args.layout_grid_size,
        num_layout_tokens=args.num_layout_tokens,
        layout_min_count_per_cell=args.layout_min_count_per_cell,
        layout_max_prototypes_per_cell=args.layout_max_prototypes_per_cell,
        num_relation_tokens=args.num_relation_tokens,
        relation_min_count_per_type=args.relation_min_count_per_type,
        relation_max_prototypes_per_type=args.relation_max_prototypes_per_type,
        relation_max_pairs_per_image=args.relation_max_pairs_per_image,
        relation_near_threshold=args.relation_near_threshold,
        relation_overlap_threshold=args.relation_overlap_threshold,
        relation_direction_margin=args.relation_direction_margin,
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
