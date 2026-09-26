from typing import Any, Optional, Union, Dict, List, Tuple
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.models.llava.modeling_llava import (
        LlavaModel as HF_LlavaModel,
        LlavaForConditionalGeneration as HF_LlavaForConditionalGeneration,
        LlavaPreTrainedModel,
    )
except Exception:
    from transformers import (
        LlavaModel as HF_LlavaModel,
        LlavaForConditionalGeneration as HF_LlavaForConditionalGeneration,
        LlavaPreTrainedModel,
    )

try:
    from transformers.models.llava.processing_llava import (
        LlavaProcessor as HF_LlavaProcessor,
        LlavaProcessorKwargs as HF_LlavaProcessorKwargs,
    )
except Exception:
    from transformers import LlavaProcessor as HF_LlavaProcessor
    HF_LlavaProcessorKwargs = None

try:
    from transformers.feature_extraction_utils import BatchFeature
except Exception:
    BatchFeature = dict

from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.processing_utils import ProcessingKwargs, Unpack

DEFAULT_NUM_CLUSTER_TOKENS = 10
DEFAULT_GLOBAL_BANK_TOPK = 4
DEFAULT_ENTITY_BANK_TOPK = 6
# DEFAULT_LAYOUT_BANK_TOPK = 64
DEFAULT_LAYOUT_BANK_TOPK = 2
# DEFAULT_RELATION_BANK_TOPK = 128
DEFAULT_RELATION_BANK_TOPK = 2
DEFAULT_LAYOUT_GRID_SIZE = 4
DEFAULT_RELATION_NEAR_THRESHOLD = 0.2
DEFAULT_RELATION_OVERLAP_THRESHOLD = 0.10
DEFAULT_RELATION_DIRECTION_MARGIN = 0.08
DEFAULT_RELATION_MAX_PAIRS = 64
DEFAULT_META_COG_ENABLED = False
DEFAULT_META_COG_NUM_STEPS = 4
DEFAULT_META_COG_MIN_STEPS = 1
DEFAULT_META_COG_STOP_THRESHOLD = 0.80
DEFAULT_META_COG_STABILITY_THRESHOLD = 0.015
DEFAULT_META_COG_EQUILIBRIUM_THRESHOLD = 0.16
DEFAULT_META_COG_ACTION_ENTROPY_WEIGHT = 0.05
DEFAULT_META_COG_CONTRADICTION_WEIGHT = 0.20
DEFAULT_META_COG_UNCERTAINTY_WEIGHT = 0.10
DEFAULT_META_COG_NUM_ACTIONS = 8
RELATION_TYPES = ["left_of", "right_of", "above", "below", "overlap", "near"]
META_COG_ACTIONS = [
    "inspect_global",
    "inspect_entity",
    "inspect_layout",
    "inspect_relation",
    "verify",
    "revise",
    "stop",
    "inspect_uncertainty",
]


def _ensure_3d(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be Tensor, got {type(x)}")
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"{name} must be [N,D] or [B,N,D], got shape={tuple(x.shape)}")



def _ensure_2d_bank(bank: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
    if bank is None:
        return None
    if not isinstance(bank, torch.Tensor):
        return None
    if bank.numel() == 0:
        return None
    if bank.dim() == 2:
        return bank
    if bank.dim() == 3 and bank.size(0) == 1:
        return bank.squeeze(0)
    raise ValueError(f"{name} must be [K,D] or [1,K,D], got shape={tuple(bank.shape)}")



def safe_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def pad_or_trim_tokens(
    x: Optional[torch.Tensor],
    target_len: int,
    hidden_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    pad_mode: str = "zero",
) -> torch.Tensor:
    if target_len < 0:
        raise ValueError(f"target_len must be >= 0, got {target_len}")
    if x is None:
        return torch.zeros(1, target_len, hidden_size, device=device, dtype=dtype)
    x = _ensure_3d(x, "pad_or_trim_tokens")
    if x.size(0) != 1:
        raise ValueError(f"pad_or_trim_tokens expects batch size 1, got {tuple(x.shape)}")
    if x.size(-1) != hidden_size:
        raise ValueError(f"hidden dim mismatch: got {x.size(-1)}, expected {hidden_size}")
    x = x.to(device=device, dtype=dtype)
    if x.size(1) == target_len:
        return x.contiguous()
    if x.size(1) > target_len:
        return x[:, :target_len, :].contiguous()
    pad_len = target_len - x.size(1)
    if pad_mode == "repeat" and x.size(1) > 0:
        idx = torch.arange(pad_len, device=device) % x.size(1)
        pad = x.index_select(1, idx)
    else:
        pad = torch.zeros(1, pad_len, hidden_size, device=device, dtype=dtype)
    return torch.cat([x, pad], dim=1).contiguous()


def pad_or_trim_coords(
    coords: torch.Tensor,
    target_len: int,
    *,
    pad_value: float = 0.5,
) -> torch.Tensor:
    coords = _ensure_3d(coords, "coords").float()
    if coords.size(1) == target_len:
        return coords.contiguous()
    if coords.size(1) > target_len:
        return coords[:, :target_len, :].contiguous()
    pad_len = target_len - coords.size(1)
    if coords.size(1) > 0:
        idx = torch.arange(pad_len, device=coords.device) % coords.size(1)
        pad = coords.index_select(1, idx)
    else:
        pad = torch.full(
            (coords.size(0), pad_len, coords.size(2)),
            fill_value=pad_value,
            device=coords.device,
            dtype=coords.dtype,
        )
    return torch.cat([coords, pad], dim=1).contiguous()


@torch.no_grad()
def kmeans_pytorch(X: torch.Tensor, num_clusters: int, max_iter: int = 100, tol: float = 1e-6):
    N, D = X.shape
    device = X.device
    if N == 0:
        raise ValueError("kmeans_pytorch received empty input")
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be > 0, got {num_clusters}")
    if N < num_clusters:
        idx = torch.randint(0, N, (num_clusters,), device=device)
        centers = X[idx]
        labels = torch.arange(N, device=device) % num_clusters
        return centers, labels

    idx = torch.randperm(N, device=device)[:num_clusters]
    centers = X[idx].clone()

    for _ in range(max_iter):
        dists = torch.cdist(X, centers, p=2)
        labels = dists.argmin(dim=1)

        one_hot = torch.zeros(N, num_clusters, device=device, dtype=X.dtype)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        new_centers = one_hot.t() @ X
        counts = one_hot.sum(0).clamp_min(1)
        new_centers = new_centers / counts.unsqueeze(1)

        if torch.allclose(centers, new_centers, atol=tol):
            centers = new_centers
            break
        centers = new_centers

    dists = torch.cdist(X, centers, p=2)
    labels = dists.argmin(dim=1)
    return centers, labels


class LlavaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_mm_token_type_ids": False},
        "images_kwargs": {},
    }


class CustomLlavaModel(HF_LlavaModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_cluster_tokens = int(getattr(config, "num_cluster_tokens", DEFAULT_NUM_CLUSTER_TOKENS))
        self.global_bank_topk = int(getattr(config, "global_bank_topk", DEFAULT_GLOBAL_BANK_TOPK))
        self.entity_bank_topk = int(getattr(config, "entity_bank_topk", DEFAULT_ENTITY_BANK_TOPK))
        self.layout_bank_topk = int(getattr(config, "layout_bank_topk", DEFAULT_LAYOUT_BANK_TOPK))
        self.relation_bank_topk = int(getattr(config, "relation_bank_topk", DEFAULT_RELATION_BANK_TOPK))
        self.layout_grid_size = int(getattr(config, "layout_grid_size", DEFAULT_LAYOUT_GRID_SIZE))
        self.relation_near_threshold = float(getattr(config, "relation_near_threshold", DEFAULT_RELATION_NEAR_THRESHOLD))
        self.relation_overlap_threshold = float(getattr(config, "relation_overlap_threshold", DEFAULT_RELATION_OVERLAP_THRESHOLD))
        self.relation_direction_margin = float(getattr(config, "relation_direction_margin", DEFAULT_RELATION_DIRECTION_MARGIN))
        self.relation_max_pairs = int(getattr(config, "relation_max_pairs", DEFAULT_RELATION_MAX_PAIRS))
        self.semantic_bank_path = getattr(config, "semantic_bank_path", None)
        self.bank_retrieval_temperature = float(getattr(config, "bank_retrieval_temperature", 1.0))
        self.use_bank = bool(getattr(config, "use_bank", True))
        self.meta_cog_enabled = bool(getattr(config, "meta_cog_enabled", DEFAULT_META_COG_ENABLED))
        self.meta_cog_num_steps = int(getattr(config, "meta_cog_num_steps", DEFAULT_META_COG_NUM_STEPS))
        self.meta_cog_min_steps = int(getattr(config, "meta_cog_min_steps", DEFAULT_META_COG_MIN_STEPS))
        self.meta_cog_stop_threshold = float(getattr(config, "meta_cog_stop_threshold", DEFAULT_META_COG_STOP_THRESHOLD))
        self.meta_cog_stability_threshold = float(
            getattr(config, "meta_cog_stability_threshold", DEFAULT_META_COG_STABILITY_THRESHOLD)
        )
        self.meta_cog_equilibrium_threshold = float(
            getattr(config, "meta_cog_equilibrium_threshold", DEFAULT_META_COG_EQUILIBRIUM_THRESHOLD)
        )
        self.meta_cog_action_entropy_weight = float(
            getattr(config, "meta_cog_action_entropy_weight", DEFAULT_META_COG_ACTION_ENTROPY_WEIGHT)
        )
        self.meta_cog_contradiction_weight = float(
            getattr(config, "meta_cog_contradiction_weight", DEFAULT_META_COG_CONTRADICTION_WEIGHT)
        )
        self.meta_cog_uncertainty_weight = float(
            getattr(config, "meta_cog_uncertainty_weight", DEFAULT_META_COG_UNCERTAINTY_WEIGHT)
        )
        self.meta_cog_num_actions = int(getattr(config, "meta_cog_num_actions", DEFAULT_META_COG_NUM_ACTIONS))
        self.meta_cog_state_tokens = int(
            getattr(
                config,
                "meta_cog_state_tokens",
                self.global_bank_topk + self.entity_bank_topk + self.layout_bank_topk + self.relation_bank_topk,
            )
        )

        self._semantic_global_bank: Optional[torch.Tensor] = None
        self._semantic_entity_bank: Optional[torch.Tensor] = None
        self._semantic_layout_cell_to_bank: Dict[int, torch.Tensor] = {}
        self._semantic_relation_type_to_bank: Dict[str, torch.Tensor] = {}
        self.entity_bank_names: List[str] = []
        self.bank_config: Dict[str, Any] = {}
        self.bank_stats: Dict[str, Any] = {}

        hidden_size = int(
            getattr(
                getattr(config, "text_config", None),
                "hidden_size",
                getattr(config.vision_config, "hidden_size"),
            )
        )
        num_heads = max(1, min(8, hidden_size // 64))
        transition_hidden = max(hidden_size, hidden_size // 2)
        halt_hidden = max(64, hidden_size // 2)
        self.meta_cog_state_norm = nn.LayerNorm(hidden_size)
        self.meta_cog_patch_norm = nn.LayerNorm(hidden_size)
        self.meta_cog_memory_norm = nn.LayerNorm(hidden_size)
        self.meta_cog_output_norm = nn.LayerNorm(hidden_size)
        self.meta_cog_patch_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.meta_cog_memory_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.meta_cog_action_embed = nn.Embedding(self.meta_cog_num_actions, hidden_size)
        self.meta_cog_action_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.meta_cog_num_actions),
        )
        self.meta_cog_transition = nn.Sequential(
            nn.Linear(hidden_size * 4, transition_hidden),
            nn.GELU(),
            nn.Linear(transition_hidden, hidden_size),
        )
        self.meta_cog_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.meta_cog_halt_head = nn.Sequential(
            nn.Linear(hidden_size, halt_hidden),
            nn.GELU(),
            nn.Linear(halt_hidden, 1),
        )
        self.meta_cog_uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, halt_hidden),
            nn.GELU(),
            nn.Linear(halt_hidden, 1),
        )
        self.meta_cog_contradiction_head = nn.Sequential(
            nn.Linear(hidden_size * 3, halt_hidden),
            nn.GELU(),
            nn.Linear(halt_hidden, 1),
        )

        if self.use_bank and self.semantic_bank_path is not None and os.path.exists(self.semantic_bank_path):
            self._load_semantic_bank(self.semantic_bank_path)

    def _load_semantic_bank(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"semantic bank payload must be dict, got {type(payload)}")

        self.bank_config = payload.get("config", {})
        self.bank_stats = payload.get("stats", {})

        global_bank = payload.get("global_bank", {})
        global_prototypes = global_bank.get("prototypes", None)
        if global_prototypes is None:
            self._semantic_global_bank = None
        else:
            global_prototypes = _ensure_2d_bank(global_prototypes.float(), "global_bank['prototypes']")
            # BUG 已经在bank生成是归一化了
            # self._semantic_global_bank = None if global_prototypes is None else safe_normalize(global_prototypes.contiguous())
            self._semantic_global_bank = None if global_prototypes is None else global_prototypes.contiguous()

        entity_bank = payload.get("entity_bank", {})
        entity_entries = entity_bank.get("entries", [])
        entity_prototypes = []
        entity_names = []
        for item in entity_entries:
            if not isinstance(item, dict):
                continue
            feats = item.get("prototypes", None)
            name = item.get("name", "")
            if feats is None or not torch.is_tensor(feats):
                continue
            feats = _ensure_2d_bank(feats.float(), f"entity entry '{name}' prototypes")
            if feats is None:
                continue
            # feats = safe_normalize(feats.contiguous()) # BUG 不需要归一化
            feats = feats.contiguous()
            entity_prototypes.append(feats)
            entity_names.extend([name] * feats.shape[0])
        if len(entity_prototypes) > 0:
            self._semantic_entity_bank = torch.cat(entity_prototypes, dim=0).contiguous()
            self.entity_bank_names = entity_names
        else:
            self._semantic_entity_bank = None
            self.entity_bank_names = []

        self._semantic_layout_cell_to_bank = {}
        for item in payload.get("layout_bank", {}).get("entries", []):
            if not isinstance(item, dict):
                continue
            feats = item.get("prototypes", None)
            if not torch.is_tensor(feats):
                continue
            feats = _ensure_2d_bank(feats.float(), f"layout cell {item.get('cell_id')} prototypes")
            if feats is None:
                continue
            cell_id = int(item.get("cell_id", -1))
            if cell_id >= 0:
                # BUG 不需要归一化
                # self._semantic_layout_cell_to_bank[cell_id] = safe_normalize(feats.contiguous())
                self._semantic_layout_cell_to_bank[cell_id] = feats.contiguous()

        self._semantic_relation_type_to_bank = {}
        for item in payload.get("relation_bank", {}).get("entries", []):
            if not isinstance(item, dict):
                continue
            feats = item.get("prototypes", None)
            if not torch.is_tensor(feats):
                continue
            feats = _ensure_2d_bank(feats.float(), f"relation type {item.get('relation')} prototypes")
            if feats is None:
                continue
            rel_type = str(item.get("relation", ""))
            if rel_type:
                # BUG 不需要归一化
                # self._semantic_relation_type_to_bank[rel_type] = safe_normalize(feats.contiguous())
                self._semantic_relation_type_to_bank[rel_type] = feats.contiguous()

        print(f"[SemanticBank] loaded from: {path}")
        print(f"[SemanticBank] stats.feature_dim = {self.bank_stats.get('feature_dim', 'N/A')}")
        print("[SemanticBank] global shape =", None if self._semantic_global_bank is None else tuple(self._semantic_global_bank.shape))
        print("[SemanticBank] entity shape =", None if self._semantic_entity_bank is None else tuple(self._semantic_entity_bank.shape))
        print("[SemanticBank] layout cells =", len(self._semantic_layout_cell_to_bank))
        print("[SemanticBank] relation types =", sorted(self._semantic_relation_type_to_bank.keys()))

    def _fix_one(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            raise TypeError(f"Expected tensor in _fix_one, got {type(t)}")
        if t.dim() == 2:
            t = t.unsqueeze(0)
        elif t.dim() != 3:
            raise ValueError(f"Unexpected tensor shape in get_image_features: {tuple(t.shape)}")
        B, N, D = t.shape
        if N >= 201:
            t = t[:, 5:201, :]
        elif N >= 196:
            t = t[:, :196, :]
        else:
            raise ValueError(f"Unexpected patch token length: {N}")
        return t.contiguous()

    @torch.no_grad()
    def _build_online_cluster_tokens(self, patch_feats: torch.Tensor) -> torch.Tensor:
        patch_feats = _ensure_3d(patch_feats, "patch_feats")
        B, N, D = patch_feats.shape
        outputs = []
        for b in range(B):
            one = patch_feats[b:b + 1]
            patches = safe_normalize(one.float())
            sim_matrix = patches @ patches.transpose(-1, -2)
            X = sim_matrix[0]
            _, labels = kmeans_pytorch(X, self.num_cluster_tokens)
            cluster_feat = torch.zeros(self.num_cluster_tokens, D, device=one.device, dtype=one.dtype)
            for k in range(self.num_cluster_tokens):
                mask = labels == k
                cluster_feat[k] = one[0, mask].mean(dim=0) if mask.any() else one[0].mean(dim=0)
            cluster_feat = safe_normalize(cluster_feat.float()).to(dtype=one.dtype).unsqueeze(0)
            outputs.append(cluster_feat)
        return torch.cat(outputs, dim=0).contiguous()

    @torch.no_grad()
    def _build_cluster_coords(self, patch_tokens: torch.Tensor, cluster_tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens = _ensure_3d(patch_tokens, "patch_tokens")
        cluster_tokens = _ensure_3d(cluster_tokens, "cluster_tokens")
        B, N, _ = patch_tokens.shape
        K = cluster_tokens.shape[1]
        patch_coords = reconstruct_patch_grid_coords(N, device=patch_tokens.device, dtype=patch_tokens.dtype)
        patch_coords = patch_coords.unsqueeze(0).expand(B, -1, -1)

        patch_n = safe_normalize(patch_tokens.float())
        # cluster_n = safe_normalize(cluster_tokens.float()) # BUG cluster_tokens已经归一化的
        cluster_n = cluster_tokens.float()
        sim = torch.matmul(cluster_n, patch_n.transpose(-1, -2))
        assign = sim.argmax(dim=-1)

        coords = []
        for b in range(B):
            one = []
            for k in range(K):
                idx = assign[b, k]
                one.append(patch_coords[b, idx].unsqueeze(0))
            coords.append(torch.cat(one, dim=0).unsqueeze(0))
        return torch.cat(coords, dim=0)

    @torch.no_grad()
    def _retrieve_global_prior(self, patch_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if self._semantic_global_bank is None:
            return None
        x = _ensure_3d(patch_tokens, "patch_tokens")
        orig_dtype = x.dtype
        # x = F.normalize(x.float(), dim=-1) # 平均后在归一化
        bank = _ensure_2d_bank(self._semantic_global_bank, "_semantic_global_bank")
        if bank is None:
            return None
        # bank = F.normalize(bank.to(device=x.device, dtype=x.dtype), dim=-1) # BUG 不用归一化
        bank = bank.to(device=x.device, dtype=x.dtype) # BUG 不用归一化
        B, _, D = x.shape
        if bank.size(-1) != D:
            raise ValueError(f"global bank dim mismatch: patch_tokens dim={D}, bank dim={bank.size(-1)}")
        img_global = F.normalize(x.mean(dim=1), dim=-1)
        sim = torch.matmul(img_global, bank.transpose(0, 1))
        # BUG 不进行训练的话，没必要。除法是单调函数
        # if self.bank_retrieval_temperature != 1.0:
        #     sim = sim / self.bank_retrieval_temperature
        topk = min(int(self.global_bank_topk), bank.size(0))
        top_idx = sim.topk(k=topk, dim=-1).indices
        global_prior = bank.index_select(0, top_idx.reshape(-1)).view(B, topk, D)
        # global_prior = F.normalize(global_prior, dim=-1).to(dtype=orig_dtype) # BUG bank是已经归一化的
        return global_prior[:1] if patch_tokens.dim() == 2 else global_prior

    @torch.no_grad()
    def _retrieve_entity_prior(self, local_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if self._semantic_entity_bank is None:
            return None
        x = _ensure_3d(local_tokens, "local_tokens")
        orig_dtype = x.dtype
        # x = F.normalize(x.float(), dim=-1) # BUG local_tokens是归一化后的
        bank = _ensure_2d_bank(self._semantic_entity_bank, "_semantic_entity_bank")
        if bank is None:
            return None
        # bank = F.normalize(bank.to(device=x.device, dtype=x.dtype), dim=-1) # BUG bank一样
        bank = bank.to(device=x.device, dtype=x.dtype) # BUG bank一样
        B, _, D = x.shape
        if bank.size(-1) != D:
            raise ValueError(f"entity bank dim mismatch: local_tokens dim={D}, bank dim={bank.size(-1)}")
        scores = torch.matmul(x, bank.transpose(0, 1))
        # BUG 同样
        # if self.bank_retrieval_temperature != 1.0:
        #     scores = scores / self.bank_retrieval_temperature
        best_scores = scores.max(dim=1).values
        topk = min(int(self.entity_bank_topk), bank.size(0))
        top_idx = best_scores.topk(k=topk, dim=-1).indices
        entity_prior = bank.index_select(0, top_idx.reshape(-1)).view(B, topk, D)
        # entity_prior = F.normalize(entity_prior, dim=-1).to(dtype=orig_dtype) # BUG 同样
        return entity_prior[:1] if local_tokens.dim() == 2 else entity_prior

    @torch.no_grad()
    def _retrieve_layout_prior(self, local_tokens: torch.Tensor, local_coords: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._semantic_layout_cell_to_bank:
            return None
        feats = _ensure_3d(local_tokens, "local_tokens")
        coords = _ensure_3d(local_coords, "local_coords")
        orig_dtype = feats.dtype
        # feats = F.normalize(feats.float(), dim=-1) # BUG 同样
        coords = coords.float()
        B, N, D = feats.shape
        gathered = []
        for b in range(B):
            per_img = []
            for i in range(N):
                cell_id = coord_to_cell_id(coords[b, i], self.layout_grid_size)
                bank = self._semantic_layout_cell_to_bank.get(cell_id, None)
                if bank is None or bank.numel() == 0:
                    continue
                # bank = F.normalize(bank.to(device=feats.device, dtype=feats.dtype), dim=-1) # BUG 同样
                bank = bank.to(device=feats.device, dtype=feats.dtype) # BUG 同样
                if bank.size(-1) != D:
                    raise ValueError(f"layout bank dim mismatch: local_tokens dim={D}, bank dim={bank.size(-1)}")
                score = torch.matmul(feats[b, i:i+1], bank.transpose(0, 1))
                # BUG 同样
                # if self.bank_retrieval_temperature != 1.0:
                #     score = score / self.bank_retrieval_temperature
                k = min(1, bank.shape[0])
                idx = score.topk(k=k, dim=-1).indices[0]
                per_img.append(bank.index_select(0, idx))
            if len(per_img) == 0:
                gathered.append(None)
                continue
            one = torch.cat(per_img, dim=0)
            one = unique_rows_by_similarity(one, sim_threshold=0.999)
            limit = min(int(self.layout_bank_topk), one.shape[0])
            one = one[:limit].to(dtype=orig_dtype)
            gathered.append(one.unsqueeze(0))
        if all(x is None for x in gathered):
            return None
        max_len = max(x.shape[1] for x in gathered if x is not None)
        out = []
        for x in gathered:
            if x is None:
                out.append(torch.zeros(1, 0 if max_len == 0 else max_len, D, device=feats.device, dtype=orig_dtype))
            elif x.shape[1] < max_len:
                pad = x[:, :1, :].expand(1, max_len - x.shape[1], D)
                out.append(torch.cat([x, pad], dim=1))
            else:
                out.append(x)
        out = torch.cat(out, dim=0)
        return out[:, :max_len, :]

    def _expected_cognitive_token_len(self) -> int:
        return int(self.global_bank_topk) + int(self.entity_bank_topk) + int(self.layout_bank_topk) + int(self.relation_bank_topk)

    def _build_action_priors(
        self,
        state_tokens: torch.Tensor,
        state_coords: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden_size = state_tokens.size(-1)
        orig_dtype = state_tokens.dtype
        device = state_tokens.device
        priors: Dict[str, torch.Tensor] = {}
        priors["global"] = pad_or_trim_tokens(
            self._retrieve_global_prior(state_tokens),
            int(self.global_bank_topk),
            hidden_size,
            device=device,
            dtype=orig_dtype,
            pad_mode="repeat",
        )
        priors["entity"] = pad_or_trim_tokens(
            self._retrieve_entity_prior(state_tokens),
            int(self.entity_bank_topk),
            hidden_size,
            device=device,
            dtype=orig_dtype,
            pad_mode="zero",
        )
        priors["layout"] = pad_or_trim_tokens(
            self._retrieve_layout_prior(state_tokens, state_coords),
            int(self.layout_bank_topk),
            hidden_size,
            device=device,
            dtype=orig_dtype,
            pad_mode="zero",
        )
        priors["relation"] = pad_or_trim_tokens(
            self._retrieve_relation_prior(state_tokens, state_coords),
            int(self.relation_bank_topk),
            hidden_size,
            device=device,
            dtype=orig_dtype,
            pad_mode="zero",
        )
        return priors

    def _compose_action_memory(
        self,
        action_probs: torch.Tensor,
        priors: Dict[str, torch.Tensor],
        state_tokens: torch.Tensor,
    ) -> torch.Tensor:
        global_prior = priors["global"]
        entity_prior = priors["entity"]
        layout_prior = priors["layout"]
        relation_prior = priors["relation"]
        global_summary = global_prior.mean(dim=1, keepdim=True)
        entity_summary = entity_prior.mean(dim=1, keepdim=True)
        layout_summary = layout_prior.mean(dim=1, keepdim=True)
        relation_summary = relation_prior.mean(dim=1, keepdim=True)
        state_summary = state_tokens.mean(dim=1, keepdim=True)
        verify_prior = 0.5 * (entity_summary + relation_summary)
        revise_prior = 0.5 * (layout_summary + relation_summary)
        uncertainty_prior = state_summary
        stop_prior = global_summary

        action_memory = (
            action_probs[:, 0:1, None] * global_summary
            + action_probs[:, 1:2, None] * entity_summary
            + action_probs[:, 2:3, None] * layout_summary
            + action_probs[:, 3:4, None] * relation_summary
            + action_probs[:, 4:5, None] * verify_prior
            + action_probs[:, 5:6, None] * revise_prior
            + action_probs[:, 6:7, None] * stop_prior
            + action_probs[:, 7:8, None] * uncertainty_prior
        )
        return action_memory.expand(-1, state_tokens.size(1), -1).contiguous()

    def _retrieve_meta_cog_memory(
        self,
        state_tokens: torch.Tensor,
        state_coords: torch.Tensor,
    ) -> torch.Tensor:
        priors = self._build_action_priors(state_tokens, state_coords)
        return torch.cat([priors["global"], priors["entity"], priors["layout"], priors["relation"]], dim=1).contiguous()

    def _run_meta_cog_reasoning(
        self,
        patch_tokens: torch.Tensor,
        cluster_tokens: torch.Tensor,
        cluster_coords: torch.Tensor,
    ) -> torch.Tensor:
        patch_tokens = _ensure_3d(patch_tokens, "patch_tokens")
        cluster_tokens = _ensure_3d(cluster_tokens, "cluster_tokens")
        cluster_coords = _ensure_3d(cluster_coords, "cluster_coords")

        state_token_budget = max(1, int(self.meta_cog_state_tokens or self._expected_cognitive_token_len()))
        hidden_size = patch_tokens.size(-1)
        orig_dtype = patch_tokens.dtype
        device = patch_tokens.device

        state_coords = pad_or_trim_coords(cluster_coords, state_token_budget)
        seed_memory = self._retrieve_meta_cog_memory(
            pad_or_trim_tokens(
                cluster_tokens,
                state_token_budget,
                hidden_size,
                device=device,
                dtype=orig_dtype,
                pad_mode="repeat",
            ),
            state_coords,
        )
        state = pad_or_trim_tokens(
            seed_memory if seed_memory.size(1) > 0 else cluster_tokens,
            state_token_budget,
            hidden_size,
            device=device,
            dtype=orig_dtype,
            pad_mode="repeat",
        )
        patch_summary = patch_tokens.mean(dim=1, keepdim=True).expand(-1, state.size(1), -1)
        state = self.meta_cog_state_norm(state + patch_summary)

        aggregated = torch.zeros_like(state)
        residual_mass = torch.ones(state.size(0), 1, 1, device=device, dtype=state.dtype)
        action_indices = torch.arange(self.meta_cog_num_actions, device=device)

        for step_idx in range(max(1, int(self.meta_cog_num_steps))):
            state_summary = state.mean(dim=1)
            action_logits = self.meta_cog_action_head(torch.cat([state_summary, patch_tokens.mean(dim=1)], dim=-1))
            action_probs = torch.softmax(action_logits, dim=-1)
            action_context = torch.matmul(action_probs, self.meta_cog_action_embed(action_indices))
            action_context = action_context.unsqueeze(1).expand_as(state)
            patch_context, _ = self.meta_cog_patch_attn(
                query=self.meta_cog_state_norm(state + action_context),
                key=self.meta_cog_patch_norm(patch_tokens),
                value=patch_tokens,
                need_weights=False,
            )
            priors = self._build_action_priors(state, state_coords)
            dynamic_memory = torch.cat([priors["global"], priors["entity"], priors["layout"], priors["relation"]], dim=1).contiguous()
            action_memory = self._compose_action_memory(action_probs, priors, state)
            memory_context, _ = self.meta_cog_memory_attn(
                query=self.meta_cog_state_norm(state + action_context),
                key=self.meta_cog_memory_norm(torch.cat([dynamic_memory, action_memory[:, :1, :]], dim=1)),
                value=torch.cat([dynamic_memory, action_memory[:, :1, :]], dim=1),
                need_weights=False,
            )
            shared_patch = patch_context.mean(dim=1, keepdim=True).expand_as(state)
            transition_input = torch.cat([state + action_context, patch_context, memory_context + action_memory, shared_patch], dim=-1)
            gate_input = torch.cat([state + action_context, patch_context, memory_context + action_memory], dim=-1)
            delta = self.meta_cog_transition(transition_input)
            gate = self.meta_cog_gate(gate_input)
            next_state = self.meta_cog_output_norm(state + gate * delta)

            state_shift = torch.norm((next_state - state).float(), dim=-1).mean(dim=-1, keepdim=True)
            uncertainty = torch.sigmoid(self.meta_cog_uncertainty_head(next_state.mean(dim=1)))
            patch_alignment = F.cosine_similarity(next_state.mean(dim=1), patch_tokens.mean(dim=1), dim=-1, eps=1e-6).unsqueeze(-1)
            contradiction_input = torch.cat(
                [
                    next_state.mean(dim=1),
                    dynamic_memory.mean(dim=1),
                    patch_tokens.mean(dim=1),
                ],
                dim=-1,
            )
            contradiction = torch.sigmoid(self.meta_cog_contradiction_head(contradiction_input)) * (1.0 - patch_alignment.clamp(-1.0, 1.0)) * 0.5
            action_entropy = -(action_probs * torch.log(action_probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
            equilibrium_energy = (
                state_shift
                + self.meta_cog_action_entropy_weight * action_entropy
                + self.meta_cog_contradiction_weight * contradiction
                + self.meta_cog_uncertainty_weight * uncertainty
            )
            stability = torch.exp(-equilibrium_energy).unsqueeze(-1).to(dtype=state.dtype)
            halt_prob = torch.sigmoid(self.meta_cog_halt_head(next_state.mean(dim=1))).unsqueeze(1)
            stop_mass = residual_mass * halt_prob * stability

            aggregated = aggregated + stop_mass * next_state
            residual_mass = residual_mass * (1.0 - halt_prob * stability)
            state = next_state

            if (
                (not self.training)
                and step_idx + 1 >= int(self.meta_cog_min_steps)
                and (
                    float(halt_prob.mean()) >= float(self.meta_cog_stop_threshold)
                    or float(equilibrium_energy.mean()) <= float(self.meta_cog_equilibrium_threshold)
                )
                and float(state_shift.mean()) <= float(self.meta_cog_stability_threshold)
            ):
                break

        aggregated = aggregated + residual_mass * state
        return aggregated.to(dtype=orig_dtype).contiguous()

    @torch.no_grad()
    def _retrieve_relation_prior(self, local_tokens: torch.Tensor, local_coords: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._semantic_relation_type_to_bank:
            return None
        feats = _ensure_3d(local_tokens, "local_tokens")
        coords = _ensure_3d(local_coords, "local_coords")
        orig_dtype = feats.dtype
        # feats = F.normalize(feats.float(), dim=-1) # BUG
        coords = coords.float()
        B, N, D = feats.shape
        outputs = []
        for b in range(B):
            relation_queries = build_relation_queries(
                local_features=feats[b],
                local_coords=coords[b],
                near_threshold=self.relation_near_threshold,
                overlap_threshold=self.relation_overlap_threshold,
                direction_margin=self.relation_direction_margin,
                max_pairs=self.relation_max_pairs,
            )
            if len(relation_queries) == 0:
                outputs.append(None)
                continue
            picked = []
            for rel_type, rel_query in relation_queries:
                bank = self._semantic_relation_type_to_bank.get(rel_type, None)
                if bank is None or bank.numel() == 0:
                    continue
                # bank = F.normalize(bank.to(device=feats.device, dtype=feats.dtype), dim=-1) # BUG
                bank = bank.to(device=feats.device, dtype=feats.dtype) # BUG
                if bank.size(-1) != D:
                    raise ValueError(f"relation bank dim mismatch: local_tokens dim={D}, bank dim={bank.size(-1)}")
                score = torch.matmul(rel_query.unsqueeze(0), bank.transpose(0, 1))
                # BUG
                # if self.bank_retrieval_temperature != 1.0:
                #     score = score / self.bank_retrieval_temperature
                k = min(1, bank.shape[0])
                idx = score.topk(k=k, dim=-1).indices[0]
                picked.append(bank.index_select(0, idx))
            if len(picked) == 0:
                outputs.append(None)
                continue
            one = torch.cat(picked, dim=0)
            one = unique_rows_by_similarity(one, sim_threshold=0.999)
            limit = min(int(self.relation_bank_topk), one.shape[0])
            one = one[:limit].to(dtype=orig_dtype)
            outputs.append(one.unsqueeze(0))
        if all(x is None for x in outputs):
            return None
        max_len = max(x.shape[1] for x in outputs if x is not None)
        out = []
        for x in outputs:
            if x is None:
                out.append(torch.zeros(1, 0 if max_len == 0 else max_len, D, device=feats.device, dtype=orig_dtype))
            elif x.shape[1] < max_len:
                pad = x[:, :1, :].expand(1, max_len - x.shape[1], D)
                out.append(torch.cat([x, pad], dim=1))
            else:
                out.append(x)
        out = torch.cat(out, dim=0)
        return out[:, :max_len, :]

    def get_image_features(self, *args: Any, **kwargs: Any):
        feats = super().get_image_features(*args, **kwargs)
        if not isinstance(feats, list):
            raise TypeError(f"Expected image features as a list, got {type(feats)}")

        outputs = []
        for t in feats:
            if not torch.is_tensor(t):
                raise TypeError(f"Each image feature must be Tensor, got {type(t)}")
            orig_dtype = t.dtype
            t = self._fix_one(t)
            cluster_tokens = self._build_online_cluster_tokens(t) # cluster_tokens是归一化后的结果
            cluster_coords = self._build_cluster_coords(t, cluster_tokens)
            global_prior = self._retrieve_global_prior(t)
            entity_prior = self._retrieve_entity_prior(cluster_tokens)
            layout_prior = self._retrieve_layout_prior(cluster_tokens, cluster_coords)
            relation_prior = self._retrieve_relation_prior(cluster_tokens, cluster_coords)

            # tokens = [t, cluster_tokens]
            tokens = [t]
            if global_prior is not None:
                tokens.append(global_prior)
            if entity_prior is not None:
                tokens.append(entity_prior)
            if layout_prior is not None and layout_prior.shape[1] > 0:
                tokens.append(layout_prior)
            if relation_prior is not None and relation_prior.shape[1] > 0:
                tokens.append(relation_prior)

            new_feats = torch.cat(tokens, dim=1).contiguous()
            if new_feats.dtype != orig_dtype:
                new_feats = new_feats.to(orig_dtype)
            outputs.append(new_feats)
        return outputs

    def get_image_features(self, *args: Any, **kwargs: Any):
        feats = super().get_image_features(*args, **kwargs)
        if not isinstance(feats, list):
            raise TypeError(f"Expected image features as a list, got {type(feats)}")

        outputs = []
        for t in feats:
            if not torch.is_tensor(t):
                raise TypeError(f"Each image feature must be Tensor, got {type(t)}")
            orig_dtype = t.dtype
            t = self._fix_one(t)
            tokens = [t]
            cluster_tokens = self._build_online_cluster_tokens(t)
            cluster_coords = self._build_cluster_coords(t, cluster_tokens)
            if self.meta_cog_enabled:
                tokens.append(self._run_meta_cog_reasoning(t, cluster_tokens, cluster_coords))
            else:
                global_prior = self._retrieve_global_prior(t)
                entity_prior = self._retrieve_entity_prior(cluster_tokens)
                layout_prior = self._retrieve_layout_prior(cluster_tokens, cluster_coords)
                relation_prior = self._retrieve_relation_prior(cluster_tokens, cluster_coords)
                if global_prior is not None:
                    tokens.append(global_prior)
                if entity_prior is not None:
                    tokens.append(entity_prior)
                if layout_prior is not None and layout_prior.shape[1] > 0:
                    tokens.append(layout_prior)
                if relation_prior is not None and relation_prior.shape[1] > 0:
                    tokens.append(relation_prior)

            new_feats = torch.cat(tokens, dim=1).contiguous()
            if new_feats.dtype != orig_dtype:
                new_feats = new_feats.to(orig_dtype)
            outputs.append(new_feats)
        return outputs


class CustomLlavaForConditionalGeneration(HF_LlavaForConditionalGeneration):
    _checkpoint_conversion_mapping = getattr(HF_LlavaForConditionalGeneration, "_checkpoint_conversion_mapping", {})
    _tied_weights_keys = getattr(HF_LlavaForConditionalGeneration, "_tied_weights_keys", [])

    def __init__(self, config):
        LlavaPreTrainedModel.__init__(self, config)
        self.model = CustomLlavaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


class CustomLlavaProcessor(HF_LlavaProcessor):
    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[LlavaProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            LlavaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"]) if images is not None else {}

        if text is None:
            prompt_strings = None
        elif isinstance(text, str):
            prompt_strings = [text]
        elif isinstance(text, list):
            if len(text) == 0:
                prompt_strings = text
            elif isinstance(text[0], str):
                prompt_strings = text
            else:
                raise TypeError("Invalid input text: expected list[str]")
        else:
            raise TypeError("Invalid input text")

        if prompt_strings is not None and image_inputs.get("pixel_values") is not None:
            add_global = int(getattr(self, "global_bank_topk", DEFAULT_GLOBAL_BANK_TOPK))
            add_entity = int(getattr(self, "entity_bank_topk", DEFAULT_ENTITY_BANK_TOPK))
            add_layout = int(getattr(self, "layout_bank_topk", DEFAULT_LAYOUT_BANK_TOPK))
            add_relation = int(getattr(self, "relation_bank_topk", DEFAULT_RELATION_BANK_TOPK))
            if bool(getattr(self, "meta_cog_enabled", DEFAULT_META_COG_ENABLED)):
                add_reason = int(
                    getattr(
                        self,
                        "meta_cog_state_tokens",
                        add_global + add_entity + add_layout + add_relation,
                    )
                )
            else:
                add_reason = add_global + add_entity + add_layout + add_relation
            num_image_tokens = 196 + add_reason
            prompt_strings = [sample.replace(self.image_token, self.image_token * num_image_tokens) for sample in prompt_strings]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        if return_tensors is not None:
            text_inputs = BatchFeature(data=text_inputs, tensor_type=return_tensors)
            image_inputs = BatchFeature(data=image_inputs, tensor_type=return_tensors)

        if isinstance(text_inputs, BatchFeature):
            text_inputs.update(image_inputs)
            return text_inputs
        text_inputs.update(image_inputs)
        return BatchFeature(data=text_inputs)


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


def reconstruct_patch_grid_coords(num_patches: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if num_patches <= 0:
        raise ValueError(f"num_patches must be positive, got {num_patches}")
    h, w = infer_grid_shape(num_patches)
    xs, ys = [], []
    for idx in range(num_patches):
        y = idx // w
        x = idx % w
        y = min(y, h - 1)
        x = min(x, w - 1)
        xs.append((x + 0.5) / max(w, 1))
        ys.append((y + 0.5) / max(h, 1))
    return torch.tensor(list(zip(xs, ys)), dtype=dtype, device=device)


def coord_to_cell_id(coord: torch.Tensor, grid_size: int) -> int:
    x = float(coord[0])
    y = float(coord[1])
    gx = min(max(int(x * grid_size), 0), grid_size - 1)
    gy = min(max(int(y * grid_size), 0), grid_size - 1)
    return gy * grid_size + gx


def classify_relation(src_coord: torch.Tensor, dst_coord: torch.Tensor, near_threshold: float = 0.22, overlap_threshold: float = 0.10, direction_margin: float = 0.08) -> Optional[str]:
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


def build_relation_queries(local_features: torch.Tensor, local_coords: torch.Tensor, near_threshold: float = 0.22, overlap_threshold: float = 0.10, direction_margin: float = 0.08, max_pairs: int = 64) -> List[Tuple[str, torch.Tensor]]:
    # local_features = F.normalize(local_features.float(), dim=-1) # BUG
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
            rel_type = classify_relation(ci, cj, near_threshold, overlap_threshold, direction_margin)
            if rel_type is None:
                continue
            fi = local_features[i]
            fj = local_features[j]
            rel_feat = 0.5 * fi + 0.5 * fj + 0.25 * (fj - fi)
            rel_feat = F.normalize(rel_feat.unsqueeze(0), dim=-1)[0]
            dist = torch.norm(cj - ci, p=2).item()
            pair_candidates.append((relation_priority(rel_type, dist), rel_type, rel_feat))
    if not pair_candidates:
        return []
    pair_candidates.sort(key=lambda x: x[0], reverse=True)
    pair_candidates = pair_candidates[:max_pairs]
    return [(rel_type, rel_feat) for _, rel_type, rel_feat in pair_candidates]


def unique_rows_by_similarity(x: torch.Tensor, sim_threshold: float = 0.999) -> torch.Tensor:
    if x.shape[0] <= 1:
        return x
    # x = F.normalize(x.float(), dim=-1) # BUG 同样
    keep = []
    for i in range(x.shape[0]):
        if not keep:
            keep.append(i)
            continue
        sims = x[i].unsqueeze(0) @ x[keep].t()
        if float(sims.max()) < sim_threshold:
            keep.append(i)
    keep_idx = torch.tensor(keep, dtype=torch.long, device=x.device)
    return x.index_select(0, keep_idx)
