from typing import Any, Optional, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
from transformers.processing_utils import (
    ProcessingKwargs,
    Unpack,
)

# -----------------------------
# config defaults
# -----------------------------
DEFAULT_NUM_CLUSTER_TOKENS = 10
DEFAULT_GLOBAL_BANK_TOPK = 4
DEFAULT_ENTITY_BANK_TOPK = 6


def _ensure_3d(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    [N, D]    -> [1, N, D]
    [B, N, D] -> [B, N, D]
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be Tensor, got {type(x)}")
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"{name} must be [N,D] or [B,N,D], got shape={tuple(x.shape)}")


def _ensure_2d_bank(bank: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
    """
    allow:
      [K, D]
      [1, K, D] -> squeeze(0)
    disallow:
      [B, K, D] with B > 1
    """
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
        dists = torch.cdist(X, centers, p=2)  # [N, K]
        labels = dists.argmin(dim=1)          # [N]

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
    """
    在原 get_image_features 基础上做四件事：
    1) patch token 裁成 196
    2) 在线聚类得到 cluster tokens
    3) global semantic bank 检索 global prior tokens
    4) entity semantic bank 检索 entity prior tokens
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_cluster_tokens = int(getattr(config, "num_cluster_tokens", DEFAULT_NUM_CLUSTER_TOKENS))
        self.global_bank_topk = int(getattr(config, "global_bank_topk", DEFAULT_GLOBAL_BANK_TOPK))
        self.entity_bank_topk = int(getattr(config, "entity_bank_topk", DEFAULT_ENTITY_BANK_TOPK))
        self.semantic_bank_path = getattr(config, "semantic_bank_path", None)
        self.bank_retrieval_temperature = float(getattr(config, "bank_retrieval_temperature", 1.0))
        self.use_bank = bool(getattr(config, "use_bank", True))

        # 不用 register_buffer("global_bank", None)
        # 避免和父类 / HF / DeepSpeed / state_dict 流程冲突
        self._semantic_global_bank: Optional[torch.Tensor] = None   # [Kg, D]
        self._semantic_entity_bank: Optional[torch.Tensor] = None   # [Ke, D]

        self.entity_bank_names = []
        self.bank_config = {}
        self.bank_stats = {}

        if self.use_bank and self.semantic_bank_path is not None and os.path.exists(self.semantic_bank_path):
            self._load_semantic_bank(self.semantic_bank_path)

    def _load_semantic_bank(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu")

        if not isinstance(payload, dict):
            raise TypeError(f"semantic bank payload must be dict, got {type(payload)}")

        self.bank_config = payload.get("config", {})
        self.bank_stats = payload.get("stats", {})

        # ---------- global bank ----------
        global_bank = payload.get("global_bank", {})
        global_prototypes = global_bank.get("prototypes", None)

        if global_prototypes is None:
            self._semantic_global_bank = None
        else:
            if not torch.is_tensor(global_prototypes):
                raise TypeError(
                    f"global_bank['prototypes'] must be Tensor, got {type(global_prototypes)}"
                )

            global_prototypes = global_prototypes.float()
            global_prototypes = _ensure_2d_bank(global_prototypes, "global_bank['prototypes']")
            if global_prototypes is None:
                self._semantic_global_bank = None
            else:
                self._semantic_global_bank = safe_normalize(global_prototypes.contiguous())

        # ---------- entity bank ----------
        entity_bank = payload.get("entity_bank", {})
        entity_entries = entity_bank.get("entries", [])

        entity_prototypes = []
        entity_names = []

        for item in entity_entries:
            if not isinstance(item, dict):
                continue

            feats = item.get("prototypes", None)
            name = item.get("name", "")

            if feats is None:
                continue
            if not torch.is_tensor(feats):
                raise TypeError(f"entity entry prototypes must be Tensor, got {type(feats)}")

            feats = feats.float()
            feats = _ensure_2d_bank(feats, f"entity entry '{name}' prototypes")
            if feats is None:
                continue

            feats = safe_normalize(feats.contiguous())
            entity_prototypes.append(feats)
            entity_names.extend([name] * feats.shape[0])

        if len(entity_prototypes) > 0:
            self._semantic_entity_bank = torch.cat(entity_prototypes, dim=0).contiguous()
            self.entity_bank_names = entity_names
        else:
            self._semantic_entity_bank = None
            self.entity_bank_names = []

        # 可选调试输出：建议先开着，确认训练时拿到的维度
        print(f"[SemanticBank] loaded from: {path}")
        print(f"[SemanticBank] stats.feature_dim = {self.bank_stats.get('feature_dim', 'N/A')}")
        print(
            "[SemanticBank] global shape =",
            None if self._semantic_global_bank is None else tuple(self._semantic_global_bank.shape),
        )
        print(
            "[SemanticBank] entity shape =",
            None if self._semantic_entity_bank is None else tuple(self._semantic_entity_bank.shape),
        )

    def _fix_one(self, t: torch.Tensor) -> torch.Tensor:
        """
        input:
          [N, D]
          [1, N, D]
          [B, N, D]

        output:
          [B, 196, D]
        """
        if not torch.is_tensor(t):
            raise TypeError(f"Expected tensor in _fix_one, got {type(t)}")

        if t.dim() == 2:
            t = t.unsqueeze(0)  # [1, N, D]
        elif t.dim() != 3:
            raise ValueError(f"Unexpected tensor shape in get_image_features: {tuple(t.shape)}")

        B, N, D = t.shape

        if N >= 201:
            # 跳过前 5 个，保留后面 196 个
            t = t[:, 5:201, :]
        elif N >= 196:
            t = t[:, :196, :]
        else:
            raise ValueError(f"Unexpected patch token length: {N}")

        return t.contiguous()  # [B, 196, D]

    @torch.no_grad()
    def _build_online_cluster_tokens(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """
        patch_feats: [B, 196, D]
        return: [B, K, D]
        """
        patch_feats = _ensure_3d(patch_feats, "patch_feats")
        B, N, D = patch_feats.shape

        outputs = []
        for b in range(B):
            one = patch_feats[b:b + 1]                         # [1, 196, D]
            patches = safe_normalize(one.float())              # [1, 196, D]
            sim_matrix = patches @ patches.transpose(-1, -2)   # [1, 196, 196]
            X = sim_matrix[0]                                  # [196, 196]

            centers, labels = kmeans_pytorch(X, self.num_cluster_tokens)

            cluster_feat = torch.zeros(
                self.num_cluster_tokens,
                D,
                device=one.device,
                dtype=one.dtype,
            )

            for k in range(self.num_cluster_tokens):
                mask = labels == k
                if mask.any():
                    cluster_feat[k] = one[0, mask].mean(dim=0)
                else:
                    cluster_feat[k] = one[0].mean(dim=0)

            cluster_feat = safe_normalize(cluster_feat.float()).to(dtype=one.dtype).unsqueeze(0)  # [1, K, D]
            outputs.append(cluster_feat)

        return torch.cat(outputs, dim=0).contiguous()  # [B, K, D]

    @torch.no_grad()
    def _retrieve_global_prior(self, patch_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        patch_tokens:
            [N, D] or [B, N, D]

        return:
            [1, Kg, D] if input is [N, D]
            [B, Kg, D] if input is [B, N, D]
        """
        if self._semantic_global_bank is None:
            return None

        x = _ensure_3d(patch_tokens, "patch_tokens")  # [B, N, D]
        orig_dtype = x.dtype
        x = F.normalize(x.float(), dim=-1)

        bank = _ensure_2d_bank(self._semantic_global_bank, "_semantic_global_bank")
        if bank is None:
            return None

        bank = F.normalize(bank.to(device=x.device, dtype=x.dtype), dim=-1)   # [G, D]

        B, N, D = x.shape
        if bank.size(-1) != D:
            raise ValueError(
                f"global bank dim mismatch: patch_tokens dim={D}, bank dim={bank.size(-1)}"
            )

        img_global = F.normalize(x.mean(dim=1), dim=-1)  # [B, D]

        if self.bank_retrieval_temperature != 1.0:
            sim = torch.matmul(img_global, bank.transpose(0, 1)) / self.bank_retrieval_temperature
        else:
            sim = torch.matmul(img_global, bank.transpose(0, 1))

        topk = min(int(self.global_bank_topk), bank.size(0))
        top_idx = sim.topk(k=topk, dim=-1).indices  # [B, Kg]

        global_prior = bank.index_select(0, top_idx.reshape(-1)).view(B, topk, D)
        global_prior = F.normalize(global_prior, dim=-1).to(dtype=orig_dtype)

        if patch_tokens.dim() == 2:
            return global_prior[:1]
        return global_prior

    @torch.no_grad()
    def _retrieve_entity_prior(self, local_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        local_tokens:
            [N, D] or [B, N, D]

        return:
            [1, Ke, D] if input is [N, D]
            [B, Ke, D] if input is [B, N, D]
        """
        if self._semantic_entity_bank is None:
            return None

        x = _ensure_3d(local_tokens, "local_tokens")  # [B, N, D]
        orig_dtype = x.dtype
        x = F.normalize(x.float(), dim=-1)

        bank = _ensure_2d_bank(self._semantic_entity_bank, "_semantic_entity_bank")
        if bank is None:
            return None

        bank = F.normalize(bank.to(device=x.device, dtype=x.dtype), dim=-1)   # [E, D]

        B, N, D = x.shape
        if bank.size(-1) != D:
            raise ValueError(
                f"entity bank dim mismatch: local_tokens dim={D}, bank dim={bank.size(-1)}"
            )

        if self.bank_retrieval_temperature != 1.0:
            scores = torch.matmul(x, bank.transpose(0, 1)) / self.bank_retrieval_temperature
        else:
            scores = torch.matmul(x, bank.transpose(0, 1))

        best_scores = scores.max(dim=1).values  # [B, E]

        topk = min(int(self.entity_bank_topk), bank.size(0))
        top_idx = best_scores.topk(k=topk, dim=-1).indices  # [B, Ke]

        entity_prior = bank.index_select(0, top_idx.reshape(-1)).view(B, topk, D)
        entity_prior = F.normalize(entity_prior, dim=-1).to(dtype=orig_dtype)

        if local_tokens.dim() == 2:
            return entity_prior[:1]
        return entity_prior

    def get_image_features(self, *args: Any, **kwargs: Any):
        feats = super().get_image_features(*args, **kwargs)

        if not isinstance(feats, list):
            raise TypeError(f"Expected image features as a list, got {type(feats)}")

        outputs = []
        for t in feats:
            if not torch.is_tensor(t):
                raise TypeError(f"Each image feature must be Tensor, got {type(t)}")

            orig_dtype = t.dtype
            t = self._fix_one(t)  # [B, 196, D]

            # online semantic tokens: [B, Kc, D]
            cluster_tokens = self._build_online_cluster_tokens(t)

            # retrieved priors
            global_prior = self._retrieve_global_prior(t)               # [B, Kg, D] or None
            entity_prior = self._retrieve_entity_prior(cluster_tokens)  # [B, Ke, D] or None

            tokens = [t]
            tokens.append(cluster_tokens)

            if global_prior is not None:
                tokens.append(global_prior)
            if entity_prior is not None:
                tokens.append(entity_prior)

            # 全部都是 [B, *, D]
            new_feats = torch.cat(tokens, dim=1).contiguous()

            # 保持和输入视觉特征同 dtype，减少不必要的 dtype 混乱
            if new_feats.dtype != orig_dtype:
                new_feats = new_feats.to(orig_dtype)

            outputs.append(new_feats)

        return outputs


class CustomLlavaForConditionalGeneration(HF_LlavaForConditionalGeneration):
    _checkpoint_conversion_mapping = getattr(
        HF_LlavaForConditionalGeneration, "_checkpoint_conversion_mapping", {}
    )
    _tied_weights_keys = getattr(
        HF_LlavaForConditionalGeneration, "_tied_weights_keys", []
    )

    def __init__(self, config):
        LlavaPreTrainedModel.__init__(self, config)
        self.model = CustomLlavaModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()


class CustomLlavaProcessor(HF_LlavaProcessor):
    """
    关键点：<image> 展开数量要同步增加：
        + num_cluster_tokens
        + global_bank_topk
        + entity_bank_topk
    """

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
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

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

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
            # 这里不要依赖 image_processor 里一定有这些字段
            add_cluster = int(getattr(self, "num_cluster_tokens", DEFAULT_NUM_CLUSTER_TOKENS))
            add_global = int(getattr(self, "global_bank_topk", DEFAULT_GLOBAL_BANK_TOPK))
            add_entity = int(getattr(self, "entity_bank_topk", DEFAULT_ENTITY_BANK_TOPK))

            # 196 patch + cluster + global prior + entity prior
            num_image_tokens = 196 + add_cluster + add_global + add_entity

            expanded_prompt_strings = []
            for sample in prompt_strings:
                expanded_prompt_strings.append(
                    sample.replace(self.image_token, self.image_token * num_image_tokens)
                )
            prompt_strings = expanded_prompt_strings

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)

        text_inputs = self.tokenizer(
            prompt_strings,
            **output_kwargs["text_kwargs"],
            return_tensors=None,
        )
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)