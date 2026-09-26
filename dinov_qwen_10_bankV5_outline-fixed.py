import os
import types
import torch
from PIL import Image

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AddedToken,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModel,
    LlavaConfig,
)
from transformers import DINOv3ViTConfig

from custom_llava_10_bankV5 import (
    CustomLlavaForConditionalGeneration,
    CustomLlavaProcessor,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm_model_name = "./external/huggingface/Qwen2.5-1.5B-Instruct"
vision_model_name = "./external/huggingface/dinov3-vitl16-pretrain-lvd1689m"
export_path = "./external/granulon_work/dinov_qwen_10_bankV5"
semantic_bank_path = "./external/granulon/Bank/semantic_bankV5/semantic_bank.pt"

num_cluster_tokens = 10
global_bank_topk = 4
entity_bank_topk = 6
layout_bank_topk = 4
relation_bank_topk = 4
layout_grid_size = 4
relation_near_threshold = 0.22
relation_overlap_threshold = 0.10
relation_direction_margin = 0.08
relation_max_pairs = 64
patch_drop = 5
use_bank = True

def patch_custom_model_runtime(model):
    """
    只在导出阶段动态修复 custom_llava_10_bankV5.py 里把 layout/relation bank 注释掉的问题，
    避免你必须手改原文件后才能导出。
    """
    mm = model.model

    def fixed_get_image_features(self, *args, **kwargs):
        feats = super(type(self), self).get_image_features(*args, **kwargs)
        if not isinstance(feats, list):
            raise TypeError(f"Expected image features as a list, got {type(feats)}")

        outputs = []
        for t in feats:
            if not torch.is_tensor(t):
                raise TypeError(f"Each image feature must be Tensor, got {type(t)}")
            orig_dtype = t.dtype
            t = self._fix_one(t)
            cluster_tokens = self._build_online_cluster_tokens(t)
            cluster_coords = self._build_cluster_coords(t, cluster_tokens)

            global_prior = self._retrieve_global_prior(t)
            entity_prior = self._retrieve_entity_prior(cluster_tokens)
            layout_prior = self._retrieve_layout_prior(cluster_tokens, cluster_coords)
            relation_prior = self._retrieve_relation_prior(cluster_tokens, cluster_coords)

            tokens = [t]
            if global_prior is not None and global_prior.shape[1] > 0:
                tokens.append(global_prior)
            if entity_prior is not None and entity_prior.shape[1] > 0:
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

    mm.get_image_features = types.MethodType(fixed_get_image_features, mm)


def patch_custom_processor_runtime(processor):
    """
    修复 processor 里 image token 展开长度没把 layout / relation 两个 bank 算进去的问题。
    """
    processor.num_cluster_tokens = num_cluster_tokens
    processor.global_bank_topk = global_bank_topk
    processor.entity_bank_topk = entity_bank_topk
    processor.layout_bank_topk = layout_bank_topk
    processor.relation_bank_topk = relation_bank_topk

    orig_call = processor.__class__.__call__

    def fixed_call(self, images=None, text=None, audio=None, videos=None, **kwargs):
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            self.__class__.__mro__[0].__dict__.get('LlavaProcessorKwargs', object),
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # 直接复用原逻辑不稳，下面走更直接的实现
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
            add_global = int(getattr(self, "global_bank_topk", 0))
            add_entity = int(getattr(self, "entity_bank_topk", 0))
            add_layout = int(getattr(self, "layout_bank_topk", 0))
            add_relation = int(getattr(self, "relation_bank_topk", 0))
            num_image_tokens = 196 + add_global + add_entity + add_layout + add_relation
            prompt_strings = [
                sample.replace(self.image_token, self.image_token * num_image_tokens)
                for sample in prompt_strings
            ]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        try:
            import numpy as np
            from transformers.feature_extraction_utils import BatchFeature
        except Exception:
            np = None
            BatchFeature = dict

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        if return_tensors is not None:
            text_inputs = BatchFeature(data=text_inputs, tensor_type=return_tensors)
            image_inputs = BatchFeature(data=image_inputs, tensor_type=return_tensors)

        if hasattr(text_inputs, "update"):
            text_inputs.update(image_inputs)
            return text_inputs
        text_inputs.update(image_inputs)
        return BatchFeature(data=text_inputs)

    processor.__call__ = types.MethodType(fixed_call, processor)
    return processor


os.makedirs(export_path, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    llm_model_name,
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True,
)
num_added = tokenizer.add_tokens([AddedToken("<image>", special=True, normalized=False)], special_tokens=True)
print("num_added_tokens:", num_added)
image_token_index = tokenizer.encode("<image>", add_special_tokens=False)
assert len(image_token_index) == 1
image_token_index = image_token_index[0]

image_processor = AutoImageProcessor.from_pretrained(vision_model_name, local_files_only=True)
processor = CustomLlavaProcessor(
    tokenizer=tokenizer,
    image_processor=image_processor,
    patch_size=16,
    vision_feature_select_strategy="full",
    image_token="<image>",
    num_additional_image_tokens=0,
    num_cluster_tokens=num_cluster_tokens,
    global_bank_topk=global_bank_topk,
    entity_bank_topk=entity_bank_topk,
    layout_bank_topk=layout_bank_topk,
    relation_bank_topk=relation_bank_topk,
)
processor = patch_custom_processor_runtime(processor)

full_vision = AutoModel.from_pretrained(vision_model_name, local_files_only=True)
vision_model = getattr(full_vision, "vision_model", full_vision)
vision_model.eval()

with torch.no_grad():
    dummy = Image.new("RGB", (224, 224), color=0)
    pv = image_processor(images=dummy, return_tensors="pt")["pixel_values"]
    out = vision_model(pv)
    seq_len = out.last_hidden_state.shape[1]
    print("raw vision seq_len:", seq_len)

llm_config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True, local_files_only=True)
vision_config = DINOv3ViTConfig.from_pretrained(vision_model_name, local_files_only=True)
vision_config.vision_use_head = False

base_patch_tokens = 196
extra_bank_tokens = global_bank_topk + entity_bank_topk + layout_bank_topk + relation_bank_topk
image_seq_length = base_patch_tokens + extra_bank_tokens

config = LlavaConfig(
    text_config=llm_config,
    vision_config=vision_config,
    image_token_index=image_token_index,
    vision_feature_layer=-1,
    vision_feature_select_strategy="full",
    image_seq_length=image_seq_length,
)

# ===== 与 custom_llava_10_bankV5.py 中读取字段完全对齐 =====
config.num_cluster_tokens = num_cluster_tokens
config.global_bank_topk = global_bank_topk
config.entity_bank_topk = entity_bank_topk
config.layout_bank_topk = layout_bank_topk
config.relation_bank_topk = relation_bank_topk
config.layout_grid_size = layout_grid_size
config.relation_near_threshold = relation_near_threshold
config.relation_overlap_threshold = relation_overlap_threshold
config.relation_direction_margin = relation_direction_margin
config.relation_max_pairs = relation_max_pairs
config.patch_drop = patch_drop
config.semantic_bank_path = semantic_bank_path
config.use_bank = use_bank
config.bank_retrieval_temperature = 1.0

model = CustomLlavaForConditionalGeneration(config)
patch_custom_model_runtime(model)

language_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    trust_remote_code=True,
    local_files_only=True,
)
language_model.resize_token_embeddings(len(tokenizer))
model.language_model = language_model

if hasattr(model, "config"):
    model.config.use_cache = False
if hasattr(model.language_model, "config"):
    model.language_model.config.use_cache = False

vision_state = full_vision.state_dict()
load_result = model.vision_tower.load_state_dict(vision_state, strict=False)
print("vision load missing:", getattr(load_result, "missing_keys", []))
print("vision load unexpected:", getattr(load_result, "unexpected_keys", []))

if hasattr(model, "resize_token_embeddings"):
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print("skip top-level resize_token_embeddings:", repr(e))

# ===== 导出前做一次自检，确认四类 bank token 都能拼进去 =====
with torch.no_grad():
    dummy = Image.new("RGB", (224, 224), color=0)
    pv = image_processor(images=dummy, return_tensors="pt")["pixel_values"]
    feats = model.model.get_image_features(
        pixel_values=pv,
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    )
    feat0 = feats[0]
    print("export sanity image feature shape:", tuple(feat0.shape))
    expected = base_patch_tokens + extra_bank_tokens
    print("expected image_seq_length:", expected)
    if feat0.shape[1] != expected:
        raise RuntimeError(
            f"bank token count mismatch: got {feat0.shape[1]}, expected {expected}. "
            f"layout/relation bank may still be missing."
        )

tokenizer.save_pretrained(export_path)
processor.save_pretrained(export_path)
model.config.save_pretrained(export_path)
model.save_pretrained(export_path)

print("model save done:", export_path)
