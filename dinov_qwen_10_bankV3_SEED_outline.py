import os
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

from custom_llava_10_bankV3 import (
    CustomLlavaForConditionalGeneration,
    CustomLlavaProcessor,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm_model_name = "./external/huggingface/Qwen2.5-1.5B-Instruct"
vision_model_name = "./external/huggingface/dinov3-vitl16-pretrain-lvd1689m"
export_path = "./external/granulon_work/dinov_qwen_10_bankV3_SEED"
semantic_bank_path = "./external/granulon/Bank_SEED-Bench/semantic_bankV3/semantic_bank.pt"

semantic_cluster_num = 10
global_bank_topk = 4
entity_bank_topk = 6
patch_drop = 5

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
    semantic_cluster_num=semantic_cluster_num,
    global_bank_topk=global_bank_topk,
    entity_bank_topk=entity_bank_topk,
)

full_vision = AutoModel.from_pretrained(vision_model_name, local_files_only=True)
vision_model = getattr(full_vision, "vision_model", full_vision)
vision_model.eval()

with torch.no_grad():
    dummy = Image.new("RGB", (224, 224), color=0)
    pv = image_processor(images=dummy, return_tensors="pt")["pixel_values"]
    out = vision_model(pv)
    seq_len = out.last_hidden_state.shape[1]

llm_config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True, local_files_only=True)
vision_config = DINOv3ViTConfig.from_pretrained(vision_model_name, local_files_only=True)
vision_config.vision_use_head = False

image_seq_length = seq_len - 1
config = LlavaConfig(
    text_config=llm_config,
    vision_config=vision_config,
    image_token_index=image_token_index,
    vision_feature_layer=-1,
    vision_feature_select_strategy="full",
    image_seq_length=image_seq_length,
)
config.semantic_cluster_num = semantic_cluster_num
config.global_bank_topk = global_bank_topk
config.entity_bank_topk = entity_bank_topk
config.patch_drop = patch_drop
config.semantic_bank_path = semantic_bank_path

model = CustomLlavaForConditionalGeneration(config)
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

tokenizer.save_pretrained(export_path)
processor.save_pretrained(export_path)
model.config.save_pretrained(export_path)
model.save_pretrained(export_path)

print("model save done:", export_path)
