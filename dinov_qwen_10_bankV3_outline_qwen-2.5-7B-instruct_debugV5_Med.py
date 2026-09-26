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

llm_model_name = "./external/huggingface/Qwen2.5-7B-Instruct"
vision_model_name = "./external/huggingface/dinov3-vitl16-pretrain-lvd1689m"
export_path = "./external/granulon_work/dinov_qwen_10_bankV3_qwen2.5-7B-Instruct_Med"
semantic_bank_path = "./external/granulon/Bank_SurgSigma/semantic_bankV3/semantic_bank_qwen-2.5-7B-Instruct.pt"

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

llm_config = AutoConfig.from_pretrained(
    llm_model_name,
    trust_remote_code=True,
    local_files_only=True,
)
llm_config.vocab_size = len(tokenizer)

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
config.vocab_size = len(tokenizer)
if hasattr(config, "text_config") and config.text_config is not None:
    config.text_config.vocab_size = len(tokenizer)
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

old_vocab_size = language_model.get_input_embeddings().weight.shape[0]
new_vocab_size = len(tokenizer)
print(f"language_model vocab_size before resize: {old_vocab_size}")
print(f"tokenizer size: {new_vocab_size}")

if old_vocab_size != new_vocab_size:
    language_model.resize_token_embeddings(new_vocab_size)
    print(f"resize_token_embeddings done: {old_vocab_size} -> {new_vocab_size}")
else:
    print("skip resize_token_embeddings (same vocab size)")

real_lm_head = None
if hasattr(language_model, "lm_head") and language_model.lm_head is not None:
    real_lm_head = language_model.lm_head
else:
    real_lm_head = language_model.get_output_embeddings()

if real_lm_head is None:
    raise ValueError("Cannot find lm_head / output_embeddings from loaded language_model")

model.language_model = language_model

real_input_embeddings = language_model.get_input_embeddings()
if hasattr(model, "set_input_embeddings"):
    try:
        model.set_input_embeddings(real_input_embeddings)
        print("set_input_embeddings done")
    except Exception as e:
        print("set_input_embeddings skipped:", repr(e))

model.lm_head = real_lm_head
print("top-level lm_head <- loaded language_model output head")

if hasattr(model, "set_output_embeddings"):
    try:
        model.set_output_embeddings(model.lm_head)
        print("set_output_embeddings done")
    except Exception as e:
        print("set_output_embeddings skipped:", repr(e))

print("language model loaded")

print("\n===== CHECK LM HEAD / EMBEDDING BINDING =====")
print("top lm_head      :", model.lm_head.weight.shape)
print("loaded lm_head   :", real_lm_head.weight.shape)
print("top input_emb    :", model.get_input_embeddings().weight.shape)
print("loaded input_emb :", real_input_embeddings.weight.shape)

same_head = model.lm_head.weight.data_ptr() == real_lm_head.weight.data_ptr()
same_input = model.get_input_embeddings().weight.data_ptr() == real_input_embeddings.weight.data_ptr()
same_embed_shape = model.get_input_embeddings().weight.shape == real_input_embeddings.weight.shape

print("top lm_head == loaded lm_head:", same_head)
print("top input_emb == loaded input_emb:", same_input)
print("top input_emb shape == loaded input_emb shape:", same_embed_shape)
print("tie_word_embeddings:", getattr(language_model.config, "tie_word_embeddings", None))
print("===========================================\n")

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
