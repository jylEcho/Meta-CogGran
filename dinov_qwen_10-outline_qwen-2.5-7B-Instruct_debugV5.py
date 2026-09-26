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

from custom_llava_10 import (
    CustomLlavaForConditionalGeneration,
    CustomLlavaProcessor,
)

# =========================================================
# 0. 强制离线模式
# =========================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================
# 1. 本地模型路径
# =========================================================
llm_model_name = "./external/huggingface/Qwen2.5-7B-Instruct"
vision_model_name = "./external/huggingface/dinov3-vitl16-pretrain-lvd1689m"

print("llm_model_name:", llm_model_name)
print("vision_model_name:", vision_model_name)

export_path = "./external/granulon_work/dinov_qwen_10_qwen-2.5-7B-Instruct_debugV5"
os.makedirs(export_path, exist_ok=True)

# =========================================================
# 2. tokenizer（离线加载）
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(
    llm_model_name,
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True,
)

# 添加 <image> special token
num_added = tokenizer.add_tokens(
    [AddedToken("<image>", special=True, normalized=False)],
    special_tokens=True
)
print("num_added_tokens:", num_added)

image_token_index = tokenizer.encode("<image>", add_special_tokens=False)
print("raw image_token_index:", image_token_index)

assert len(image_token_index) == 1, "Tokenizer failed to register <image> as a single token!"
image_token_index = image_token_index[0]
print("final image_token_index:", image_token_index)

# =========================================================
# 3. image processor（离线加载）
# =========================================================
image_processor = AutoImageProcessor.from_pretrained(
    vision_model_name,
    local_files_only=True,
)

processor = CustomLlavaProcessor(
    tokenizer=tokenizer,
    image_processor=image_processor,
    patch_size=16,   # DINOv3 ViT-L/16
    vision_feature_select_strategy="full",
    image_token="<image>",
    num_additional_image_tokens=0,
)

print("processor init done")

# =========================================================
# 4. 加载视觉模型并动态探测 seq_len / hidden_size
# =========================================================
full_vision = AutoModel.from_pretrained(
    vision_model_name,
    local_files_only=True,
)

vision_model = getattr(full_vision, "vision_model", full_vision)
vision_model.eval()

with torch.no_grad():
    dummy = Image.new("RGB", (224, 224), color=0)
    pv = image_processor(images=dummy, return_tensors="pt")["pixel_values"]
    out = vision_model(pv)
    seq_len = out.last_hidden_state.shape[1]
    hidden_size = out.last_hidden_state.shape[2]

print("detected vision seq_len:", seq_len)
print("detected vision hidden_size:", hidden_size)

# =========================================================
# 5. 加载 text / vision config（离线）
# =========================================================
# llm_config = AutoConfig.from_pretrained(
#     llm_model_name,
#     trust_remote_code=True,
#     local_files_only=True,
# )
llm_config = AutoConfig.from_pretrained(
    llm_model_name,
    trust_remote_code=True,
    local_files_only=True,
)

# 关键：先把 text config 的 vocab_size 对齐到最终 tokenizer
llm_config.vocab_size = len(tokenizer)



vision_config = DINOv3ViTConfig.from_pretrained(
    vision_model_name,
    local_files_only=True,
)
vision_config.vision_use_head = False

# 去掉 CLS token 后的 patch token 数量
image_seq_length = seq_len - 1
print("image_seq_length:", image_seq_length)

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
# =========================================================
# 6. 初始化自定义多模态模型
# =========================================================
model = CustomLlavaForConditionalGeneration(config)
print("Custom multimodal model init done")

print("\n====== Projector Structure ======")
if hasattr(model, "multi_modal_projector"):
    print(model.multi_modal_projector)
elif hasattr(model, "projector"):
    print(model.projector)
else:
    print("⚠️ No projector found in model.")
print("=================================\n")

# =========================================================
# 7. 加载语言模型（离线）
# =========================================================
language_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    trust_remote_code=True,
    local_files_only=True,
)

old_vocab_size = language_model.get_input_embeddings().weight.shape[0]
new_vocab_size = len(tokenizer)

print(f"language_model vocab_size before resize: {old_vocab_size}")
print(f"tokenizer size: {new_vocab_size}")

# 只有 tokenizer 长度变了才 resize
if old_vocab_size != new_vocab_size:
    language_model.resize_token_embeddings(new_vocab_size)
    print(f"resize_token_embeddings done: {old_vocab_size} -> {new_vocab_size}")
else:
    print("skip resize_token_embeddings (same vocab size)")

# 先从完整 CausalLM 里拿真正的输出头
real_lm_head = None
if hasattr(language_model, "lm_head") and language_model.lm_head is not None:
    real_lm_head = language_model.lm_head
else:
    real_lm_head = language_model.get_output_embeddings()

if real_lm_head is None:
    raise ValueError("Cannot find lm_head / output_embeddings from loaded language_model")

# 再挂到你的多模态模型上
model.language_model = language_model


# 关键：输入 embedding 也要同步到加载后的 language_model
real_input_embeddings = language_model.get_input_embeddings()

if hasattr(model, "set_input_embeddings"):
    try:
        model.set_input_embeddings(real_input_embeddings)
        print("set_input_embeddings done")
    except Exception as e:
        print("set_input_embeddings skipped:", repr(e))


# 关键修复：顶层 lm_head 必须替换成真正加载出来的那份
model.lm_head = real_lm_head
print("top-level lm_head <- loaded language_model output head")

# 如果你的自定义模型实现了 set_output_embeddings，也一起同步
if hasattr(model, "set_output_embeddings"):
    try:
        model.set_output_embeddings(model.lm_head)
        print("set_output_embeddings done")
    except Exception as e:
        print("set_output_embeddings skipped:", repr(e))

print("language model loaded")

# =========================================================
# debug
# =========================================================
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


# =========================================================
# 8. 关闭 cache
# =========================================================
if hasattr(model, "config"):
    model.config.use_cache = False
if hasattr(model.language_model, "config"):
    model.language_model.config.use_cache = False

# =========================================================
# 9. 加载视觉塔权重（离线）
# =========================================================
vision_state = full_vision.state_dict()
load_result = model.vision_tower.load_state_dict(vision_state, strict=False)

print("vision tower:")
print(model.vision_tower)

if hasattr(load_result, "missing_keys"):
    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys
else:
    missing_keys = load_result[0] if isinstance(load_result, tuple) else []
    unexpected_keys = load_result[1] if isinstance(load_result, tuple) else []

print("vision tower loaded")
print("missing_keys count:", len(missing_keys))
print("unexpected_keys count:", len(unexpected_keys))
print("missing_keys sample:", missing_keys[:10])
print("unexpected_keys sample:", unexpected_keys[:10])

core_missing = [k for k in missing_keys if "encoder.layer" in k or "layers" in k]
print("core ViT missing sample:", core_missing[:10])

# =========================================================
# 10. 顶层 resize（如果你的自定义模型支持）
# =========================================================
# if hasattr(model, "resize_token_embeddings"):
#     try:
#         model.resize_token_embeddings(len(tokenizer))
#         print("top-level resize_token_embeddings done")
#     except Exception as e:
#         print("skip top-level resize_token_embeddings:", repr(e))

# =========================================================
# 11. 保存
# =========================================================
tokenizer.save_pretrained(export_path)
processor.save_pretrained(export_path)
model.config.save_pretrained(export_path)
model.save_pretrained(export_path)

print("model save done:", export_path)