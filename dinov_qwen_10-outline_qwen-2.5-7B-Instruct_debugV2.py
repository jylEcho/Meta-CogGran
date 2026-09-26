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

export_path = "./external/granulon_work/dinov_qwen_10_qwen-2.5-7B-Instruct_debugV2"
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
llm_config = AutoConfig.from_pretrained(
    llm_model_name,
    trust_remote_code=True,
    local_files_only=True,
)

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
# 7. 加载语言模型（离线，必须是 CausalLM）
# =========================================================
from transformers import AutoConfig, AutoModelForCausalLM

print("\n===== LOAD LANGUAGE MODEL =====")

# 先读配置，确认本地目录到底声明成什么架构
llm_config = AutoConfig.from_pretrained(
    llm_model_name,
    trust_remote_code=True,
    local_files_only=True,
)

print("llm config class:", llm_config.__class__.__name__)
print("llm config model_type:", getattr(llm_config, "model_type", None))
print("llm config architectures:", getattr(llm_config, "architectures", None))
print("llm config auto_map:", getattr(llm_config, "auto_map", None))

# =========================================================
# 加载完整 CausalLM
# =========================================================
full_lm = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    trust_remote_code=True,
    local_files_only=True,
)

print("loaded full_lm class:", full_lm.__class__.__name__)

# ---------------------------------------------------------
# 强校验：必须是带输出头的语言模型，而不是纯 backbone
# ---------------------------------------------------------
output_emb = None

if hasattr(full_lm, "lm_head") and full_lm.lm_head is not None:
    output_emb = full_lm.lm_head
    print("found output head from full_lm.lm_head")
elif hasattr(full_lm, "get_output_embeddings"):
    try:
        output_emb = full_lm.get_output_embeddings()
        if output_emb is not None:
            print("found output head from full_lm.get_output_embeddings()")
    except Exception as e:
        print("full_lm.get_output_embeddings() failed:", repr(e))
        output_emb = None

if output_emb is None:
    raise RuntimeError(
        "\n[ERROR] Loaded model does NOT have output embeddings / lm_head.\n"
        f"full_lm class       = {full_lm.__class__.__name__}\n"
        f"config class        = {llm_config.__class__.__name__}\n"
        f"architectures       = {getattr(llm_config, 'architectures', None)}\n"
        "This usually means your local model path is a BASE model (e.g. Qwen2Model),\n"
        "not a CausalLM model (e.g. Qwen2ForCausalLM). Please check config.json\n"
        "and make sure the checkpoint directory is a full causal language model."
    )

# ---------------------------------------------------------
# resize vocab
# 注意：resize 可能会重建 embedding / lm_head
# ---------------------------------------------------------
old_vocab_size = full_lm.get_input_embeddings().weight.shape[0]
new_vocab_size = len(tokenizer)

print(f"resize_token_embeddings: {old_vocab_size} -> {new_vocab_size}")
full_lm.resize_token_embeddings(new_vocab_size)

# resize 后重新拿输出头，防止对象被替换
output_emb = None
if hasattr(full_lm, "lm_head") and full_lm.lm_head is not None:
    output_emb = full_lm.lm_head
elif hasattr(full_lm, "get_output_embeddings"):
    try:
        output_emb = full_lm.get_output_embeddings()
    except Exception as e:
        print("full_lm.get_output_embeddings() after resize failed:", repr(e))
        output_emb = None

if output_emb is None:
    raise RuntimeError(
        f"Cannot find output embeddings after resize. "
        f"full_lm class = {full_lm.__class__.__name__}"
    )

# ---------------------------------------------------------
# tie_weights：有些模型会在这里重新绑定 / 替换 lm_head
# ---------------------------------------------------------
if hasattr(full_lm, "tie_weights"):
    try:
        full_lm.tie_weights()
        print("full_lm.tie_weights() done")
    except Exception as e:
        print("full_lm.tie_weights() skipped:", repr(e))

# tie_weights 后再同步一次，防止 lm_head 对象被替换
output_emb = None
if hasattr(full_lm, "lm_head") and full_lm.lm_head is not None:
    output_emb = full_lm.lm_head
elif hasattr(full_lm, "get_output_embeddings"):
    try:
        output_emb = full_lm.get_output_embeddings()
    except Exception as e:
        print("re-sync full_lm.get_output_embeddings() failed:", repr(e))
        output_emb = None

if output_emb is None:
    raise RuntimeError(
        f"Cannot find output embeddings after tie_weights. "
        f"full_lm class = {full_lm.__class__.__name__}"
    )

# =========================================================
# 修法 A：
# 不把完整 CausalLM 直接塞进 model.language_model
# 而是拆成 backbone + lm_head
# =========================================================
if not hasattr(full_lm, "model") or full_lm.model is None:
    raise RuntimeError(
        f"full_lm has no valid backbone `.model`. "
        f"full_lm class = {full_lm.__class__.__name__}"
    )

model.language_model = full_lm.model
model.lm_head = output_emb

print("mounted model.language_model class:", model.language_model.__class__.__name__)
print("mounted model.lm_head class:", model.lm_head.__class__.__name__)

# 如果顶层模型自己也有 tie_weights，就试一下
if hasattr(model, "tie_weights"):
    try:
        model.tie_weights()
        print("model.tie_weights() done")
    except Exception as e:
        print("model.tie_weights() skipped:", repr(e))

print("language model loaded and lm_head synced")

# =========================================================
# debug
# =========================================================
print("\n===== CHECK LM HEAD TIE =====")

# 1) 输入 embedding
input_emb = None

if hasattr(model.language_model, "embed_tokens") and model.language_model.embed_tokens is not None:
    input_emb = model.language_model.embed_tokens
elif hasattr(model, "get_input_embeddings"):
    try:
        input_emb = model.get_input_embeddings()
    except Exception as e:
        print("model.get_input_embeddings() failed:", repr(e))
        input_emb = None

if input_emb is None:
    raise RuntimeError("Cannot find input embeddings from model.language_model / model")

print("input_emb module:", input_emb.__class__.__name__)
print("input_emb weight shape:", tuple(input_emb.weight.shape))

# 2) 输出头
if not hasattr(model, "lm_head") or model.lm_head is None:
    raise RuntimeError("model.lm_head is None after sync")

if not hasattr(model.lm_head, "weight") or model.lm_head.weight is None:
    raise RuntimeError("model.lm_head has no weight")

print("lm_head module:", model.lm_head.__class__.__name__)
print("lm_head weight shape:", tuple(model.lm_head.weight.shape))

# 3) tokenizer / vocab 检查
print("tokenizer size:", len(tokenizer))
print("config vocab_size:", getattr(llm_config, "vocab_size", None))

if model.lm_head.weight.shape[0] != len(tokenizer):
    print(
        f"⚠️ WARNING: lm_head vocab size != tokenizer size: "
        f"{model.lm_head.weight.shape[0]} vs {len(tokenizer)}"
    )

if input_emb.weight.shape[0] != len(tokenizer):
    print(
        f"⚠️ WARNING: input_emb vocab size != tokenizer size: "
        f"{input_emb.weight.shape[0]} vs {len(tokenizer)}"
    )

# 4) 是否共享权重
same_ptr = model.lm_head.weight.data_ptr() == input_emb.weight.data_ptr()
print("lm_head tied with embedding:", same_ptr)

if not same_ptr:
    print("⚠️ WARNING: lm_head 没有和 embedding 共享权重！！！")
else:
    print("✅ lm_head 和 embedding 已共享权重")

print("================================\n")
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
if hasattr(model, "resize_token_embeddings"):
    try:
        model.resize_token_embeddings(len(tokenizer))
        print("top-level resize_token_embeddings done")
    except Exception as e:
        print("skip top-level resize_token_embeddings:", repr(e))

# =========================================================
# 11. 保存
# =========================================================
tokenizer.save_pretrained(export_path)
processor.save_pretrained(export_path)
model.config.save_pretrained(export_path)
model.save_pretrained(export_path)

print("model save done:", export_path)