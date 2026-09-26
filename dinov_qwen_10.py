import os
# os.environ["HF_ENDPOINT"] = "API_ENDPOINT_NOT_CONFIGURED"
os.environ["HF_TOKEN"] = "hf_XXXXXXXXXXXXXXXX"
os.environ["HF_ENDPOINT"] = "API_ENDPOINT_NOT_CONFIGURED"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from PIL import Image
from transformers import AutoConfig,AutoTokenizer,AddedToken,AutoImageProcessor,AutoModelForCausalLM,AutoModel
from transformers import LlavaProcessor,LlavaConfig,LlavaForConditionalGeneration
from transformers import DINOv3ViTConfig
from custom.custom_llava_10 import CustomLlavaForConditionalGeneration, CustomLlavaProcessor


#########1.模型名
# llm_model_name = 'Qwen/Qwen2-7B'
llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
vision_model_name = 'facebook/dinov3-vitl16-pretrain-lvd1689m'

print("llm_model_name:",llm_model_name)
print("vision_model_name:",vision_model_name)
##############2.处理器
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True, use_fast=False)
## 添加 <image>
# 添加<image> token后，确保tokenizer能识别
tokenizer.add_tokens([AddedToken("<image>", special=True, normalized=False)], special_tokens=True)
image_token_index = tokenizer.encode("<image>", add_special_tokens=False)
print("image_token_index:", image_token_index)
assert len(image_token_index) == 1, "Tokenizer failed to register <image> as a single token!"
print("image_token_index:",image_token_index)
print(len(image_token_index))
image_token_index = image_token_index[0]
# raise SystemExit("stop here for debug")
image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

processor = CustomLlavaProcessor(tokenizer=tokenizer, image_processor=image_processor,
                            patch_size =16,  # ✅ DINOv3 是 patch16
                            vision_feature_select_strategy="full",
                            image_token="<image>",
                            num_additional_image_tokens=0)

print("image_token_index:",image_token_index)

######### 3. 加载视觉模型并动态探测 seq_len
# # 先把视觉骨干加载出来
full_vision = AutoModel.from_pretrained(vision_model_name)
vision_model = getattr(full_vision, "vision_model", full_vision)  # ✅ 兼容不同模型结构
vision_model.eval()
with torch.no_grad():
    dummy = Image.new("RGB", (224, 224), color=0)  # ✅ DINOv3 输入224x224
    pv = image_processor(images=dummy, return_tensors="pt")["pixel_values"]
    out = vision_model(pv)
    seq_len = out.last_hidden_state.shape[1]
    hidden_size = out.last_hidden_state.shape[2]

print("detected vision seq_len:", seq_len)
print("detected hidden_size:", hidden_size)

################3.模型结构
llm_config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)


vision_config = DINOv3ViTConfig.from_pretrained(vision_model_name)
vision_config.vision_use_head = False
config = LlavaConfig(
        text_config=llm_config,
        vision_config=vision_config,
        image_token_index=image_token_index,
        vision_feature_layer=-1,## 把 视觉塔最后一层 Transformer 输出的 hidden-state 拿来喂给 projector
        vision_feature_select_strategy="full",  # ✅ 关键改动：与 processor 对齐
        image_seq_length=seq_len - 1,##（384//14）^2
)

#初始化模型
# model = LlavaForConditionalGeneration(config)
model = CustomLlavaForConditionalGeneration(config)
print("model init done")

############### 6. 打印 projector 结构
print("\n====== Projector Structure ======")
if hasattr(model, "multi_modal_projector"):
    print(model.multi_modal_projector)
elif hasattr(model, "projector"):
    print(model.projector)
else:
    print("⚠️ No projector found in model (check your LlavaConfig or model structure).")
print("=================================\n")
# raise SystemExit("stop here for debug")
###############4.加载模型
model.language_model = AutoModelForCausalLM.from_pretrained(llm_model_name, trust_remote_code=True)


# 训练时建议关闭缓存（若后续训练）
if hasattr(model, "config"):
    model.config.use_cache = False
if hasattr(model.language_model, "config"):
    model.language_model.config.use_cache = False
# vision 部分有一些麻烦
dino_full = AutoModel.from_pretrained(vision_model_name)
missing = model.vision_tower.load_state_dict(dino_full.state_dict(), strict=False)

print(model.vision_tower)
print("missing:",missing)
print("vision tower 加载完成 | missing:", len(missing))
print("核心 ViT 缺失示例:", [k for k in missing if 'encoder.layer' in k][:5])

#######5.保存模型
export_path = './external/e1374390/work/dinov_qwen_10'
os.makedirs(export_path, exist_ok=True)
tokenizer.save_pretrained(export_path)
model.config.save_pretrained(export_path)
processor.save_pretrained(export_path)
model.save_pretrained(export_path)
print("model save done:",export_path)

