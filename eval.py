import argparse
import json
import os
import sys
from typing import List, Dict, Any
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, LlavaProcessor, LlavaForConditionalGeneration
import base64
import io
import shutil
from peft import PeftModel, PeftConfig
from pycocoevalcap.bleu.bleu import Bleu
# 新 API（可用则用），否则回退旧 API；若都不可用则报错
try:
    from transformers import AutoModelForImageTextToText
    HAS_IT2T = True
except Exception:
    HAS_IT2T = False


from custom_llava_5 import CustomLlavaForConditionalGeneration as LlavaForConditionalGeneration5, CustomLlavaProcessor as LlavaProcessor5
from custom_llava_10 import CustomLlavaForConditionalGeneration as LlavaForConditionalGeneration10, CustomLlavaProcessor as LlavaProcessor10
from custom_llava_10_siglip_qwen257B import CustomLlavaForConditionalGeneration as LlavaForConditionalGeneration10_siglip_qwen257B, CustomLlavaProcessor as LlavaProcessor10_siglip_qwen257B
from custom_llava_10_siglip_qwen38B import CustomLlavaForConditionalGeneration as LlavaForConditionalGeneration10_siglip_qwen38B, CustomLlavaProcessor as LlavaProcessor10_siglip_qwen38B
from custom_llava_qwen3_bankV5 import CustomLlavaForConditionalGeneration as LlavaForConditionalGeneration10_qwen38B_bankV5, CustomLlavaProcessor as LlavaProcessor10_qwen38B_bankV5
from custom_llava_10_bankV3 import CustomLlavaForConditionalGeneration as LlavaForConditionalGeneration10_bankV3, CustomLlavaProcessor as LlavaProcessor10_bankV3
from custom_llava_10_bankV5 import CustomLlavaForConditionalGeneration as LlavaForConditionalGeneration10_bankV5, CustomLlavaProcessor as LlavaProcessor10_bankV5
from custom_llava_30 import CustomLlavaForConditionalGeneration as LlavaForConditionalGeneration30, CustomLlavaProcessor as LlavaProcessor30
from custom_llava_50 import CustomLlavaForConditionalGeneration as LlavaForConditionalGeneration50, CustomLlavaProcessor as LlavaProcessor50
import sys
from pathlib import Path

# 获取项目根目录（llm-debug 的路径，根据实际路径调整）
root_dir = Path(__file__).parent.parent  # ../../ 对应从 ablation → train → llm-debug
sys.path.append(str(root_dir))
# from table1.models.custom.custom_siglip import CustomLlavaForConditionalGeneration as SiglipLlavaForConditionalGeneration

def parse_samples(index_path: str, dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    samples = []
    index_path = Path(index_path)
    dataset_dir = Path(dataset_dir)
    # 读取索引文件
    chat_data = pd.read_json(index_path).to_dict(orient="records")
    image_dir = dataset_dir.joinpath("images") 
    for index in range(num_samples):
        # 获取当前数据项
        cur_data = chat_data[index]  # {'id': 'GCC_train_002109690', 'image': 'GCC_train_002109690.jpg', 'conversations': [{...}, {...}]}
        # 获取对话内容
        conversations = cur_data.get("conversations")  # [{'from': 'human', 'value': 'Offer a succinct explanation of the picture presented.\n<image>'}, {'from': 'gpt', 'value': "it 's appropriate for teens to want to spend more time with their peers than their parents as they get older ."}]

        # 提取人类输入(即问题)【其中 <image> 是图像的占位符】
        human_input = conversations[0].get("value")   # 'Offer a succinct explanation of the picture presented.\n<image>'
        # 提取标准的机器人输出(即答案)
        chatbot_output = conversations[1].get("value")  # "it 's appropriate for teens to want to spend more time with their peers than their parents as they get older ."

        # 构建图像路径
        image_path = image_dir.joinpath(cur_data.get("image"))
        samples.append({
            "image_path": str(image_path),
            "human_question": human_input,  # 保留<image>标签
            "reference": chatbot_output
        })
    return samples


@torch.inference_mode()
def generate_qwen(model, processor, text: str, image: Image.Image, max_new_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 移动到模型设备
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        pad_token_id=getattr(processor.tokenizer, "eos_token_id", None),
    )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_llava(model, processor: LlavaProcessor, text: str, image: Image.Image, max_new_tokens: int) -> str:
    # 与训练/示例代码一致：通过聊天模板构建 prompt，并包含 <image> 占位符
    if "<image>" not in text:
        # 与训练脚本保持一致的约束；如果需要也可以自动补上占位符
        raise ValueError("<image> not in question for LLaVA input.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 使用 LlavaProcessor 同时处理图像与文本 prompt
    inputs = processor(image, prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # 不强制设置 eos/pad 也可正常工作；若需要，可解注释下一行，保持与示例一致
            # eos_token_id=processor.tokenizer.eos_token_id,
            # pad_token_id=processor.tokenizer.eos_token_id,
        )

    # 只解码新生成的内容，避免把 prompt 一起解码
    prompt_len = inputs["input_ids"].shape[1]
    gen_only = out_ids[0][prompt_len:]
    return processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()



def load_qwen_model(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = dict(trust_remote_code=True, dtype=dtype)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"  # 有 GPU 时自动放置
    else:
        model_kwargs["low_cpu_mem_usage"] = True  # CPU 上减少内存峰值

    if HAS_IT2T:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        print("Using AutoModelForImageTextToText.")
    else:
        sys.exit("ERROR: Neither AutoModelForImageTextToText nor AutoModelForVision2Seq is available in this transformers version.")
    model.eval()
    return model, processor


# ...existing code...
def load_llava_model(model_path: str, use_custom: str = "None", use_lora: str = "NO"):
    """
    Load a LLaVA model according to use_custom:
      - "All": use both CustomLlavaProcessor and AllLlavaForConditionalGeneration (requires HAS_CUSTOM_LLAVA_FULL)
      - "Part": use standard LlavaProcessor but PartLlavaForConditionalGeneration as the model (requires HAS_CUSTOM_LLAVA_PART)
      - "None": use standard LlavaProcessor and LlavaForConditionalGeneration from transformers
    """
    model_path = Path(model_path)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Auto device placement when CUDA is available
    model_kwargs = dict(trust_remote_code=True, dtype=dtype)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["low_cpu_mem_usage"] = True

    if use_custom == "dinov_5":
        processor = LlavaProcessor5.from_pretrained(model_path, local_files_only=True)
        print("Loading custom ALL LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration5.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.")
    elif use_custom == "dinov_10":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.")
# dinov_llama_10
    elif use_custom == "dinov_llama_10":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.")
    elif use_custom == "dinov_10_bankV3":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_bankV3.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_bankV3.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.") 
# dinov_llama_10_bankV3    
    elif use_custom == "dinov_llama_10_bankV3":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_bankV3.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_bankV3.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.") 
# SIGLIP_10_qwen257BInstruct_debugV5    
    elif use_custom == "siglip_qwen_10_qwen-2.5-7B-Instruct":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_siglip_qwen257B.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_siglip_qwen257B.from_pretrained(
            model_path.joinpath("reason_trained_undo5"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.") 
# SIGLIP_10_qwen38Bbase_debugV5    
    elif use_custom == "siglip_qwen_10_qwen3-8B-Base_debugV5":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_siglip_qwen38B.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_siglip_qwen38B.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.") 
# dinov_qwen_10_bankV3_SEED   
    elif use_custom == "dinov_qwen_10_bankV3_SEED":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_bankV3.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_bankV3.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.") 

    elif use_custom == "dinov_10_bankV5_liteV1":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_bankV5.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_bankV5.from_pretrained(
            model_path.joinpath("reason_trainedV3-sota"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.")    
    elif use_custom == "dinov_qwen_10_qwen2.5-7B-Instruct":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.")     
    elif use_custom == "dinov_qwen_10_bankV3_qwen2.5-7B-Instruct":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_bankV3.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_bankV3.from_pretrained(
            # model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.") 
    elif use_custom == "dinov_qwen_10_bankV3_qwen2.5-7B-Instruct_Med":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_bankV3.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_bankV3.from_pretrained(
            # model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.") 
    elif use_custom == "dinov_qwen_10_qwen2.5-7B-Instruct_debugV5":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.")   
    elif use_custom == "dinov_qwen_10_bankV3_qwen2.5-7B-Instruct":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_bankV3.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_bankV3.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.")          
    elif use_custom == "dinov_qwen_10_qwen3-8B-Base":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.") 
# dinov_qwen_10_bankV3_qwen3-8B-Base
    elif use_custom == "dinov_qwen_10_bankV3_qwen3-8B-Base":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_bankV3.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_bankV3.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.")    
# dinov_qwen_10_bankV3_qwen3-8B-Base
    elif use_custom == "dinov_qwen3-8B_10_bankV5":
        # Use standard LlavaProcessor but the PART custom model class
        processor = LlavaProcessor10_qwen38B_bankV5.from_pretrained(model_path, local_files_only=True)
        print("Loading custom PART LLaVA model from:", model_path)
        model = LlavaForConditionalGeneration10_qwen38B_bankV5.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )
        # lora_config = PeftConfig.from_pretrained(model_path.joinpath("alignment"), local_files_only=True)
        # model = PeftModel.from_pretrained(model, model_path.joinpath("alignment"), device_map="auto", local_files_only=True)
        # print("LoRA adapter merged successfully.")   
    else:  # "None"
        # Use transformers' LlavaProcessor and LlavaForConditionalGeneration
        processor = LlavaProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        print("Loading standard LLaVA model from:", model_path)
        model = SiglipLlavaForConditionalGeneration.from_pretrained(
            model_path.joinpath("reason_trained"), local_files_only=True, **model_kwargs
        )

    if use_lora.upper() == "YES":
        print("Loading LoRA adapter from:", model_path.joinpath("instruct"))
        lora_config = PeftConfig.from_pretrained(model_path.joinpath("instruct"), local_files_only=True)
        model = PeftModel.from_pretrained(model, model_path.joinpath("instruct"), device_map="auto", local_files_only=True)
        print("LoRA adapter merged successfully.")

    model.eval()
    return model, processor


from bert_score import score

def bertscore_metric(predictions: list[str], references: list[str]) -> float:
    """
    计算平均 BERTScore F1（语义相似度）
    Args:
        predictions: 模型生成的句子列表
        references: 参考句子列表（一对一）
    Returns:
        平均 BERTScore F1 值（float）
    """
    P, R, F1 = score(predictions, references, lang="en", verbose=False)
    return float(F1.mean().item())

def recall_metric(predictions: list[str], references: list[str]) -> float:
    """
    基于词的平均 recall。
    Args:
        predictions: 模型输出
        references: 参考文本
    Returns:
        平均 recall (0~1)
    """
    import re
    
    recalls = []
    for pred, ref in zip(predictions, references):
        pred_tokens = re.findall(r"\w+", pred.lower())
        ref_tokens = re.findall(r"\w+", ref.lower())
        if not ref_tokens:
            continue
        overlap = len(set(pred_tokens) & set(ref_tokens))
        recalls.append(overlap / len(set(ref_tokens)))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


# def compute_accuracy(predictions: List[str], references: List[str]) -> float:
#     """
#     计算准确率。统计预测与参考答案完全匹配的样本比例。
#     """
#     correct = sum(p == r for p, r in zip(predictions, references))
#     return correct / len(predictions) if predictions else 0.0


# ----------  辅助函数 ----------
def _coco_format(predictions: List[str], references: List[str]):
    preds = {i: [p] for i, p in enumerate(predictions)}
    gts   = {i: [r] for i, r in enumerate(references)}
    return preds, gts


def metrics_all(predictions: List[str], references: List[str]) -> Dict[str, float]:
    preds, gts = _coco_format(predictions, references)

    # 1. BLEU-1~4
    bleu = Bleu(4)
    bleu_scores, _ = bleu.compute_score(gts, preds)

    # 5. BERTScore & 6. Recall（已有）
    bert_f1 = bertscore_metric(predictions, references)
    recall   = recall_metric(predictions, references)

    return {
        "BLEU-1": bleu_scores[0],
        "BLEU-2": bleu_scores[1],
        "BLEU-3": bleu_scores[2],
        "BLEU-4": bleu_scores[3],
        "BERTScore-F1": bert_f1,
        "Recall": recall,
    }

def main():
    parser = argparse.ArgumentParser(description="Minimal MM eval on CC3M: caption F1 over N samples.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["qwen2p5_vl", "llava"], required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--use_lora", type=str, default="NO",
                        help="Whether to use LoRA adapter when loading LLaVA model.")
    parser.add_argument("--use_custom_model", type=str, default="None",
                        help="Whether to use custom LLaVA components: All=processor+model, Part=model only, None=standard")
    # 不再依赖 --local_rank/deepspeed
    args = parser.parse_args()
    index_path = Path(args.dataset_dir)
    index_path = index_path.joinpath("chat.json")
    print(f"Using index file: {index_path}")
    # with open('./external/e1374390/work/llm-debug/train/evaldata/processed_data.json', 'r', encoding='utf-8') as f:
    #     samples = json.load(f)
    # print(f'type of samples: {type(samples)}    number of samples: {len(samples)}')
    # print(f'type of samples[0]: {type(samples[0])}    keys of samples[0]: {list(samples[0].keys())}')
    samples = parse_samples(index_path, args.dataset_dir, args.num_samples)

    if args.model_type == "qwen2p5_vl":
        model, proc = load_qwen_model(args.model)
        gen_fn = lambda text, img: generate_qwen(model, proc, text, img, args.max_new_tokens)
    else:
        model, proc = load_llava_model(args.model, args.use_custom_model, args.use_lora)
        gen_fn = lambda text, img: generate_llava(model, proc, text, img, args.max_new_tokens)
    
    samples_out = []
    predictions = []
    references = []

    model_name = Path(args.model).name
    eval_dir = Path("./external/granulon/Eval/" + f"{model_name}" )
    images_dir = eval_dir.joinpath("images")
    eval_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    # 遍历样本时添加打印
    for i, ex in enumerate(samples, 1):  # 从1开始计数
        print(f"\n===== 样本 {i}/{len(samples)} =====")
        # print(f"图像: {ex['image']}")
        # print(f"人类提问: {ex['human_question']}")
        # print(f"参考回答: {ex['reference']}")

        # image_bytes = base64.b64decode(ex['image'])
        # img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = Image.open(ex['image_path']).convert("RGB")
        img = img.resize((224, 224))
        # print(f"样本 {i}: 图像打开成功. 格式: {img.format}, 模式: {img.mode}, 大小: {img.size}")
        
        # 检查是否为 RGB 模式
        # if img.mode != "RGB":
        #     img = img.convert("RGB")
            # print(f"样本 {i}: 图像已转换为 RGB 模式.")
        # print(f"样本 {i}: 图像验证通过.")
        pred = gen_fn(ex["human_question"], img)
        predictions.append(pred)
        references.append(ex["reference"])

        # print(f"模型预测: {pred}")

        src_img = Path(ex["image_path"])
        if not src_img.exists():
            print(f"Warning: image not found, skipping: {src_img}")
            continue
        dst_img = images_dir.joinpath(src_img.name)
        # 覆盖已存在同名文件（若需保留可改为加后缀）
        shutil.copy2(src_img, dst_img)
        samples_out.append({
            "image_path": str(dst_img),  # 保存到 eval_sample/images 下的实际路径
            "human_question": ex["human_question"],
            "reference": ex["reference"],
            "prediction": pred
        })

    json_path = eval_dir.joinpath("eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples_out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(samples_out)} samples to {json_path}")

    # accuracy = compute_accuracy(predictions, references)
    accuracy = metrics_all(predictions, references)
    print(f"\n===== 最终结果 =====")
    # print(f"MODEL={args.model} TYPE={args.model_type} ACCURACY={accuracy:.6f} TOTAL={len(samples)}")
    # print(f"MODEL={args.model} TYPE={args.model_type} Score={accuracy['bertscore_f1']:.6f}, Recall={accuracy['recall']:.4f} TOTAL={len(samples)}")
    print(f"MODEL={args.model} TYPE={args.model_type} Score={accuracy} TOTAL={len(samples)}")
    print("Evaluation complete.")


    # f1_sum = 0.0
    # # 遍历样本时添加打印
    # for i, ex in enumerate(samples, 1):  # 从1开始计数
    #     print(f"\n===== 样本 {i}/{len(samples)} =====")
    #     print(f"图像路径: {ex['image_path']}")
    #     print(f"人类提问: {ex['human_question']}")
    #     print(f"参考回答: {ex['reference']}")

    #     img = Image.open(ex["image_path"]).convert("RGB")
    #     print("image path:", ex["image_path"])
    #     print(f"样本 {i}: 图像打开成功. 格式: {img.format}, 模式: {img.mode}, 大小: {img.size}")
        
    #     # 检查是否为 RGB 模式
    #     if img.mode != "RGB":
    #         img = img.convert("RGB")
    #         print(f"样本 {i}: 图像已转换为 RGB 模式.")
    #     print(f"样本 {i}: 图像验证通过.")
    #     pred = gen_fn(ex["human_question"], img)
    #     current_f1 = token_f1(pred, ex["reference"])
    #     f1_sum += current_f1

    #     print(f"模型预测: {pred}")
    #     print(f"当前F1分数: {current_f1:.4f}")

    # avg_f1 = f1_sum / len(samples)
    # print(f"\n===== 最终结果 =====")
    # print(f"MODEL={args.model} TYPE={args.model_type} F1={avg_f1:.6f} TOTAL={len(samples)}")
    # print("Evaluation complete.")


if __name__ == "__main__":
    main()