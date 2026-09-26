#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单模型批量评测，实时累加，最后给平均分
"""
# from tomlkit import item
import json, os, base64, tqdm
from openai import OpenAI

# ========== 1. 配置 ==========
client = OpenAI(
    base_url="API_ENDPOINT_NOT_CONFIGURED",
    api_key="REPLACE_WITH_ENV_VAR"
)

IMAGE_ROOT = "./external/granulon/Eval_CC12M/dinov_qwen3-8B_10/images"          # 图片根目录
MODEL_JSON = "./external/granulon/Eval_CC12M/dinov_qwen3-8B_10/eval.json"     # 待测模型输出
SAVE_FILE  = "./external/granulon/Eval_CC12M/dinov_qwen3-8B_10/dinov_llama_10.jsonl"  # 逐条结果
# =============================

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def build_prompt(pred: str) -> str:
    return f"""
You are an objective evaluator for AI image descriptions.
Evaluate the following MODEL OUTPUT against the PROVIDED IMAGE,
and assign TWO INDEPENDENT scores (0-100 Accuracy, 0-80 Hallucination).

1. ACCURACY SCORE (0-100): how well the text matches the actual image content.
2. HALLUCINATION SCORE (0-80): how much content is NOT present in the image.

MODEL OUTPUT: {pred}

Return ONLY a valid JSON object:
{{"accuracy_score": int, "hallucination_score": int}}
"""

def call_gpt4o(image_b64: str, prompt: str):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.1
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print("GPT-4o error:", e)
        return None

def main():
    with open(MODEL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    fw = open(SAVE_FILE, "w", encoding="utf-8")

    # 累加器
    total_acc = 0
    total_hal = 0
    valid_cnt = 0

    for idx, item in enumerate(tqdm.tqdm(data, desc="Eval")):
        # img_path = item["image_path"]
        # if not os.path.isabs(img_path):
        #     img_path = os.path.join(IMAGE_ROOT, os.path.basename(img_path))
        img_path = os.path.join(IMAGE_ROOT, os.path.basename(item["image_path"]))
        
        b64 = encode_image(img_path)
        prompt = build_prompt(item["prediction"])
        scores = call_gpt4o(b64, prompt)

        if scores is None:          # 异常用 -1 标记，不计入平均
            scores = {"accuracy_score": -1, "hallucination_score": -1}

        # 写单行结果
        out = {"sample_id": idx, "image": img_path, "scores": scores}
        fw.write(json.dumps(out, ensure_ascii=False) + "\n")

        # 累加有效分
        if scores["accuracy_score"] != -1:
            total_acc += scores["accuracy_score"]
            total_hal += scores["hallucination_score"]
            valid_cnt += 1

    fw.close()

    # ========== 最终结果 ==========
    if valid_cnt == 0:
        print("未获得任何有效评分！")
        return

    avg_acc = total_acc / valid_cnt
    avg_hal = total_hal / valid_cnt

    print("\n==========  平均得分  ==========")
    print(f"Average Accuracy    = {avg_acc:.2f} / 100")
    print(f"Average Hallucination = {avg_hal:.2f} / 80")
    print(f"有效样本数 / 总样本数 = {valid_cnt} / {len(data)}")
    print("逐条结果已写入:", SAVE_FILE)

if __name__ == "__main__":
    main()