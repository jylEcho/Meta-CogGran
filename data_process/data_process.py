import pandas as pd
import os
import base64
import json

# 加载数据
data_path = './external/hpctmp/e1374390/hf_cache/hub/datasets--lmms-lab--MME/snapshots/d6c9023f017b564f7b3ccccf5348166bce8fdbcd/data/test-00002-of-00004-594798fd3f5b029c.parquet'
data = pd.read_parquet(data_path)

# 提取10条数据
subset_data = data.head(10)

# 转换为字典格式并处理嵌套字典的 bytes 类型
processed_data = []
for _, row in subset_data.iterrows():
    # 检查 image 字段是否是字典
    image_data = row['image']
    if isinstance(image_data, dict) and 'bytes' in image_data:
        # 提取 bytes 并转为 Base64 编码
        image_base64 = base64.b64encode(image_data['bytes']).decode('utf-8')
    else:
        # 如果不是嵌套字典，直接使用原数据
        image_base64 = image_data

    # 修改 question 字段，确保每个问题以 \n<image> 结尾
    question = row['question']
    if not question.endswith("\n<image>"):
        question += "\n<image>"

    processed_data.append({
        'question': question,
        'image': image_base64,  # 保存 Base64 编码的图片数据
        'answer': row['answer']
    })

# 创建目标文件夹
output_folder = 'evaldata'
os.makedirs(output_folder, exist_ok=True)

# 保存到文件，以 JSON 格式保存
output_file_path = os.path.join(output_folder, 'mme.json')
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print(f"数据已保存到 {output_file_path}")

# 加载一个 .parquet 文件
# data_path = './external/hpctmp/e1374390/hf_cache/hub/datasets--lmms-lab--MME/snapshots/d6c9023f017b564f7b3ccccf5348166bce8fdbcd/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet'
# data = pd.read_parquet(data_path)

# # 数据预览
# print(data.head(10))
# print(len(data))

# # 加载一个 .parquet 文件
# data_path = './external/hpctmp/e1374390/hf_cache/hub/datasets--lmms-lab--MME/snapshots/d6c9023f017b564f7b3ccccf5348166bce8fdbcd/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet'
# data = pd.read_parquet(data_path)

# # 数据预览
# print(data.head(100))
# print(len(data))

# data_path = './external/hpctmp/e1374390/hf_cache/hub/datasets--lmms-lab--MME/snapshots/d6c9023f017b564f7b3ccccf5348166bce8fdbcd/data/test-00002-of-00004-594798fd3f5b029c.parquet'
# data = pd.read_parquet(data_path)

# # 数据预览
# print(data.columns)
# print(data.head(10))
# print(len(data))

# data_path = './external/hpctmp/e1374390/hf_cache/hub/datasets--lmms-lab--MME/snapshots/d6c9023f017b564f7b3ccccf5348166bce8fdbcd/data/test-00003-of-00004-53ae1794f93b1e35.parquet'
# data = pd.read_parquet(data_path)

# # 数据预览
# print(data.head(100))
# print(len(data))