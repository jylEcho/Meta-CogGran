from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, LlavaProcessor, AutoImageProcessor
import pandas as pd
import torch
# from custom_llava import CustomLlavaForConditionalGeneration

# 定义数据类，用于存储问答图像输出的结构
@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor  # 问题文本对应的输入token ids
    pixel_values: torch.Tensor  # 图像像素值张量
    a_input_ids: torch.Tensor  # 答案文本对应的输入token ids


# 自定义数据集类，继承自PyTorch的Dataset
class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        """
            初始化数据集
        Args:
            dataset_dir (str): 数据集目录路径
        """
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)  # 构建数据集，返回对话数据和图像目录

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        """
        构建数据集

        Args:
            data_dir (str): 数据集目录路径

        Returns:
            Tuple[List[Dict], Path]: 对话数据列表和图像目录路径
        """
        data_path = Path(data_dir)   # 转换为Path对象
        chat_file = data_path.joinpath("chat.json")  # 对话数据文件路径
        image_dir = data_path.joinpath("images")  # 图像目录路径

        # 读取JSON文件并转换为字典列表
        chat_data = pd.read_json(chat_file).to_dict(orient="records")  # [{'id': 'GCC_train_002582585', 'image': 'GCC_train_002582585.jpg', 'conversations': [...]}, ...] 

        return chat_data, image_dir

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        """
        获取指定索引的样本

        Args:
            index (int): 样本索引

        Returns:
            Tuple[str, str, Path]: 包含问题文本、答案文本和图像路径的元组
        """
        # 获取当前数据项
        cur_data = self.chat_data[index]  # {'id': 'GCC_train_002109690', 'image': 'GCC_train_002109690.jpg', 'conversations': [{...}, {...}]}
        # 获取对话内容
        conversations = cur_data.get("conversations")  # [{'from': 'human', 'value': 'Offer a succinct explanation of the picture presented.\n<image>'}, {'from': 'gpt', 'value': "it 's appropriate for teens to want to spend more time with their peers than their parents as they get older ."}]

        # 提取人类输入(即问题)【其中 <image> 是图像的占位符】
        human_input = conversations[0].get("value")   # 'Offer a succinct explanation of the picture presented.\n<image>'
        # 提取标准的机器人输出(即答案)
        chatbot_output = conversations[1].get("value")  # "it 's appropriate for teens to want to spend more time with their peers than their parents as they get older ."

        # 构建图像路径
        image_path = self.image_dir.joinpath(cur_data.get("image"))  # PosixPath('train_llava/LLaVA-CC3M-Pretrain-595K/images/GCC_train_002109690.jpg')
        return human_input, chatbot_output, image_path


def build_qaimage(processor: AutoProcessor, q_text: str, a_text: str, image_path: Path):
    """
    构建问答图像输入数据

    Args:
        processor (AutoProcessor): 处理器对象
        q_text (str): 问题文本
        a_text (str): 答案文本
        image_path (Path): 图像路径

    Returns:
        QaImageOutput: 包含输入数据的QaImageOutput对象
    """
    # 构建对话消息模板
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]

     # 应用对话模板生成prompt
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nOffer a succinct explanation of the picture presented.\n<image><|im_end|>\n<|im_start|>assistant\n"

    raw_image = Image.open(image_path)  # 打开图像文件 <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=224x224 at 0x7BD9094E4BE0>
    inputs = processor(raw_image, prompt, return_tensors="pt")  # 使用处理器处理文本和图像 inputs.keys(): dict_keys(['input_ids', 'attention_mask', 'pixel_values'])


    # 处理答案文本生成输入的 token id
    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]  # tensor([[  275,   364,    82,  8311,   369, 26202,   311,  1366,   311,  8329, 803,   882,   448,   862, 25029,  1091,   862,  6562,   438,   807, 633,  9014,   659]]) 

    # 返回包含所有输入数据的对象
    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    return res


class TrainLLavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int) -> None:
        """
        初始化数据整理器

        Args:
            processor (AutoProcessor): 处理器对象
            IGNORE_INDEX (int): 忽略索引值（通常为-100）
        """
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
        # pixel_values: torch.Tensor,
    ):
        """
        转换单个样本为模型输入格式

        Args:
            q_input_ids (torch.Tensor): 问题的 token ids
            a_input_ids (torch.Tensor): 答案的 token ids

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 拼接后的输入 token ids 和标签 labels
        """
        # 拼接问题、答案和结束的 token ids
        input_ids = torch.concat(
            [
                q_input_ids,  # 其中, 151646 就是 <image> 的 token id → tensor([[151644, 8948, 198, 2610, ..., 624, 151646, 151645, 198, 151644, 77091, 198]])
                a_input_ids,  # tensor([[275, 364, 82, 8311, ..., 9014, 659]])
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        ) # 得到的结果: tensor([[151644, 8948, 198, 2610, ..., 624, 151646, 151645, 198, 151644, 77091, 198, 275, 364, 82, 8311, ..., 9014,    659, 151645]])

        # 构建标签 labels ：问题部分用 IGNORE_INDEX符号 填充，答案部分保留不变
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingnore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        ) # 得到的结果: tensor([[-100, -100, ..., -100, -100, -100, 275, 364, 82, 8311, ..., 9014, 659, 151645]])

        return input_ids, labels

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        """
        处理批次数据。每次取一个 batch_size 的数据时，会调用这个函数

        Args:
            features (List): 一个列表装着的 batch
        Returns:
            Dict[str, torch.Tensor]: 包含一个 batch_size 数据的字典
        """
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            # 构建单个样本的输入数据
            qaimage_output = build_qaimage(
                self.processor, feature[0], feature[1], feature[2]
            )

            # 转换为模型的输入格式
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )

            # 记录最大长度
            max_input_len_list.append(temp_input_ids.shape[1])  # 比如: [53, 59]
            # 保存中间结果
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        # 获取每个 batch 的最大长度
        max_input_len = max(max_input_len_list)

         # 填充输入 token ids 到统一长度
        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )  # input_ids_list: [tensor([[151644, 8948, ..., 659, 151645]]), tensor([[151644, 8948, 198, 2610, 525, ..., 25956, 151645]])] → final_input_ids: [tensor([[151643, 151643, ..., 151643, 151644, 8948, ..., 659, 151645]]), tensor([[151644, 8948, 198, 2610, 525, ..., 25956, 151645]])]

        # 填充 标签labels 的 token ids 到统一长度
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )  # 同理获取 final_input_ids 的流程, 不过padding符号变为了-100, 而 final_input_ids 的padding符号为 151643

        # 拼接图像的像素值
        final_pixel_values = torch.concat(pixel_values, axis=0)
        # 构建注意力掩码
        attention_mask = torch.ones_like(final_input_ids)  # tensor([[1, 1, 1, ... 1, 1, 1], [1, 1, 1, 1, 1, 1, ..., 1, 1, 1]])
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0  # tensor([[0, 0, ..., 0, 1, 1, ... 1, 1, 1], [1, 1, ..., 1, 1, 1]])

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }


# if __name__ == "__main__":
#     # 1. 测试 LlavaDataset 类
#     data_dir = "./external/e1374390/work/dataset/LLaVA-CC3M-Pretrain-595K"  # 数据集路径（注意：当前路径可能需要调整）
#     llavadataset = LlavaDataset(data_dir)   # 创建数据集实例
#     all_data_len = len(llavadataset)   # 打印数据集总样本数
#     print("all_data_len:", all_data_len)  # all_data_len: 595375
#     one_data_example = llavadataset[168]  # 打印第168个样本示例 
#     print("one_data_example:", one_data_example)  # one_data_example: ('Offer a succinct explanation of the picture presented.\n<image>', "it 's appropriate for teens to want to spend more time with their peers than their parents as they get older .", PosixPath('train_llava/LLaVA-CC3M-Pretrain-595K/images/GCC_train_002109690.jpg'))

#     # 2. 测试 TrainLLavaModelCollator 类
#     model_name_or_path = "./external/e1374390/work/dinov_model_qwen2.5-1.5b"  # 定义模型的路径，这里指定了本地存储的模型目录
#     model_name_or_path_new = './external/e1374390/work/dinov_model_qwen2.5-1.5b/trained'
#     llava_processor = LlavaProcessor.from_pretrained(model_name_or_path)  # 处理器会加载模型对应的分词器和图像处理器等信息
#     tlmc = TrainLLavaModelCollator(processor=llava_processor, IGNORE_INDEX=-100)
#     # 自行构建一个 batch_size = 2 的批次
#     one_origin_batch = [llavadataset[168]]
#     # 得到的 one_input_batch 将会传给模型进行训练
#     one_input_batch = tlmc(one_origin_batch)  
#     # print("one_input_batch.keys():", one_input_batch.keys())  # 看一下存在哪些键 → dict_keys(['input_ids', 'labels', 'pixel_values', 'attention_mask'])
#     # print("one_input_batch:", one_input_batch)
#     # print("one_input_batch[pixel_values].shape:", one_input_batch["pixel_values"].shape)  # one_input_batch[pixel_values].shape: torch.Size([1, 3, 224, 224])
#     llava_model = CustomLlavaForConditionalGeneration.from_pretrained(model_name_or_path_new,
#                                                                       dtype=torch.bfloat16,
#                                                                       device_map='auto')  # 加载预训练模型
#     for tk in one_input_batch.keys():
#         one_input_batch[tk] = one_input_batch[tk].to(llava_model.device)  # 将输入数据移动到模型所在的设备上（如GPU）

#     outputs = llava_model(**one_input_batch)  # 前向传播，计算模型输出
#     # print("outputs:", outputs)  # 打印模型输出，通常包含损失值
#     print("outputs.loss:", outputs.loss)  # 打印损失值


#     # 3. 使用训练过的模型做一次推理，看看效果
#     llava_model.eval()

#     # 取一条样本做推理
#     infer_idx = 0  # 你也可以改成其他索引
#     q_text, gt_answer, image_path = llavadataset[infer_idx]
#     print("path:", image_path)
#     # 确保问题中包含图像占位符
#     if "<image>" not in q_text:
#         raise ValueError("<image> not in q_text")
#     else:
#         print(f'q_text: "{q_text}"')
#     # 构建与训练一致的对话模板和prompt
#     infer_messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": q_text},
#     ]
#     infer_prompt = llava_processor.tokenizer.apply_chat_template(
#         infer_messages, tokenize=False, add_generation_prompt=True
#     )
#     print(f'infer_prompt: "{infer_prompt}"')
#     # 读入图像并用processor处理
#     infer_image = Image.open(image_path).convert("RGB")
#     infer_inputs = llava_processor(infer_image, infer_prompt, return_tensors="pt")
#     print("infer_inputs.keys():", infer_inputs.keys())  # dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
#     # 移动到模型设备
#     infer_inputs = {k: v.to(llava_model.device) for k, v in infer_inputs.items()}

#     tok = llava_processor.tokenizer
#     print("eos_token_id:", tok.eos_token_id)
#     print("pad_token_id:", tok.pad_token_id)    
#     # print("tok: ", tok)
#     print("tok encode: ", tok.encode("<image>"))
#     # print(one_input_batch["input_ids"])
#     # 生成
#     with torch.no_grad():
#         generated_ids = llava_model.generate(
#             **infer_inputs,
#             max_new_tokens=128,
#             do_sample=False,
#         )

#     # 只截取新生成的部分进行解码
#     prompt_len = infer_inputs["input_ids"].shape[1]
#     gen_only = generated_ids[0][prompt_len:]
#     gen_text = llava_processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

#     print("\n=== Inference Sample ===")
#     print("Image path:", image_path)
#     print("Question:", q_text.replace("\n", " "))
#     print("Ground Truth:", gt_answer)
#     print("Model Output:", gen_text)