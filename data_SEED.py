from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import AutoProcessor

# 允许读取部分截断的图片，避免部分坏图直接崩
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor


class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        super().__init__()
        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_path = Path(data_dir)
        chat_file = data_path.joinpath("chat.json")
        image_dir = data_path.joinpath("images")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")
        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        conversations = cur_data.get("conversations")

        if conversations is None or len(conversations) < 2:
            raise ValueError(f"Bad conversations at index={index}: {cur_data}")

        human_input = conversations[0].get("value")
        chatbot_output = conversations[1].get("value")
        image_path = self.image_dir.joinpath(cur_data.get("image"))

        return human_input, chatbot_output, image_path


def normalize_image(raw_image):
    """
    把各种可能的图像格式统一成 RGB PIL.Image，
    避免 processor 无法推断通道维度。
    """
    if raw_image is None:
        raise ValueError("raw_image is None")

    if isinstance(raw_image, Image.Image):
        return raw_image.convert("RGB")

    if isinstance(raw_image, torch.Tensor):
        raw_image = raw_image.detach().cpu().numpy()

    if isinstance(raw_image, np.ndarray):
        if raw_image.ndim == 2:
            # 灰度图 H,W -> H,W,3
            raw_image = np.stack([raw_image] * 3, axis=-1)

        elif raw_image.ndim == 3:
            # 如果是 CHW，则转成 HWC
            if raw_image.shape[0] in [1, 3, 4] and raw_image.shape[-1] not in [1, 3, 4]:
                raw_image = np.transpose(raw_image, (1, 2, 0))
        else:
            raise ValueError(
                f"Unsupported image ndim: {raw_image.ndim}, shape={raw_image.shape}"
            )

        if raw_image.shape[-1] == 1:
            raw_image = np.repeat(raw_image, 3, axis=-1)
        elif raw_image.shape[-1] == 4:
            raw_image = raw_image[:, :, :3]
        elif raw_image.shape[-1] != 3:
            raise ValueError(f"Unsupported channel count: shape={raw_image.shape}")

        if raw_image.dtype != np.uint8:
            if np.issubdtype(raw_image.dtype, np.floating):
                # 若是 float，尝试缩放到 0~255
                if raw_image.max() <= 1.0:
                    raw_image = raw_image * 255.0
            raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)

        return Image.fromarray(raw_image).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(raw_image)}")


def build_qaimage(
    processor: AutoProcessor,
    q_text: str,
    a_text: str,
    image_path: Path,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        # 用 with 确保文件句柄及时释放
        with Image.open(image_path) as img:
            raw_image = img.convert("RGB")

        raw_image = normalize_image(raw_image)

        # 这里是你原来报错的位置
        inputs = processor(raw_image, prompt, return_tensors="pt")

        if "input_ids" not in inputs or "pixel_values" not in inputs:
            raise ValueError(
                f"Processor output missing keys. got keys={list(inputs.keys())}"
            )

    except Exception as e:
        print(f"[BAD IMAGE] image_path={image_path}")
        print(f"[BAD IMAGE] q_text={q_text}")
        print(f"[BAD IMAGE] error={repr(e)}")
        raise

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]

    return QaImageOutput(
        q_input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        a_input_ids=a_input_ids,
    )


class TrainLLavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int) -> None:
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
    ):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(
                    self.processor.tokenizer.eos_token_id,
                    dtype=q_input_ids.dtype,
                ).reshape(1, -1),
            ],
            axis=1,
        )

        labels = torch.concat(
            [
                torch.full(
                    q_input_ids.shape,
                    self.ingnore_index,
                    dtype=q_input_ids.dtype,
                ),
                a_input_ids,
                torch.tensor(
                    self.processor.tokenizer.eos_token_id,
                    dtype=a_input_ids.dtype,
                ).reshape(1, -1),
            ],
            axis=1,
        )

        return input_ids, labels

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            q_text, a_text, image_path = feature

            qaimage_output = build_qaimage(
                self.processor,
                q_text,
                a_text,
                image_path,
            )

            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids,
                qaimage_output.a_input_ids,
            )

            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                            dtype=value.dtype,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ],
            axis=0,
        )

        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                            dtype=value.dtype,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ],
            axis=0,
        )

        final_pixel_values = torch.concat(pixel_values, axis=0)

        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }