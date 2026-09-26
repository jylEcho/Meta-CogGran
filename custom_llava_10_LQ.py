from typing import Any
import torch
import torch.nn as nn
# from sklearn.cluster import MiniBatchKMeans   # 用 MiniBatch 更快，数据量小可直接 KMeans
import numpy as np
# 兼容不同 transformers 版本的导入路径
try:
    from transformers.models.llava.modeling_llava import (
        LlavaModel as HF_LlavaModel,
        LlavaForConditionalGeneration as HF_LlavaForConditionalGeneration,
        LlavaPreTrainedModel,
        LlavaCausalLMOutputWithPast
    )
except Exception:
    from transformers import (
        LlavaModel as HF_LlavaModel,
        LlavaForConditionalGeneration as HF_LlavaForConditionalGeneration,
        LlavaPreTrainedModel,
        LlavaCausalLMOutputWithPast
    )

# Processor imports across versions
try:
    from transformers.models.llava.processing_llava import (
        LlavaProcessor as HF_LlavaProcessor,
        LlavaProcessorKwargs,
    )
except Exception:
    from transformers import LlavaProcessor as HF_LlavaProcessor  # type: ignore
    try:
        from transformers.models.llava.processing_llava import LlavaProcessorKwargs  # type: ignore
    except Exception:
        LlavaProcessorKwargs = None  # type: ignore

try:
    from transformers.feature_extraction_utils import BatchFeature
except Exception:
    BatchFeature = dict  # very old versions fallback

from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from typing import Optional, Union

CLASS_NUM = 10  # 聚类类别数

class LlavaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_mm_token_type_ids": False},
        "images_kwargs": {},
    }

class CustomLlavaModel(HF_LlavaModel):
    """
    覆写 get_image_features，做两点改动：
    1) 将 list 中的每个 tensor 从 [patch, feature] 变成 [1, patch, feature]
    2) 将 patch 从 201 裁到 196，删除最前面的 5 个 patch
    说明：
    - 上游返回通常是 list[Tensor]，每个 Tensor 形状为 [patch, feature]（或有时为 [?, patch, feature]）。
    - 这里统一处理为 [1, 196, feature]，且只删除“前面 5 个 patch”。
    """
    def get_image_features(self, *args: Any, **kwargs: Any):
        feats = super().get_image_features(*args, **kwargs)
        # print(">>> CustomLlavaModel.get_image_features called, original feats type:", type(feats))
        # print("    original feats:", feats if not isinstance(feats, list) else f"list of length {len(feats)}")
        # print("    original feats[0] shape:", feats[0].shape if isinstance(feats, list) and len(feats) > 0 else "N/A")
        def fix_one(t: torch.Tensor) -> torch.Tensor:
            if not isinstance(t, torch.Tensor):
                return t

            # 目标：最终形状 [1, 196, feature]
            if t.dim() == 2:
                # [patch, feature] -> 删除前 5 个 patch -> [patch-5, feature] -> unsqueeze 到 [1, patch-5, feature]
                # dinov3 删除前 5 个 patch；clip 删除前 1 个 patch；siglip 不删除
                if t.size(0) >= 50 and t.size(0) < 201 or t.size(0) == 257:
                    t = t[1:, :]
                if t.size(0) > 200 and t.size(0) < 256:
                    t = t[5:, :]
                t = t.unsqueeze(0)  # -> [1, patch, feature]
            else:
                # 非预期形状，直接返回
                print("Warning: Unexpected tensor shape in get_image_features:", t.shape)
                return t

            return t
        
        if isinstance(feats, list):
            feats = [fix_one(t) for t in feats]
            return feats
        else:
            raise TypeError("feats is not a list, unexpected.")

class CustomLlavaForConditionalGeneration(HF_LlavaForConditionalGeneration):
    """
    仅在 __init__ 中把 self.model = LlavaModel(config) 改为 CustomLlavaModel(config)。
    其余保持与上游一致，保证权重映射/保存/加载兼容。
    """

    _checkpoint_conversion_mapping = getattr(
        HF_LlavaForConditionalGeneration, "_checkpoint_conversion_mapping", {}
    )
    _tied_weights_keys = getattr(
        HF_LlavaForConditionalGeneration, "_tied_weights_keys", []
    )

    def __init__(self, config):
        # 不直接 super().__init__(config)，避免父类实例化原生 LlavaModel
        LlavaPreTrainedModel.__init__(self, config)
        self.model = CustomLlavaModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

class CustomLlavaProcessor(HF_LlavaProcessor):

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[LlavaProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            LlavaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            pixel_values = image_inputs["pixel_values"]
            height, width = get_image_size(to_numpy_array(pixel_values[0]))
            num_image_tokens = (height // self.patch_size) * (
                width // self.patch_size
            ) + self.num_additional_image_tokens
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            # 唯一改动：在原版基础上 +6
            # num_image_tokens += CLASS_NUM
            # num_image_tokens = 200 # if visual encoder is dinov2

            prompt_strings = []
            for sample in text:
                # sample = f"{sample}<think>\n\n</think>\n"
                # sample = sample.replace(self.image_token, self.image_token + "/no_think")
                sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                prompt_strings.append(sample)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)


# class CustomLlavaProcessor(HF_LlavaProcessor):
#     """
#     自定义 Processor：仅修改 __call__ 展开 <image> 的数量，在原本基础上 +15。
#     其它逻辑与上游保持一致。不在此处做“去掉前 5 个 patch”的工作。
#     """

#     def __init__(
#         self,
#         image_processor=None,
#         tokenizer=None,
#         patch_size=None,
#         vision_feature_select_strategy=None,
#         chat_template=None,
#         image_token: str = "<image>",
#         num_additional_image_tokens: int = 0,
#         **kwargs,
#     ):
#         # 兜底从 kwargs 中恢复可能由 from_pretrained 注入的配置
#         if patch_size is None:
#             patch_size = kwargs.pop("patch_size", None)
#         if vision_feature_select_strategy is None:
#             vision_feature_select_strategy = kwargs.pop("vision_feature_select_strategy", None)
#         # 如果 save_pretrained 写入了这些字段，也做兜底取值
#         image_token = kwargs.pop("image_token", image_token)
#         num_additional_image_tokens = kwargs.pop(
#             "num_additional_image_tokens", num_additional_image_tokens
#         )

#         # 与上游保持一致，先设置属性
#         self.patch_size = patch_size
#         print("CustomLlavaProcessor: patch_size =", self.patch_size)
#         self.num_additional_image_tokens = num_additional_image_tokens
#         self.vision_feature_select_strategy = vision_feature_select_strategy
#         self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
#         self.image_token_id = tokenizer.encode(self.image_token, add_special_tokens=False)[0]
    
#         # 调用父类初始化，确保父类属性（如 tokenizer、image_processor 等）被正确设置
#         super().__init__(image_processor, tokenizer, chat_template=chat_template)

#     def __call__(
#         self,
#         images: Optional[ImageInput] = None,
#         text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
#         audio=None,
#         videos=None,
#         **kwargs: Unpack[LlavaProcessorKwargs],
#     ) -> BatchFeature:

#         if images is None and text is None:
#             raise ValueError("You have to specify at least one of `images` or `text`.")

#         output_kwargs = super()._merge_kwargs(
#             LlavaProcessorKwargs,
#             tokenizer_init_kwargs=self.tokenizer.init_kwargs,
#             **kwargs,
#         )
#         if images is not None:
#             image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
#         else:
#             image_inputs = {}

#         if isinstance(text, str):
#             text = [text]
#         elif not isinstance(text, list) and not isinstance(text[0], str):
#             raise TypeError("Invalid input text. Please provide a string, or a list of strings")

#         # try to expand inputs in processing if we have the necessary parts
#         prompt_strings = text
#         if image_inputs.get("pixel_values") is not None:
#             # Replace the image token with the expanded image token sequence
#             pixel_values = image_inputs["pixel_values"]
#             height, width = get_image_size(to_numpy_array(pixel_values[0]))
#             num_image_tokens = (height // self.patch_size) * (
#                 width // self.patch_size
#             ) + self.num_additional_image_tokens
#             if self.vision_feature_select_strategy == "default":
#                 num_image_tokens -= 1

#             # 唯一改动：在原有基础上 +15
#             num_image_tokens += CLASS_NUM

#             prompt_strings = []
#             for sample in text:
#                 sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
#                 prompt_strings.append(sample)

#         return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
#         return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
#         text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
#         self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

#         if return_mm_token_type_ids:
#             array_ids = np.array(text_inputs["input_ids"])
#             mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
#             mm_token_type_ids[array_ids == self.image_token_id] = 1
#             text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

#         return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)