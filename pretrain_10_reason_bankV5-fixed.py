import logging
import os
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import Trainer, TrainingArguments

from callback import ProjectorGradMonitor
from custom_llava_10_bankV5 import (
    CustomLlavaForConditionalGeneration,
    CustomLlavaProcessor,
)

try:
    from custom_llava_10_bankV5 import LlavaProcessorKwargs as CustomLlavaProcessorKwargs
except Exception:
    CustomLlavaProcessorKwargs = None
from data import LlavaDataset, TrainLLavaModelCollator
from util import print_trainable_parameters

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


class _FallbackLlavaProcessorKwargs:
    _defaults = {
        "text_kwargs": {"padding": False, "return_mm_token_type_ids": False},
        "images_kwargs": {},
    }


def _get_processor_kwargs_class(processor):
    cls = getattr(processor, "__class__", type(processor))
    candidate = getattr(cls, "LlavaProcessorKwargs", None)
    if candidate is not None and hasattr(candidate, "_defaults"):
        return candidate
    if CustomLlavaProcessorKwargs is not None and hasattr(CustomLlavaProcessorKwargs, "_defaults"):
        return CustomLlavaProcessorKwargs
    return _FallbackLlavaProcessorKwargs


class LossLoggingCallback(transformers.TrainerCallback):
    """在 Trainer 的 on_log 中把包含 'loss' 的日志写入 logging"""

    def __init__(self):
        self.logger = logging.getLogger("training.loss")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = getattr(state, "global_step", None)
        for k, v in (logs.items() if isinstance(logs, dict) else []):
            if isinstance(k, str) and "loss" in k:
                try:
                    self.logger.info("step=%s %s=%.6f", step, k, float(v))
                except Exception:
                    self.logger.info("step=%s %s=%s", step, k, v)


@dataclass
class ModelArguments:
    """模型参数配置类"""

    model_name_or_path: Optional[str] = field(
        default="none",
        metadata={"help": "预训练模型的路径或名称"},
    )
    train_type: Optional[str] = field(
        default="none",
        metadata={
            "help": """
            1. use_lora: 使用lora训练
            2. none: 全量参数训练
            3. freeze_vision: 冻结 vision_tower
            4. tune_mm_mlp_adapter: 只训练 projector
            """
        },
    )
    semantic_bank_path: Optional[str] = field(
        default=None,
        metadata={"help": "semantic bank 路径"},
    )
    semantic_cluster_num: int = field(
        default=10,
        metadata={"help": "semantic cluster 数"},
    )
    global_bank_topk: int = field(
        default=4,
        metadata={"help": "global bank topk"},
    )
    entity_bank_topk: int = field(
        default=6,
        metadata={"help": "entity bank topk"},
    )
    layout_bank_topk: int = field(
        default=4,
        metadata={"help": "layout bank topk"},
    )
    relation_bank_topk: int = field(
        default=4,
        metadata={"help": "relation bank topk"},
    )
    layout_grid_size: int = field(
        default=4,
        metadata={"help": "layout 网格大小"},
    )
    relation_near_threshold: float = field(
        default=0.22,
        metadata={"help": "relation near 阈值"},
    )
    relation_overlap_threshold: float = field(
        default=0.10,
        metadata={"help": "relation overlap 阈值"},
    )
    relation_direction_margin: float = field(
        default=0.08,
        metadata={"help": "relation direction margin"},
    )
    relation_max_pairs: int = field(
        default=64,
        metadata={"help": "relation 最大 pair 数"},
    )
    patch_drop: int = field(
        default=5,
        metadata={"help": "和 custom_llava 对齐的 patch drop"},
    )
    bank_retrieval_temperature: float = field(
        default=1.0,
        metadata={"help": "bank retrieval temperature"},
    )
    use_bank: bool = field(
        default=True,
        metadata={"help": "是否启用 semantic bank"},
    )


@dataclass
class DataArguments:
    """数据参数配置类"""

    data_path: str = field(
        default=None,
        metadata={"help": "训练数据路径"},
    )


def _expected_image_seq_length(modelargs: ModelArguments) -> int:
    return (
        196
        + int(modelargs.global_bank_topk)
        + int(modelargs.entity_bank_topk)
        + int(modelargs.layout_bank_topk)
        + int(modelargs.relation_bank_topk)
    )


def patch_custom_model_runtime(model) -> None:
    """
    训练时动态修复 custom_llava_10_bankV5.py 中把 layout / relation bank 注释掉的问题。
    不改原文件，也能保证四个 bank 都真实参与前向。
    """
    mm = getattr(model, "model", None)
    if mm is None:
        raise ValueError("model.model 不存在，无法 patch get_image_features")

    def fixed_get_image_features(self, *args, **kwargs):
        feats = super(type(self), self).get_image_features(*args, **kwargs)
        if not isinstance(feats, list):
            raise TypeError(f"Expected image features as a list, got {type(feats)}")

        outputs = []
        for t in feats:
            if not torch.is_tensor(t):
                raise TypeError(f"Each image feature must be Tensor, got {type(t)}")

            orig_dtype = t.dtype
            t = self._fix_one(t)
            cluster_tokens = self._build_online_cluster_tokens(t)
            cluster_coords = self._build_cluster_coords(t, cluster_tokens)

            global_prior = self._retrieve_global_prior(t)
            entity_prior = self._retrieve_entity_prior(cluster_tokens)
            layout_prior = self._retrieve_layout_prior(cluster_tokens, cluster_coords)
            relation_prior = self._retrieve_relation_prior(cluster_tokens, cluster_coords)

            tokens = [t]
            if global_prior is not None and global_prior.shape[1] > 0:
                tokens.append(global_prior)
            if entity_prior is not None and entity_prior.shape[1] > 0:
                tokens.append(entity_prior)
            if layout_prior is not None and layout_prior.shape[1] > 0:
                tokens.append(layout_prior)
            if relation_prior is not None and relation_prior.shape[1] > 0:
                tokens.append(relation_prior)

            new_feats = torch.cat(tokens, dim=1).contiguous()
            if new_feats.dtype != orig_dtype:
                new_feats = new_feats.to(orig_dtype)
            outputs.append(new_feats)
        return outputs

    mm.get_image_features = types.MethodType(fixed_get_image_features, mm)
    logger.info("[PATCH MODEL] get_image_features 已开启四个 bank 拼接")


def patch_custom_processor_runtime(processor, modelargs: ModelArguments):
    """
    修复 processor 里 <image> token 展开长度没把 layout / relation 算进去的问题。

    关键点：不能只给 *实例* 绑定 __call__，因为 `obj(...)` 这类特殊方法查找走的是 type(obj)。
    如果只写 `processor.__call__ = ...`，实际调用时仍可能落回类上的旧实现，
    最终把 <image> 只展开成 196+4+6=206 个 token，而模型前向真实返回 214 个特征，
    于是触发 `Image features and image tokens do not match`。
    """
    processor.num_cluster_tokens = int(modelargs.semantic_cluster_num)
    processor.global_bank_topk = int(modelargs.global_bank_topk)
    processor.entity_bank_topk = int(modelargs.entity_bank_topk)
    processor.layout_bank_topk = int(modelargs.layout_bank_topk)
    processor.relation_bank_topk = int(modelargs.relation_bank_topk)

    processor_cls = processor.__class__

    def fixed_call(self, images=None, text=None, audio=None, videos=None, **kwargs):
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            _get_processor_kwargs_class(self),
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = (
            self.image_processor(images, **output_kwargs["images_kwargs"])
            if images is not None
            else {}
        )

        if text is None:
            prompt_strings = None
        elif isinstance(text, str):
            prompt_strings = [text]
        elif isinstance(text, list):
            if len(text) == 0:
                prompt_strings = text
            elif isinstance(text[0], str):
                prompt_strings = text
            else:
                raise TypeError("Invalid input text: expected list[str]")
        else:
            raise TypeError("Invalid input text")

        if prompt_strings is not None and image_inputs.get("pixel_values") is not None:
            add_global = int(getattr(self, "global_bank_topk", 0))
            add_entity = int(getattr(self, "entity_bank_topk", 0))
            add_layout = int(getattr(self, "layout_bank_topk", 0))
            add_relation = int(getattr(self, "relation_bank_topk", 0))
            num_image_tokens = 196 + add_global + add_entity + add_layout + add_relation
            prompt_strings = [
                sample.replace(self.image_token, self.image_token * num_image_tokens)
                for sample in prompt_strings
            ]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", False
        )

        text_inputs = self.tokenizer(
            prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None
        )
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        try:
            import numpy as np
            from transformers.feature_extraction_utils import BatchFeature
        except Exception:
            np = None
            BatchFeature = dict

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        if return_tensors is not None:
            text_inputs = BatchFeature(data=text_inputs, tensor_type=return_tensors)
            image_inputs = BatchFeature(data=image_inputs, tensor_type=return_tensors)

        if hasattr(text_inputs, "update"):
            text_inputs.update(image_inputs)
            return text_inputs

        text_inputs.update(image_inputs)
        return BatchFeature(data=text_inputs)

    setattr(processor_cls, "__call__", fixed_call)
    logger.info("[PATCH PROCESSOR] %s.__call__ 已改为四个 bank 版本，expected_image_tokens=%s", processor_cls.__name__, _expected_image_seq_length(modelargs))
    return processor


def _apply_custom_config(model, modelargs: ModelArguments) -> None:
    """
    把 CLI 参数强制写进 model.config，避免 from_pretrained 读到旧 config.json。
    """
    if not hasattr(model, "config") or model.config is None:
        raise ValueError("model.config 不存在，无法注入 semantic bank 配置")

    cfg = model.config

    logger.info("[CONFIG BEFORE] semantic_bank_path=%s", getattr(cfg, "semantic_bank_path", None))
    logger.info(
        "[CONFIG BEFORE] g=%s e=%s l=%s r=%s image_seq_length=%s",
        getattr(cfg, "global_bank_topk", None),
        getattr(cfg, "entity_bank_topk", None),
        getattr(cfg, "layout_bank_topk", None),
        getattr(cfg, "relation_bank_topk", None),
        getattr(cfg, "image_seq_length", None),
    )

    cfg.semantic_bank_path = modelargs.semantic_bank_path
    cfg.num_cluster_tokens = int(modelargs.semantic_cluster_num)
    cfg.global_bank_topk = int(modelargs.global_bank_topk)
    cfg.entity_bank_topk = int(modelargs.entity_bank_topk)
    cfg.layout_bank_topk = int(modelargs.layout_bank_topk)
    cfg.relation_bank_topk = int(modelargs.relation_bank_topk)
    cfg.layout_grid_size = int(modelargs.layout_grid_size)
    cfg.relation_near_threshold = float(modelargs.relation_near_threshold)
    cfg.relation_overlap_threshold = float(modelargs.relation_overlap_threshold)
    cfg.relation_direction_margin = float(modelargs.relation_direction_margin)
    cfg.relation_max_pairs = int(modelargs.relation_max_pairs)
    cfg.patch_drop = int(modelargs.patch_drop)
    cfg.bank_retrieval_temperature = float(modelargs.bank_retrieval_temperature)
    cfg.use_bank = bool(modelargs.use_bank)
    cfg.use_cache = False
    cfg.image_seq_length = _expected_image_seq_length(modelargs)

    logger.info("[CONFIG AFTER] semantic_bank_path=%s", cfg.semantic_bank_path)
    logger.info(
        "[CONFIG AFTER] num_cluster_tokens=%s global_bank_topk=%s entity_bank_topk=%s "
        "layout_bank_topk=%s relation_bank_topk=%s image_seq_length=%s "
        "layout_grid_size=%s relation_near_threshold=%s relation_overlap_threshold=%s "
        "relation_direction_margin=%s relation_max_pairs=%s patch_drop=%s temperature=%s use_bank=%s",
        cfg.num_cluster_tokens,
        cfg.global_bank_topk,
        cfg.entity_bank_topk,
        cfg.layout_bank_topk,
        cfg.relation_bank_topk,
        cfg.image_seq_length,
        cfg.layout_grid_size,
        cfg.relation_near_threshold,
        cfg.relation_overlap_threshold,
        cfg.relation_direction_margin,
        cfg.relation_max_pairs,
        cfg.patch_drop,
        cfg.bank_retrieval_temperature,
        cfg.use_bank,
    )

    if cfg.use_bank:
        if cfg.semantic_bank_path is None:
            raise ValueError("use_bank=True，但 semantic_bank_path 为 None")
        if not os.path.exists(cfg.semantic_bank_path):
            raise FileNotFoundError(f"semantic_bank_path 不存在: {cfg.semantic_bank_path}")


def _reload_semantic_bank_if_needed(model) -> None:
    """
    from_pretrained 初始化时，CustomLlavaModel.__init__ 可能先按旧 config 路径加载了旧 bank。
    这里在 config 被覆盖后，强制按新路径重新加载一次，并同步四类 bank 的运行时参数。
    """
    inner_model = getattr(model, "model", None)
    if inner_model is None:
        raise ValueError("model.model 不存在，无法重新加载 semantic bank")

    if not hasattr(inner_model, "_load_semantic_bank"):
        raise ValueError("CustomLlavaModel 缺少 _load_semantic_bank 方法")

    cfg = model.config
    use_bank = bool(getattr(cfg, "use_bank", True))
    bank_path = getattr(cfg, "semantic_bank_path", None)

    inner_model.use_bank = use_bank
    inner_model.semantic_bank_path = bank_path
    inner_model.num_cluster_tokens = int(getattr(cfg, "num_cluster_tokens", 10))
    inner_model.global_bank_topk = int(getattr(cfg, "global_bank_topk", 4))
    inner_model.entity_bank_topk = int(getattr(cfg, "entity_bank_topk", 6))
    inner_model.layout_bank_topk = int(getattr(cfg, "layout_bank_topk", 2))
    inner_model.relation_bank_topk = int(getattr(cfg, "relation_bank_topk", 2))
    inner_model.layout_grid_size = int(getattr(cfg, "layout_grid_size", 4))
    inner_model.relation_near_threshold = float(
        getattr(cfg, "relation_near_threshold", 0.22)
    )
    inner_model.relation_overlap_threshold = float(
        getattr(cfg, "relation_overlap_threshold", 0.10)
    )
    inner_model.relation_direction_margin = float(
        getattr(cfg, "relation_direction_margin", 0.08)
    )
    inner_model.relation_max_pairs = int(getattr(cfg, "relation_max_pairs", 64))
    inner_model.bank_retrieval_temperature = float(
        getattr(cfg, "bank_retrieval_temperature", 1.0)
    )

    if use_bank:
        if bank_path is None:
            raise ValueError("重新加载 semantic bank 时发现 semantic_bank_path 为 None")
        if not os.path.exists(bank_path):
            raise FileNotFoundError(f"重新加载 semantic bank 时路径不存在: {bank_path}")

        logger.info("[RELOAD BANK] path=%s", bank_path)
        inner_model._load_semantic_bank(bank_path)

        gb = getattr(inner_model, "_semantic_global_bank", None)
        eb = getattr(inner_model, "_semantic_entity_bank", None)
        lb = getattr(inner_model, "_semantic_layout_cell_to_bank", None)
        rb = getattr(inner_model, "_semantic_relation_type_to_bank", None)

        logger.info("[RELOAD BANK] global shape=%s", None if gb is None else tuple(gb.shape))
        logger.info("[RELOAD BANK] entity shape=%s", None if eb is None else tuple(eb.shape))
        logger.info("[RELOAD BANK] layout cell count=%s", 0 if lb is None else len(lb))
        logger.info("[RELOAD BANK] relation type count=%s", 0 if rb is None else len(rb))
    else:
        logger.info("[RELOAD BANK] use_bank=False, skip loading semantic bank")
        inner_model._semantic_global_bank = None
        inner_model._semantic_entity_bank = None
        inner_model._semantic_layout_cell_to_bank = {}
        inner_model._semantic_relation_type_to_bank = {}


def _sync_processor_attrs(processor, modelargs: ModelArguments) -> None:
    """
    processor 里 <image> token 展开数必须和模型一致。
    """
    processor.num_cluster_tokens = int(modelargs.semantic_cluster_num)
    processor.global_bank_topk = int(modelargs.global_bank_topk)
    processor.entity_bank_topk = int(modelargs.entity_bank_topk)
    processor.layout_bank_topk = int(modelargs.layout_bank_topk)
    processor.relation_bank_topk = int(modelargs.relation_bank_topk)

    logger.info(
        "[PROCESSOR] num_cluster_tokens=%s global_bank_topk=%s entity_bank_topk=%s "
        "layout_bank_topk=%s relation_bank_topk=%s expected_image_tokens=%s",
        processor.num_cluster_tokens,
        processor.global_bank_topk,
        processor.entity_bank_topk,
        processor.layout_bank_topk,
        processor.relation_bank_topk,
        _expected_image_seq_length(modelargs),
    )




def _sanity_check_processor_model_alignment(model, processor, modelargs: ModelArguments) -> None:
    """
    在真正训练前，检查 processor 展开的 <image> token 数是否与模型返回的 image features 一致。
    这里只做轻量级静态检查，不依赖真实图片。
    """
    expected = _expected_image_seq_length(modelargs)
    cfg_seq = int(getattr(model.config, "image_seq_length", -1))
    if cfg_seq != expected:
        raise RuntimeError(
            f"model.config.image_seq_length={cfg_seq}, expected={expected}"
        )

    prompt = f"prefix {processor.image_token} suffix"
    enc = processor(text=prompt, images=[torch.zeros(3, 224, 224)], return_tensors=None)
    input_ids = enc["input_ids"][0] if isinstance(enc["input_ids"], list) else enc["input_ids"]
    image_token_id = processor.image_token_id
    image_token_count = sum(1 for x in input_ids if x == image_token_id)

    if image_token_count != expected:
        raise RuntimeError(
            "processor 与模型 image token 数不一致: "
            f"processor expanded {image_token_count}, expected {expected}. "
            "请检查 processor.__call__ 是否真的被 patch 到类级别。"
        )

    logger.info(
        "[SANITY] processor-model aligned: image_token_count=%s image_seq_length=%s",
        image_token_count,
        expected,
    )

def _sanity_check_four_bank_runtime(model, modelargs: ModelArguments) -> None:
    """
    不跑真实图片，只检查关键属性是否同步到位。
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError("模型内部 model 缺失")

    for name, expected in [
        ("global_bank_topk", int(modelargs.global_bank_topk)),
        ("entity_bank_topk", int(modelargs.entity_bank_topk)),
        ("layout_bank_topk", int(modelargs.layout_bank_topk)),
        ("relation_bank_topk", int(modelargs.relation_bank_topk)),
        ("layout_grid_size", int(modelargs.layout_grid_size)),
        ("relation_max_pairs", int(modelargs.relation_max_pairs)),
    ]:
        actual = getattr(inner, name, None)
        if actual != expected:
            raise RuntimeError(f"运行时参数未同步: {name}={actual}, expected={expected}")

    logger.info(
        "[SANITY] four-bank runtime ready: g=%s e=%s l=%s r=%s seq=%s",
        inner.global_bank_topk,
        inner.entity_bank_topk,
        inner.layout_bank_topk,
        inner.relation_bank_topk,
        getattr(model.config, "image_seq_length", None),
    )


def load_model_processor(modelargs: ModelArguments):
    """
    加载模型和处理器
    """
    model_cls = CustomLlavaForConditionalGeneration
    processor_cls = CustomLlavaProcessor

    if modelargs.train_type == "use_lora":
        model_path = os.path.join(modelargs.model_name_or_path, "trained")
    else:
        model_path = modelargs.model_name_or_path

    logger.info("[ARGS] model_name_or_path=%s", modelargs.model_name_or_path)
    logger.info("[ARGS] resolved model_path=%s", model_path)
    logger.info("[ARGS] train_type=%s", modelargs.train_type)
    logger.info("[ARGS] semantic_bank_path=%s", modelargs.semantic_bank_path)
    logger.info(
        "[ARGS] semantic_cluster_num=%s global_bank_topk=%s entity_bank_topk=%s "
        "layout_bank_topk=%s relation_bank_topk=%s layout_grid_size=%s "
        "relation_near_threshold=%s relation_overlap_threshold=%s "
        "relation_direction_margin=%s relation_max_pairs=%s patch_drop=%s "
        "bank_retrieval_temperature=%s use_bank=%s",
        modelargs.semantic_cluster_num,
        modelargs.global_bank_topk,
        modelargs.entity_bank_topk,
        modelargs.layout_bank_topk,
        modelargs.relation_bank_topk,
        modelargs.layout_grid_size,
        modelargs.relation_near_threshold,
        modelargs.relation_overlap_threshold,
        modelargs.relation_direction_margin,
        modelargs.relation_max_pairs,
        modelargs.patch_drop,
        modelargs.bank_retrieval_temperature,
        modelargs.use_bank,
    )

    model = model_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    _apply_custom_config(model, modelargs)
    _reload_semantic_bank_if_needed(model)
    patch_custom_model_runtime(model)
    _sanity_check_four_bank_runtime(model, modelargs)

    processor = processor_cls.from_pretrained(
        modelargs.model_name_or_path,
        local_files_only=True,
    )

    if getattr(processor, "patch_size", None) is None:
        if hasattr(model, "config") and getattr(model.config, "vision_config", None) is not None:
            processor.patch_size = getattr(model.config.vision_config, "patch_size", None)

    if getattr(processor, "patch_size", None) is None:
        raise ValueError("processor.patch_size is None，请检查 processor/config.json 或手动指定 patch_size")

    logger.info("processor.patch_size=%s", processor.patch_size)

    _sync_processor_attrs(processor, modelargs)
    processor = patch_custom_processor_runtime(processor, modelargs)
    _sanity_check_processor_model_alignment(model, processor, modelargs)

    if modelargs.train_type == "use_lora":
        logger.warning("Loading model to LoRA mode")

        from peft import LoraConfig, get_peft_model

        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.05

        for p in model.multi_modal_projector.parameters():
            p.requires_grad = False

        layers = list(range(22, 28))
        projs = ["q_proj", "k_proj", "v_proj", "o_proj"]

        target_modules = [
            f"model.language_model.layers.{layer}.self_attn.{proj}"
            for layer in layers
            for proj in projs
        ]

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)

        hit = []
        for n, m in model.named_modules():
            if any(t in n for t in target_modules):
                hit.append((n, type(m).__name__))

        print(f"[LoRA] matched modules: {len(hit)}")
        for n, t in hit[:10]:
            print("  ", n, "->", t)

        if len(hit) == 0:
            raise ValueError("LoRA 没命中任何模块，请检查 target_modules 和 peft 版本。")

    elif modelargs.train_type == "none":
        logger.warning("使用全量参数进行训练")

    elif modelargs.train_type == "freeze_vision":
        logger.warning("冻结 vision_tower 网络层，其余参数训练")
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    elif modelargs.train_type == "tune_mm_mlp_adapter":
        print(">>> 已进入 projector 解冻分支 <<<")
        model.requires_grad_(False)
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = True
            print(f"  解冻: {tuple(p.shape)}")

    else:
        raise ValueError(f"未知 train_type: {modelargs.train_type}")

    print_trainable_parameters(model)
    return model, processor


def load_dataset_collator(processor, dataargs: DataArguments):
    """
    加载数据集和数据整理器
    """
    llava_dataset = LlavaDataset(dataargs.data_path)
    logger.info("Loaded dataset from %s", dataargs.data_path)

    data_collator = TrainLLavaModelCollator(
        processor=processor,
        IGNORE_INDEX=-100,
    )
    return llava_dataset, data_collator


def train():
    """
    训练主函数
    """
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    logger.info("========== Parsed Arguments ==========")
    logger.info("model_args=%s", model_args)
    logger.info("data_args=%s", data_args)
    logger.info("output_dir=%s", training_args.output_dir)
    logger.info("=====================================")

    model, processor = load_model_processor(model_args)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        try:
            model.enable_input_require_grads()
            print("已开启 enable_input_require_grads()")
        except AttributeError:
            if hasattr(model, "language_model") and hasattr(model.language_model, "enable_input_require_grads"):
                model.language_model.enable_input_require_grads()
                print("已开启 language_model.enable_input_require_grads()")
    else:
        raise ValueError("请务必开启 gradient_checkpointing 以节省显存！")

    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    callbacks = [LossLoggingCallback()]
    try:
        callbacks.append(ProjectorGradMonitor())
    except Exception as e:
        logger.warning("ProjectorGradMonitor 加载失败，已跳过: %s", repr(e))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info("开始训练...")
    trainer.train()
    logger.info("训练完成")

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    logger.info("模型已保存到 %s", training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="pretrain.log",
        filemode="a",
    )
    train()
