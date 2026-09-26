import logging
import os
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
from data import LlavaDataset, TrainLLavaModelCollator
from util import print_trainable_parameters

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


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


def _apply_custom_config(model, modelargs: ModelArguments) -> None:
    """
    把 CLI 参数强制写进 model.config，避免 from_pretrained 读到旧 config.json。
    """
    if not hasattr(model, "config") or model.config is None:
        raise ValueError("model.config 不存在，无法注入 semantic bank 配置")

    cfg = model.config

    old_path = getattr(cfg, "semantic_bank_path", None)
    logger.info("[CONFIG BEFORE] semantic_bank_path=%s", old_path)

    cfg.semantic_bank_path = modelargs.semantic_bank_path
    cfg.num_cluster_tokens = int(modelargs.semantic_cluster_num)
    cfg.global_bank_topk = int(modelargs.global_bank_topk)
    cfg.entity_bank_topk = int(modelargs.entity_bank_topk)
    cfg.bank_retrieval_temperature = float(modelargs.bank_retrieval_temperature)
    cfg.use_bank = bool(modelargs.use_bank)
    cfg.use_cache = False

    logger.info("[CONFIG AFTER] semantic_bank_path=%s", cfg.semantic_bank_path)
    logger.info(
        "[CONFIG AFTER] num_cluster_tokens=%s global_bank_topk=%s entity_bank_topk=%s temperature=%s use_bank=%s",
        cfg.num_cluster_tokens,
        cfg.global_bank_topk,
        cfg.entity_bank_topk,
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
    关键修复：
    from_pretrained 初始化时，CustomLlavaModel.__init__ 可能先按旧 config 路径加载了错误 bank。
    这里在 config 被我们覆盖后，强制按新路径重新加载一次。
    """
    inner_model = getattr(model, "model", None)
    if inner_model is None:
        raise ValueError("model.model 不存在，无法重新加载 semantic bank")

    if not hasattr(inner_model, "_load_semantic_bank"):
        raise ValueError("CustomLlavaModel 缺少 _load_semantic_bank 方法")

    use_bank = bool(getattr(model.config, "use_bank", True))
    bank_path = getattr(model.config, "semantic_bank_path", None)

    # 同步 inner_model 上的运行时属性
    inner_model.use_bank = use_bank
    inner_model.semantic_bank_path = bank_path
    inner_model.num_cluster_tokens = int(getattr(model.config, "num_cluster_tokens", 10))
    inner_model.global_bank_topk = int(getattr(model.config, "global_bank_topk", 4))
    inner_model.entity_bank_topk = int(getattr(model.config, "entity_bank_topk", 6))
    inner_model.bank_retrieval_temperature = float(
        getattr(model.config, "bank_retrieval_temperature", 1.0)
    )

    if use_bank:
        if bank_path is None:
            raise ValueError("重新加载 semantic bank 时发现 semantic_bank_path 为 None")
        if not os.path.exists(bank_path):
            raise FileNotFoundError(f"重新加载 semantic bank 时路径不存在: {bank_path}")

        logger.info("[RELOAD BANK] path=%s", bank_path)
        inner_model._load_semantic_bank(bank_path)

        # 运行时检查
        gb = getattr(inner_model, "_semantic_global_bank", None)
        eb = getattr(inner_model, "_semantic_entity_bank", None)

        logger.info(
            "[RELOAD BANK] global shape=%s",
            None if gb is None else tuple(gb.shape),
        )
        logger.info(
            "[RELOAD BANK] entity shape=%s",
            None if eb is None else tuple(eb.shape),
        )
    else:
        logger.info("[RELOAD BANK] use_bank=False, skip loading semantic bank")
        inner_model._semantic_global_bank = None
        inner_model._semantic_entity_bank = None


def _sync_processor_attrs(processor, modelargs: ModelArguments) -> None:
    """
    processor 里 <image> token 展开数必须和模型一致。
    """
    processor.num_cluster_tokens = int(modelargs.semantic_cluster_num)
    processor.global_bank_topk = int(modelargs.global_bank_topk)
    processor.entity_bank_topk = int(modelargs.entity_bank_topk)

    logger.info(
        "[PROCESSOR] num_cluster_tokens=%s global_bank_topk=%s entity_bank_topk=%s",
        processor.num_cluster_tokens,
        processor.global_bank_topk,
        processor.entity_bank_topk,
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
        "[ARGS] semantic_cluster_num=%s global_bank_topk=%s entity_bank_topk=%s bank_retrieval_temperature=%s use_bank=%s",
        modelargs.semantic_cluster_num,
        modelargs.global_bank_topk,
        modelargs.entity_bank_topk,
        modelargs.bank_retrieval_temperature,
        modelargs.use_bank,
    )

    # 先加载模型
    model = model_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    # 强制把 CLI 参数写进 config
    _apply_custom_config(model, modelargs)

    # 关键：强制按新路径重载 bank
    _reload_semantic_bank_if_needed(model)

    # 加载 processor
    processor = processor_cls.from_pretrained(
        modelargs.model_name_or_path,
        local_files_only=True,
    )

    # patch_size 兜底
    if getattr(processor, "patch_size", None) is None:
        if hasattr(model, "config") and getattr(model.config, "vision_config", None) is not None:
            processor.patch_size = getattr(model.config.vision_config, "patch_size", None)

    if getattr(processor, "patch_size", None) is None:
        raise ValueError("processor.patch_size is None，请检查 processor/config.json 或手动指定 patch_size")

    logger.info("processor.patch_size=%s", processor.patch_size)

    # 同步 processor 的 token 展开参数
    _sync_processor_attrs(processor, modelargs)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=[LossLoggingCallback()],
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