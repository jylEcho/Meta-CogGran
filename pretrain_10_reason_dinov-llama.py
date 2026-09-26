import logging
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
)
from callback import ProjectorGradMonitor 
from custom_llava_10 import CustomLlavaForConditionalGeneration, CustomLlavaProcessor
# 导入自定义的数据处理模块（需确保路径正确）
from data import LlavaDataset, TrainLLavaModelCollator
from util import print_trainable_parameters
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)



class LossLoggingCallback(transformers.TrainerCallback):
    """在 Trainer 的 on_log 中把包含 'loss' 的日志写入 logging（pretrain_clip.log）"""
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
    model_name_or_path: Optional[str] = field(default="none", metadata={"help": "预训练模型的路径或名称"})
    train_type: Optional[str] = field(
        default="none",
        metadata={
            "help": """
            1. use_lora: 使用lora训练,
            2. none: 全量参数训练;
            3. freeze_vision: 只冻结vision_tower进行训练
            4.tune_mm_mlp_adapter，只训练投影器
            """
        },
    )


@dataclass
class DataArguments:
    """数据参数配置类"""
    data_path: str = field(
        default=None, metadata={"help": "训练数据的路径"}
    )


def load_model_processor(modelargs: ModelArguments):
    """
    加载模型和处理器
    :param modelargs: 模型参数配置
    :return: 模型和处理器对象
    """
    """
    根据 use_custom_model 决定用哪一套类加载
    """

    model_cls = CustomLlavaForConditionalGeneration

    # 2. 选处理器类
    processor_cls = CustomLlavaProcessor

    # 加载LLaVA模型
    if modelargs.train_type == "use_lora":
        model_path = modelargs.model_name_or_path + '/trained'
    else:
        model_path = modelargs.model_name_or_path
    model = model_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,  # 使用BF16精度
        low_cpu_mem_usage=True,  # 优化CPU内存使用
        local_files_only=True   # 仅使用本地文件
    )
    print("modelargs.model_name_or_path:", modelargs.model_name_or_path)
    # for name, _ in model.named_modules():
    #     if "layers." in name:
    #         print(name)
    # 训练时禁用缓存，避免警告与潜在不一致
    if hasattr(model, "config"):
        model.config.use_cache = False
    # 加载模型对应的处理器（包含分词器和图像处理器）
    processor = processor_cls.from_pretrained(modelargs.model_name_or_path, local_files_only=True)
##########################################################################################################
    # 新增：兜底，防止 processor 里没有 pad_token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 再同步到 model config
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if hasattr(model, "language_model") and getattr(model.language_model.config, "pad_token_id", None) is None:
        model.language_model.config.pad_token_id = processor.tokenizer.pad_token_id
##########################################################################################################

    print(f'trainer_type: {modelargs.train_type}')
    if modelargs.train_type == "use_lora":
        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model

        # LoRA参数配置
        LORA_R = 16  # 秩参数，控制低秩近似的维度
        LORA_ALPHA = 32  # 缩放因子，用于调整 LoRA 模块中权重更新的幅度
        LORA_DROPOUT = 0.05  # dropout率
        # TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 需要应用LoRA的模块名称
        # 冻结 projector
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = False
       
        # 1. 只保留 22-27 层的四个 attention proj
        layers = list(range(22, 28))          # 22 23 24 25 26 27
        projs  = ["q_proj", "k_proj", "v_proj", "o_proj"]

        TARGET_MODULES = [
            f"model.language_model.layers.{layer}.self_attn.{proj}"
            for layer in layers
            for proj in projs
        ]

        # TARGET_MODULES = [
        #     "self_attn.q_proj",
        #     "self_attn.k_proj",
        #     "self_attn.v_proj",
        #     "self_attn.o_proj",
        # ]

        # 初始化LoRA配置
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",  # LoRA 只作用于权重矩阵，而不影响偏置项
            task_type="CAUSAL_LM",  # 表示任务类型是因果语言模型（Causal Language Modeling），即根据前面的文本预测下一个单词
            # modules_to_save=["multi_modal_projector"], 
        )

        # 应用LoRA到模型
        model = get_peft_model(model, config)

        # 强制校验是否命中
        hit = []
        for n, m in model.named_modules():
            if any(t in n for t in TARGET_MODULES):
                hit.append((n, type(m).__name__))
        print(f"[LoRA] matched modules: {len(hit)}")
        for n, t in hit[:10]:
            print("  ", n, "->", t)
        if len(hit) > 0:
            print(f"[LoRA] total matched modules: {len(hit)}")
        else:
            raise ValueError("LoRA 没命中任何模块，请检查 target_modules 和 peft 版本。")

    elif modelargs.train_type == "none":
        """全量参数训练"""
        logging.warning("使用全量参数进行训练")

        pass
    elif modelargs.train_type == "freeze_vision":
        """冻结视觉塔的所有参数"""
        logging.warning("冻结vision_tower网络层，剩下的网络权重进行训练")
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    elif modelargs.train_type=="tune_mm_mlp_adapter":
        print(">>> 已进入 projector 解冻分支 <<<")
        model.requires_grad_(False)
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = True
            print(f"  解冻: {p.shape}")
    # 打印可训练参数信息
    print_trainable_parameters(model)

    return model, processor


def load_dataset_collator(processor, dataargs: DataArguments):
    """
    加载数据集和数据整理器
    :param processor: 模型处理器
    :param dataargs: 数据参数配置
    :return: 数据集和数据整理器对象
    """
    llava_dataset = LlavaDataset(
        dataargs.data_path  # xxxxx/LLaVA-CC3M-Pretrain-595K
    )

    logger.info(f"Loaded dataset from {dataargs.data_path}")

    # 初始化数据整理器
    data_collator = TrainLLavaModelCollator(
        processor=processor,
        IGNORE_INDEX=-100  # 损失计算时忽略的索引值
    )

    return llava_dataset, data_collator


def train():
    """训练模型的主函数"""
    # 初始化参数解析器
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    # 解析命令行参数
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()  

    # 防止 Trainer 把 batch 里用不到的键（比如 pixel_values）裁掉
    training_args.remove_unused_columns = False
    # 加载模型和处理器
    model, processor = load_model_processor(model_args)
    # 保证和 LoRA + gradient checkpointing 兼容
    if training_args.gradient_checkpointing:
        # 双保险：开启梯度检查点
        model.gradient_checkpointing_enable()
        # 关键：让输入 hidden states 需要梯度，否则被当成 no_grad 执行，导致 loss 无 grad_fn
        try:
            model.enable_input_require_grads()
            print("已开启 enable_input_require_grads()")
        except AttributeError:
            # Llava 顶层可能没有该方法，转到底层 language_model 尝试
            if hasattr(model, "language_model") and hasattr(model.language_model, "enable_input_require_grads"):
                model.language_model.enable_input_require_grads()
                print("已开启 language_model.enable_input_require_grads()")
    else:
        raise ValueError("请务必开启 gradient_checkpointing 以节省显存！")
    # 加载数据集和数据整理器
    train_dataset, data_collator = load_dataset_collator(processor, data_args)
    # print(train_dataset[0])          # 看第一条样本结构
    trainer = Trainer(
        model=model,
        args=training_args,  # 训练参数
        train_dataset=train_dataset,
        eval_dataset=None,   # 暂时不使用验证集
        data_collator=data_collator,  # 传入数据整理器
        callbacks=[LossLoggingCallback()],
    )
    # ========== 打印可训练参数 ==========
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(f"{n:<60} {p.numel():>12,}")
    # ===================================
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    logger.info("训练完成")

     # 保存训练状态和模型
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    logger.info(f"模型已保存到 {training_args.output_dir}")


if __name__ == "__main__":
    # 配置日志格式
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="pretrain.log",      # 新增
        filemode="a"                  # 追加
    )
    # 执行模型训练
    train()


