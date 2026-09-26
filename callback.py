
import logging
from transformers import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)

class ProjectorGradMonitor(TrainerCallback):
    """
    监控 multi_modal_projector 的梯度：
    - on_substep_end: 每个累积子步结束后（loss.backward 之后、但未真正 optimizer.step）打印一次
    - on_pre_optimizer_step: 真正优化步之前（裁剪之后）打印一次
    - on_optimizer_step: 优化步之后（zero_grad 之前）再打印一次兜底
    注意：on_step_end 往往太晚，可能已经被 zero_grad(set_to_none=True) 置空。
    """
    def _log_projector_grads(self, tag: str, model, step: int):
        total_norm_sq = 0.0
        n_params = 0
        for n, p in model.named_parameters():
            if "multi_modal_projector" in n:
                logger.info(f"[{tag}] {n}: requires_grad={p.requires_grad}, grad_is_none={p.grad is None}, shape={tuple(p.shape)}")
            if "multi_modal_projector" in n and p.grad is not None:
                g = p.grad.detach()
                total_norm_sq += float(g.norm(2).item() ** 2)
                n_params += 1
        if n_params:
            logger.info(f"[{tag}] step={step} projector grad norm={(total_norm_sq ** 0.5):.6f}")
        else:
            logger.warning(f"[{tag}] step={step} projector 无梯度！")

    def on_substep_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 每个梯度累积子步结束（loss.backward调用后）触发
        model = kwargs["model"]
        # 仅打印第一步和每50步（注意：global_step 只在真正优化步时递增，子步可用 state.total_flos 或自增计数控制，简单起见不额外计数）
        if state.global_step == 0 or state.global_step % 50 == 0:
            logger.info(f"=== on_substep_end: global_step={state.global_step}, total_flos={state.total_flos} ===")
            self._log_projector_grads("substep_end", model, state.global_step)

    def on_pre_optimizer_step(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 梯度裁剪之后、optimizer.step 之前
        model = kwargs["model"]
        if state.global_step == 0 or state.global_step % 50 == 0:
            logger.info(f"=== on_pre_optimizer_step: global_step={state.global_step}, total_flos={state.total_flos} ===")
            self._log_projector_grads("pre_optimizer_step", model, state.global_step)

    def on_optimizer_step(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # optimizer.step 之后、zero_grad 之前（有些实现会延后 zero_grad，但不保证）
        model = kwargs["model"]
        if state.global_step == 0 or state.global_step % 50 == 0:
            logger.info(f"=== on_optimizer_step: global_step={state.global_step}, total_flos={state.total_flos} ===")
            self._log_projector_grads("optimizer_step", model, state.global_step)