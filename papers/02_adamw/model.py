import math
from typing import Callable, Iterable, Optional, Tuple, Union
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Custom implementation of Adam with Decoupled Weight Decay (AdamW).

    Args:
        params       (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr              (float): Learning rate.
        betas   (Tuple[float]): Coefficients (beta1, beta2) used for computing running averages of gradient and its square.
        eps             (float): Term added to the denominator to improve numerical stability.
        weight_decay    (float): Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params: Iterable[Union[torch.nn.Parameter, dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model and returns the loss.

        Returns:
            float: Optional loss value evaluated by closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ################################################################
        # TODO: Implement the AdamW optimization step
        # 1. Iterate over param_groups
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 2. Iterate over params in each group
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                # 3. Initialize state ('step', 'exp_avg', 'exp_avg_sq') if not present
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                mt = state["exp_avg"]
                vt = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]
                
                # 4. Apply decoupled weight decay directly to p
                if weight_decay != 0 :
                    p.mul_(1 - lr * weight_decay)

                # 5. Update exp_avg and exp_avg_sq with beta1 and beta2
                mt.lerp_(grad, 1 - beta1)
                vt.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # 6. Compute bias corrections (bias_correction1, bias_correction2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # 7. Update p using step_size * (exp_avg / denom)
                step_size = lr / bias_correction1
                denom = vt.sqrt() / math.sqrt(bias_correction2) + eps
                p -= step_size * (mt / denom)
        ################################################################

        return loss
