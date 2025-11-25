import torch
from typing import Iterable, Callable


class AdamSania(torch.optim.Optimizer):
    """A very small standalone Adam implementation.

    Mirrors the structure of `Muon` (custom optimizer class) but only performs
    the classic Adam update without any orthogonalization or auxiliary logic.

    Arguments:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-4)
        betas: Coefficients used for computing running averages of gradient
            and its square (default: (0.9, 0.999))
        eps: Term added to the denominator for numerical stability (default: 1e-8)
        weight_decay: L2 penalty (applied in-gradient, not decoupled) (default: 0.0)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0.0:
                    g = g.add(p.data, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Update first and second moments
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                denom = (exp_avg_sq.abs() / (bias_correction2)).add_(eps)
                print("denom:", denom)
                step_size = lr / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
