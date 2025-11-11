import math
import warnings
from typing import Callable
import numpy as np
import torch
import torch.optim as optim


class FatAdamW(optim.Optimizer):
    """
    Implements Adam algorithm with weight decay for Weight Lora adapter

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        num_adapters: int = 36,
        lora_extention: str = "dummy",
        fat_step: int = 10,
        max_fat_steps: int = 3,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.k = num_adapters
        self.chosen_layers = list(range(num_adapters))
        if lora_extention not in ["smart", "dummy", "restart"]:
            raise ValueError(f"Wrong lora_extention: {lora_extention}")
        self.lora_extention = lora_extention
        self.fat_step = fat_step
        self.max_fat_steps = max_fat_steps
        self.R, self.R_mom, self.R_mom_sq = None, None, None

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group["name"] != "weight_params":
                ############################ Adam Step #############################
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    if (i // 2) not in self.chosen_layers and group["name"] == "loraAB":
                        p.data = torch.zeros_like(p, requires_grad=False)
                        continue

                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    if (
                        group["name"] == "loraAB"
                        and state["step"] % self.fat_step == 0
                        and self.max_fat_steps >= 0
                    ):
                        A_or_B = np.argmin(p.data.shape)
                        # self.state[p] = {}
                        if A_or_B == 0:  # lora_A
                            if self.lora_extention == "dummy":
                                N = torch.rand_like(p.data, requires_grad=True)
                                p.data = torch.concat([p.data, N], dim=1)
                            elif self.lora_extention == "smart":
                                Q, self.R = torch.linalg.qr(p.data.T, mode="reduced")
                                N = torch.rand_like(p.data.T, requires_grad=True)
                                I = torch.eye(
                                    np.max(p.data.shape),
                                    requires_grad=True,
                                    device=p.data.device,
                                )
                                p.data = torch.concat([Q, (I - Q @ Q.T) @ N], dim=1).T.contiguous()
                            elif self.lora_extention == "restart":
                                N_1 = torch.rand_like(p.data, requires_grad=True)
                                N_2 = torch.rand_like(p.data, requires_grad=True)
                                p.data = torch.concat([N_1, N_2], dim=1)
                        else:  # lora_B
                            O = torch.zeros_like(p.data.T, requires_grad=True)
                            if self.lora_extention == "dummy":
                                p.data = torch.concat([p.data, O], dim=0)
                            elif self.lora_extention == "smart":
                                p.data = torch.concat([self.R @ p.data.T, O], dim=0).T.contiguous()
                            elif self.lora_extention == "restart":
                                p.data = torch.concat([O, O], dim=0)

                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        continue

                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = (
                            step_size * math.sqrt(bias_correction2) / bias_correction1
                        )

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
            else:
                ######################## StoIHT step for w #########################
                if self.max_fat_steps == 0:
                    self.max_fat_steps -= 1
                if self.max_fat_steps > 0:
                    w_vector = []
                    if "w_step" not in group.keys():
                        group["w_step"] = 0
                    group["w_step"] += 1
                    for i, p in enumerate(group["params"]):
                        if p.grad is None or i not in self.chosen_layers:
                            continue
                        p.add_(p.grad, alpha=-group["lr"])
                        if group["w_step"] % self.fat_step == 0:
                            w_vector.append(p.data.item())

                    if group["w_step"] % self.fat_step == 0:
                        self.k //= 2
                        self.max_fat_steps -= 1
                        new_chosen_layers = []
                        w_vector = torch.tensor(w_vector)
                        w_vector = group["proj"](w_vector, self.k)
                        j = 0
                        for i, p in enumerate(group["params"]):
                            if p.grad is None or i not in self.chosen_layers:
                                continue
                            if w_vector[j] > 0:
                                new_chosen_layers.append(i)
                            p.data = torch.tensor([w_vector[j]], device=p.device)
                            j += 1
                        self.chosen_layers = new_chosen_layers
                        print("$$$$$$$$", self.chosen_layers, "$$$$$$$$")
            ####################################################################
        return loss
