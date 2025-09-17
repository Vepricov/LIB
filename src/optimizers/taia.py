"""
Here is an original implementation of Muon.
Source: https://github.com/KellerJordan/modded-nanogpt
"""

import os

from numpy import dtype
import torch
import torch.distributed as dist
from typing import Callable


def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T

    return X.to(dtype=G.dtype, device=G.device)


class TAIA(torch.optim.Optimizer):
    """
    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        taia_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
        lmo=None,
        precondition_type="norm",
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
            lmo=lmo,
            precondition_type=precondition_type,
        )

        params = list(taia_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in taia_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]["use_taia"] = True
            else:
                self.state[p]["use_taia"] = False
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_taia"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

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
            ############################
            #           TAIA           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_taia"]]
            lr = group["lr"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(params):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None
                    state = self.state[p]
                    if "step    " not in state:
                        state["step"] = 0
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(g)
                    if "prec_L" not in state and group["precondition_type"] == "fisher":
                        # state["prec_L"] = shampoo_eps * torch.eye(g.size(0), device=g.device)
                        # state["prec_R"] = shampoo_eps * torch.eye(g.size(1), device=g.device)
                        state["prec_L"] = torch.zeros(
                            g.size(0), g.size(0), device=g.device
                        )
                        state["prec_R"] = torch.zeros(
                            g.size(1), g.size(1), device=g.device
                        )
                    if (
                        "exp_avg_sq" not in state
                        and group["precondition_type"] == "adam"
                    ):
                        state["exp_avg_sq"] = torch.zeros_like(g)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    # поменял вот здесь
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(momentum).add_(g, alpha=1.0)
                    g = exp_avg

                    state["step"] += 1
                    if group["precondition_type"] == "fisher":
                        H_L = state["prec_L"].clone()
                        H_R = state["prec_R"].clone()
                        # print(H_R)
                        state["prec_L"].add_(g @ g.T)
                        state["prec_R"].add_(g.T @ g)
                    elif group["precondition_type"] == "adam":
                        beta2 = group["adamw_betas"][1]
                        state["exp_avg_sq"].mul_(beta2).addcmul_(
                            g, g, value=1.0 - beta2
                        )
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        D = (
                            state["exp_avg_sq"]
                            .sqrt()
                            .add_(group["adamw_eps"])
                            .div_(bias_correction2**0.5)
                        )
                        g /= D.sqrt()

                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)

                    if group["precondition_type"] == "norm":
                        L = g.norm(dim=0, keepdim=True)
                        L = torch.where(L == 0, 1e-8, L)
                        g /= L
                    elif group["precondition_type"] == "fisher":
                        # L = torch.linalg.cholesky(H_L, upper=True)
                        # R = torch.linalg.cholesky(H_R, upper=False)
                        # L_inv = torch.linalg.inv(L).to(dtype=g.dtype, device=g.device)
                        # R_inv = torch.linalg.inv(R).to(dtype=g.dtype, device=g.device)
                        U_L, sigma_L, _ = torch.linalg.svd(H_L)
                        U_R, sigma_R, _ = torch.linalg.svd(H_R)
                        for i in range(sigma_L.size(0)):
                            sigma_L[i] = 1.0 / sigma_L[i] if sigma_L[i] > 1e-10 else 0
                        for i in range(sigma_R.size(0)):
                            sigma_R[i] = 1.0 / sigma_R[i] if sigma_R[i] > 1e-10 else 0
                        L_inv = U_L @ torch.diag(sigma_L**1 / 8)
                        R_inv = torch.diag(sigma_R**1 / 8) @ U_R.T
                        g = L_inv.T @ g @ R_inv.T

                    if group["lmo"] == "spectral":
                        g = zeropower_via_newtonschulz5(
                            g, steps=group["ns_steps"], eps=group["adamw_eps"]
                        )
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5

                    if group["precondition_type"] == "norm":
                        g /= L
                    elif group["precondition_type"] == "fisher":
                        g = L_inv @ g @ R_inv
                    elif group["precondition_type"] == "adam":
                        g /= D.sqrt()

                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in params:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )

                p.data.add_(g, alpha=-lr)

                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_taia"]]
            lr = (
                group["adamw_lr_ratio"] * group["lr"]
            )  # in order for lr schedule to work
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def separate_params(param_groups):
    param_groups_2d = []
    param_groups_non2d = []
    total_param_2d_count = 0
    total_param_non2d_count = 0

    # Check if param_groups is a list of dicts or list of params
    if (
        isinstance(param_groups, list) and isinstance(param_groups[0], dict)
    ) or isinstance(param_groups, dict):
        if isinstance(param_groups, dict):
            param_groups = [param_groups]
        # param_groups is a list of dicts
        for group in param_groups:
            (
                params_2d,
                params_non2d,
                param_2d_count,
                param_non2d_count,
            ) = separate_params(group["params"])
            param_group_2d = {"params": params_2d}
            param_group_non2d = {"params": params_non2d}
            # Copy the group dict and replace the 'params' key with the separated params
            for k in group.keys():
                if k != "params":
                    param_group_2d[k] = group[k]
                    param_group_non2d[k] = group[k]

            param_groups_2d.append(param_group_2d)
            param_groups_non2d.append(param_group_non2d)
            total_param_2d_count += param_2d_count
            total_param_non2d_count += param_non2d_count

        return (
            param_groups_2d,
            param_groups_non2d,
            total_param_2d_count,
            total_param_non2d_count,
        )

    elif isinstance(param_groups, list) and isinstance(param_groups[0], torch.Tensor):
        params_2d = []
        params_non2d = []
        param_group = param_groups
        # param_group is a list of param tensors
        for param in param_group:
            if param.ndim == 2:
                params_2d.append(param)
            else:
                params_non2d.append(param)
        return params_2d, params_non2d, len(params_2d), len(params_non2d)
    else:
        breakpoint()
