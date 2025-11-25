"""AdaMuon optimizer implementation (replaces Muon placeholder).

Algorithm (per paper draft provided):
M_t = beta * M_{t-1} + G_t
O_t = NewtonSchulz(Sign(M_t))
V_t = beta * V_{t-1} + (1 - beta) * (O_t ⊙ O_t)
O_hat = O_t / (sqrt(V_t) + eps)
γ_t = 0.2 * sqrt(m*n) / ||O_hat||_F
W_{t+1} = W_t - lr * (γ_t * O_hat + weight_decay * W_t)

We only apply AdaMuon to 2D parameters; others fall back to internal AdamW.
Changes kept minimal relative to original Muon structure for integration simplicity.
"""

import os

import torch
import torch.distributed as dist


@torch.compile
def zeropower_via_newtonschulz5(G, steps=6, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16)
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class AdaMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=3e-4,
        momentum=0.95,
        ns_steps=6,
        weight_decay=0.0,
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0.0,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr if lr != 0 else 1.0,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
            eps=eps,
        )
        param_list = list(params)
        super().__init__(param_list, defaults)

        for p in param_list:
            if p.ndim == 2 and p.size(0) < 10000:
                self.state[p]["use_adamuon"] = True
            else:
                self.state[p]["use_adamuon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.rank = int(os.environ.get("RANK", 0))
        else:
            self.world_size = 1
            self.rank = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            adamuon_params = [p for p in group["params"] if self.state[p]["use_adamuon"]]

            total_params = sum(p.numel() for p in adamuon_params)
            updates_flat = torch.zeros(total_params, device=adamuon_params[0].device if adamuon_params else "cpu", dtype=torch.bfloat16)
            curr_idx = 0

            for i, p in enumerate(adamuon_params):
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        curr_idx += p.numel()
                        continue
                    g_view = g.view(g.size(0), -1) if g.ndim > 2 else g
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g_view)
                    if "rms_buffer" not in state:
                        state["rms_buffer"] = torch.zeros_like(g_view)
                    m_buf = state["momentum_buffer"]
                    v_buf = state["rms_buffer"]
                    m_buf.mul_(beta).add_(g_view)
                    sign_m = m_buf.sign()
                    O = zeropower_via_newtonschulz5(sign_m, steps=ns_steps)
                    O *= max(1, O.size(0) / O.size(1)) ** 0.5
                    v_buf.mul_(beta).addcmul_(O, O, value=(1 - beta))
                    O_hat = O / (v_buf.sqrt() + eps)
                    frob = O_hat.norm() + eps
                    gamma = 0.2 * (p.shape[0] * p.shape[1]) ** 0.5 / frob
                    update = (gamma * O_hat).to(dtype=torch.bfloat16)
                    updates_flat[curr_idx : curr_idx + p.numel()] = update.view(-1)[: p.numel()]
                curr_idx += p.numel()

            if self.world_size > 1 and total_params > 0:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in adamuon_params:
                if p.grad is None:
                    curr_idx += p.numel()
                    continue
                raw_update = updates_flat[curr_idx : curr_idx + p.numel()].view_as(p.data).to(p.data.dtype)
                if weight_decay != 0:
                    raw_update = raw_update + weight_decay * p.data
                p.data.add_(raw_update, alpha=-lr)
                curr_idx += p.numel()

            # AdamW fallback for non-2D params
            adamw_params = [p for p in group["params"] if not self.state[p]["use_adamuon"]]
            if adamw_params:
                aw_lr = group["adamw_lr_ratio"] * group["lr"]
                beta1, beta2 = group["adamw_betas"]
                aw_eps = group["adamw_eps"]
                aw_wd = group["adamw_wd"]
                for p in adamw_params:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    state["step"] += 1
                    step = state["step"]
                    m1 = state["moment1"]
                    m2 = state["moment2"]
                    m1.lerp_(g, 1 - beta1)
                    m2.lerp_(g.square(), 1 - beta2)
                    g_hat = m1 / (aw_eps + m2.sqrt())
                    bc1 = 1 - beta1**step
                    bc2 = 1 - beta2**step
                    scale = bc1 / bc2**0.5
                    if aw_wd != 0:
                        p.data.mul_(1 - aw_lr * aw_wd)
                    p.data.add_(g_hat, alpha=-aw_lr / scale)
        return loss
