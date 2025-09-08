"""
Here is an original implementation of SOAP.
Source: https://github.com/nikhilvyas/SOAP
"""

from itertools import chain

import torch
import torch.nn as nn

from typing import Callable

# Parts of the code are modifications of Pytorch's AdamW optimizer
# Parts of the code are modifications of code from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py


def kron_appr(g):
    U, S, Vt = torch.linalg.svd(g)
    L = (U[:, 0] * S[0])[:, None] * U[:, 0][None, :]
    R = (Vt[0, :] * S[0])[:, None] * Vt[0, :][None, :]
    return L, R


def proj_split(L, R, g, beta=-1, init="kron"):
    if torch.norm(L) == 0 or torch.norm(R) == 0:
        if init == "eps":
            eps = 1e-3  # max(1.0, torch.norm(g))
            L = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
            L = L / torch.norm(L) * eps
            R = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
            R = R / torch.norm(R) * eps
        elif init == "kron":
            L, R = kron_appr(g)
        else:
            L += g @ g.T
            R += g.T @ g
    if beta != -1:
        L *= beta**0.5
        R *= beta**0.5
        if beta != 1:
            g *= (1 - beta) ** 0.5
    left_factor_norm = torch.linalg.norm(L)
    right_factor_norm = torch.linalg.norm(R)

    norm_product = left_factor_norm * right_factor_norm
    L /= left_factor_norm
    R /= right_factor_norm

    K1 = L * norm_product + g @ R @ g.T
    L1 = R * norm_product + g.T @ L @ g

    K_norm = torch.linalg.norm(K1)
    L_norm = torch.linalg.norm(L1)

    U1 = K1 / K_norm
    V1 = L1 / L_norm

    M = torch.sum(L * U1)
    N = torch.sum(R * V1)

    S1 = M * N * norm_product + torch.sum(U1 * (g @ V1 @ g.T))
    return U1 * (S1**0.5), V1 * (S1**0.5)


class MIKOLA_DROP_SOAP(torch.optim.Optimizer):
    """
    Implements MIKOLA_DROP_SOAP algorithm (TODO).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner.
        max_precond_dim (`int`, *optional*, defaults to 10000):
            Maximum dimension of the preconditioner.
            Set to 10000, so that we exclude most common vocab sizes while including layers.
        merge_dims (`bool`, *optional*, defaults to `False`):
            Whether or not to merge dimensions of the preconditioner.
        precondition_1d (`bool`, *optional*, defaults to `False`):
            Whether or not to precondition 1D gradients.
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer.
            Helps at large precondition_frequency (~100 in our experiments),
            but hurts performance at small precondition_frequency (~10 in our experiments).
        data_format (`str`, *optional*, defaults to `channels_first`):
            Data format of the input for convolutional layers.
            Should be "channels_last" for data_format of NHWC and "channels_first" for NCHW.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias correction in Adam.

    Example of usage:
        optim = MIKOLA_DROP_SOAP(lr = 3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,  #
        merge_dims: bool = False,  # Merge dimensions till the product of the dimensions is less than or equal to max_precond_dim.
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        init: str = "kron",
        report_fisher_diff: bool = False,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
            "init": init,
        }
        super().__init__(params, defaults)
        self._data_format = data_format
        self.report_fisher_diff = report_fisher_diff
        if report_fisher_diff:
            print(
                f"$$$$$$$$$$$$$ precondition_frequency = {precondition_frequency} $$$$$$$$$$$$$"
            )
            self.reported_diff = {
                "step": -1,
                "fisher_diff": None,
            }

    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
        """
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []

        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape

        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)

        new_grad = grad.reshape(new_shape)
        return new_grad

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    if len(grad.shape) == 2:
                        ##############################
                        ##### RANK-1 ADAM UPDATE #####
                        ##############################
                        state["l_t"] = (
                            torch.ones(
                                [grad.shape[0], 1], dtype=grad.dtype, device=grad.device
                            )
                            * group["eps"]
                        )
                        state["r_t"] = (
                            torch.ones(
                                [grad.shape[1], 1], dtype=grad.dtype, device=grad.device
                            )
                            * group["eps"]
                        )
                if "Q" not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(
                            group["shampoo_beta"]
                            if group["shampoo_beta"] >= 0
                            else group["betas"][1]
                        ),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                self.update_preconditioner(
                    grad,
                    state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                    init=group["init"],
                )
                # continue  # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']

                grad_projected = self.project(
                    grad,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                if len(grad.shape) == 2:
                    ##############################
                    ##### RANK-1 ADAM UPDATE #####
                    ##############################
                    # exp_avg_sq += grad_projected.square()
                    # TODO: revrite in-place
                    l_prev = state["l_t"].clone()
                    r_prev = state["r_t"].clone()
                    state["l_t"] = (
                        beta2 * state["l_t"] @ state["r_t"].T @ state["r_t"]
                        + (1 - beta2) * grad_projected.square() @ state["r_t"]
                        # state["l_t"] @ state["r_t"].T @ state["r_t"]
                        # + grad_projected.square() @ state["r_t"]
                    )
                    # state["l_t"] = exp_avg_sq @ state["r_t"]
                    state["l_t"] /= state["l_t"].norm()

                    state["r_t"] = (
                        beta2 * state["r_t"] @ state["l_t"].T @ state["l_t"]
                        + (1 - beta2) * grad_projected.square().T @ state["l_t"]
                        # state["r_t"] @ state["l_t"].T @ state["l_t"]
                        # + grad_projected.square().T @ state["l_t"]
                    )
                    # state["r_t"] = exp_avg_sq.T @ state["l_t"]
                    state["r_t"] /= state["r_t"].norm()

                    c = (
                        beta2 * (state["l_t"].T @ l_prev) * (state["r_t"].T @ r_prev)
                        + (1 - beta2)
                        * state["l_t"].T
                        @ grad_projected.square()
                        @ state["r_t"]
                        # (state["l_t"].T @ l_prev) * (state["r_t"].T @ r_prev)
                        # + state["l_t"].T @ grad_projected.square() @ state["r_t"]
                    )
                    # c = state["l_t"].T @ exp_avg_sq @ state["r_t"]
                    state["l_t"] *= torch.sqrt(c)
                    state["r_t"] *= torch.sqrt(c)

                    # denom = exp_avg_sq.sqrt().add_(group["eps"])
                    denom = (state["l_t"] @ state["r_t"].T).sqrt()
                else:
                    ###############################
                    ##### DEFAULT ADAM UPDATE #####
                    ###############################
                    exp_avg_sq.mul_(beta2).add_(
                        grad_projected.square(), alpha=(1.0 - beta2)
                    )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                if (
                    self.report_fisher_diff
                    and len(grad.shape) == 2
                    and max(grad.shape) < group["max_precond_dim"]
                ):
                    import wandb

                    if "H" not in state:
                        state["H"] = torch.zeros(
                            [len(grad.reshape(-1)), len(grad.reshape(-1))],
                            dtype=grad.dtype,
                            device=grad.device,
                        )
                    state["H"] += torch.outer(grad.reshape(-1), grad.reshape(-1))

                    if state["step"] > self.reported_diff["step"]:
                        if self.reported_diff["fisher_diff"] is not None:
                            wandb.log(
                                {
                                    "fisher_diff": self.reported_diff["fisher_diff"],
                                    "diag_diff": self.reported_diff["diag_diff"],
                                    "rank_1_adam_diff": self.reported_diff[
                                        "rank_1_adam_diff"
                                    ],
                                }
                            )
                        self.reported_diff["fisher_diff"] = 0
                        self.reported_diff["diag_diff"] = 0
                        self.reported_diff["rank_1_adam_diff"] = 0
                        self.reported_diff["step"] = state["step"]

                    Q_approx = torch.kron(state["Q"][0], state["Q"][1])
                    H_approx = (
                        Q_approx
                        @ torch.diag((state["l_t"] @ state["r_t"].T).reshape(-1))
                        @ Q_approx.T
                    )
                    H_rot = Q_approx.T @ state["H"] @ Q_approx
                    self.reported_diff["fisher_diff"] += (
                        torch.linalg.norm(state["H"] - H_approx) ** 2
                    )
                    self.reported_diff["diag_diff"] += (
                        torch.linalg.norm(torch.diag(H_rot) - H_rot) ** 2
                    )
                    self.reported_diff["rank_1_adam_diff"] += (
                        torch.linalg.norm(exp_avg_sq - state["l_t"] @ state["r_t"].T)
                        ** 2
                    )

                # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                exp_avg_projected = self.project(
                    exp_avg,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = step_size * (bias_correction2**0.5) / bias_correction1

                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                norm_grad = self.project_back(
                    exp_avg_projected / denom,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad**2) ** 0.5)

                p.add_(norm_grad, alpha=-step_size)

                # From AdamW code: Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # Update is done after the gradient step to avoid using current gradients in the projection.
                # self.update_preconditioner(
                #     grad,
                #     state,
                #     max_precond_dim=group["max_precond_dim"],
                #     merge_dims=group["merge_dims"],
                #     precondition_1d=group["precondition_1d"],
                #     init=group["init"],
                # )

        return loss

    def init_preconditioner(
        self,
        grad,
        state,
        precondition_frequency=10,
        shampoo_beta=0.95,
        max_precond_dim=10000,
        precondition_1d=False,
        merge_dims=False,
    ):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        state["GG"] = (
            []
        )  # Will hold all the preconditioner matrices (L and R in the paper).
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device)
                )
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)

            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))

        state["Q"] = None  # Will hold all the eigenbases of the preconditioner.
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [0]],
                )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)

        return grad

    def update_preconditioner(
        self,
        grad,
        state,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
        init="kron",
    ):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"]
                )
        elif grad.dim() == 2:
            # Add projector splitting procedure. Change init parameter to start with different initialization
            L, R = proj_split(
                state["GG"][0],
                state["GG"][1],
                grad,
                beta=state["shampoo_beta"],
                init=init,
            )
            state["GG"][0] = L
            state["GG"][1] = R
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            new_grad,
                            new_grad,
                            dims=[
                                [
                                    *chain(
                                        range(idx), range(idx + 1, len(new_grad.shape))
                                    )
                                ]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            grad,
                            grad,
                            # Contracts across all dimensions except for k.
                            dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
        # else:
        #     if merge_dims:
        #         new_grad = self.merge_dims(grad, max_precond_dim)
        #         for idx, sh in enumerate(new_grad.shape):
        #             if sh <= max_precond_dim:
        #                 outer_product = torch.tensordot(
        #                     new_grad,
        #                     new_grad,
        #                     dims=[
        #                         [
        #                             *chain(
        #                                 range(idx), range(idx + 1, len(new_grad.shape))
        #                             )
        #                         ]
        #                     ]
        #                     * 2,
        #                 )
        #                 state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
        #     else:
        #         for idx, sh in enumerate(grad.shape):
        #             if sh <= max_precond_dim:
        #                 outer_product = torch.tensordot(
        #                     grad,
        #                     grad,
        #                     # Contracts across all dimensions except for k.
        #                     dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]]
        #                     * 2,
        #                 )
        #                 state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(
                state, max_precond_dim, merge_dims, is_kron=len(grad.shape) == 2
            )

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient back to the original space.
        """
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [1]],
                )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def get_orthogonal_matrix(self, mat):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        """
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)

        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(
                    m + 1e-30 * torch.eye(m.shape[0], device=m.device)
                )
            except:
                _, Q = torch.linalg.eigh(
                    m.to(torch.float64) + 1e-10 * torch.eye(m.shape[0], device=m.device)
                )
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(
        self, state, max_precond_dim=10000, merge_dims=False, is_kron=False
    ):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration
        followed by torch.linalg.qr decomposition.
        """
        precond_list = state["GG"]
        orth_list = state["Q"]

        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())

        if is_kron:
            final = []
            for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
                if len(m) == 0:
                    final.append([])
                    continue
                est_eig = torch.diag(o.T @ m @ o)
                sort_idx = torch.argsort(est_eig, descending=True)
                if ind == 0:
                    state["l_t"] = state["l_t"].index_select(0, sort_idx)
                else:
                    state["r_t"] = state["r_t"].index_select(0, sort_idx)
                o = o[:, sort_idx]
                power_iter = m @ o
                Q, _ = torch.linalg.qr(power_iter)

                if not float_data:
                    Q = Q.to(original_device).type(original_type)
                final.append(Q)
        else:
            orig_shape = state["exp_avg_sq"].shape
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape
            if merge_dims:
                exp_avg_sq = self.merge_dims(state["exp_avg_sq"], max_precond_dim)
            else:
                exp_avg_sq = state["exp_avg_sq"]

            final = []
            for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
                if len(m) == 0:
                    final.append([])
                    continue
                est_eig = torch.diag(o.T @ m @ o)
                sort_idx = torch.argsort(est_eig, descending=True)
                exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
                o = o[:, sort_idx]
                power_iter = m @ o
                Q, _ = torch.linalg.qr(power_iter)

                if not float_data:
                    Q = Q.to(original_device).type(original_type)
                final.append(Q)

            if merge_dims:
                if self._data_format == "channels_last" and len(orig_shape) == 4:
                    exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
                else:
                    exp_avg_sq = exp_avg_sq.reshape(orig_shape)

            state["exp_avg_sq"] = exp_avg_sq

        return final
