import torch
import torch.nn as nn
import tqdm


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, w_0=None, dtype=None):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, dtype=dtype)
        if w_0 is None:
            nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        y = self.linear(x)
        return y


def zero_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


def uniform_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def zero_uniform_init_weights(m):
    if isinstance(m, nn.Linear) and m.weight.shape[1] == 112:
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=10,
        output_dim=2,
        weight_init="uniform",
        dtype=None,
        bias=True,
        X=None
    ):
        super(LinearClassifier, self).__init__()
        self.X = X
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, dtype=dtype, bias=bias),
                # nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim, dtype=dtype, bias=bias),
                nn.Softmax(dim=1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim, dtype=dtype, bias=bias),
                nn.ReLU(),
                nn.Softmax(dim=1),
            )

        if weight_init == "zeroes":
            init_fn = zero_init_weights
        elif weight_init == "zero/uniform":
            init_fn = zero_uniform_init_weights
        else:
            init_fn = uniform_init_weights
        self.net.apply(init_fn)
        if weight_init == "bad_scaled":
            with torch.no_grad():
                val_w, val_b = 1e2, 1e2
                for layer in self.net:
                    if hasattr(layer, "weight"):
                        layer.weight.data *= val_w
                        val_w = val_w ** (-1)
                    if hasattr(layer, "bias"):
                        layer.weight.data *= val_b
                        val_b = val_b ** (-1)

    def forward(self, x):
        out = self.net(x)
        return out

    def compute_hessian(self, X_new=None):
        """
        Compute Hessian matrix for softmax cross-entropy loss
        H = 1/N * sum_i (x_i x_i^T ⊗ (diag(p_i) - p_i p_i^T))

        Args:
            X: Input data matrix of shape (N, input_dim)

        Returns:
            Hessian matrix of shape (output_dim * input_dim, output_dim * input_dim)
        """
        if X_new is not None:
            X = X_new
        elif self.X is not None:
            X = self.X
        else:
            raise ValueError("X is not provided")
        if self.hidden_dim > 0:
            raise ValueError("Hidden layers are not supported for Hessian computation")
        N = X.shape[0]
        device = self.net[0].weight.device
        dtype = X.dtype

        # Get probabilities for all samples
        with torch.no_grad():
            probs = self.forward(X.to(device))  # (N, output_dim)

        # Initialize Hessian
        hessian = torch.zeros(
            self.output_dim * self.input_dim,
            self.output_dim * self.input_dim,
            device=device,
            dtype=dtype,
        )

        # Sum over all samples
        for i in tqdm.tqdm(range(N), desc="Computing Hessian"):
            x_i = X[i]  # (input_dim,)
            p_i = probs[i]  # (output_dim,)

            # Compute x_i x_i^T
            xx_outer = torch.outer(x_i, x_i)  # (input_dim, input_dim)

            # Compute Sigma(p_i) = diag(p_i) - p_i p_i^T
            sigma_p = torch.diag(p_i) - torch.outer(
                p_i, p_i
            )  # (output_dim, output_dim)

            # Kronecker product
            hessian_i = torch.kron(xx_outer.to(device), sigma_p.to(device))  # (mn, mn)

            hessian += hessian_i

        # Average over samples
        hessian = hessian / N

        return hessian


class SoftmaxLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, dtype=None, bias=False, X=None):
        super(SoftmaxLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X = X

        # Initialize weights to zero for consistent starting point
        nn.init.zeros_(self.linear.weight)
        if bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # output shape: (batch_size, output_dim)
        logits = self.linear(x)
        return torch.softmax(logits, dim=1)
        # return self.linear(x)

    def forward_probs(self, x):
        # x shape: (batch_size, input_dim)
        # output shape: (batch_size, output_dim)
        logits = self.linear(x)
        return torch.softmax(logits, dim=1)

    def compute_hessian(self, X_new=None):
        """
        Compute Hessian matrix for softmax cross-entropy loss
        H = 1/N * sum_i (x_i x_i^T ⊗ (diag(p_i) - p_i p_i^T))

        Args:
            X: Input data matrix of shape (N, input_dim)

        Returns:
            Hessian matrix of shape (output_dim * input_dim, output_dim * input_dim)
        """
        if X_new is not None:
            X = X_new
        elif self.X is not None:
            X = self.X
        else:
            raise ValueError("X is not provided")
        N = X.shape[0]
        device = self.linear.weight.device
        dtype = X.dtype

        # Get probabilities for all samples
        with torch.no_grad():
            probs = self.forward_probs(X.to(device))  # (N, output_dim)

        # Initialize Hessian
        hessian = torch.zeros(
            self.output_dim * self.input_dim,
            self.output_dim * self.input_dim,
            device=device,
            dtype=dtype,
        )

        # Sum over all samples
        for i in range(N):
            x_i = X[i]  # (input_dim,)
            p_i = probs[i]  # (output_dim,)

            # Compute x_i x_i^T
            xx_outer = torch.outer(x_i, x_i).to(device)  # (input_dim, input_dim)

            # Compute Sigma(p_i) = diag(p_i) - p_i p_i^T
            sigma_p = torch.diag(p_i) - torch.outer(
                p_i, p_i
            ).to(device)  # (output_dim, output_dim)

            # Kronecker product
            hessian_i = torch.kron(xx_outer, sigma_p)  # (mn, mn)

            hessian += hessian_i

        # Average over samples
        hessian = hessian / N

        return hessian
