import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import os
import numpy as np
from sklearn.datasets import load_svmlight_file

import utils
from libsvm import models_libsvm


def generate_synthetic_classification_data(
    n_samples, input_dim, n_classes, noise_std=0.1, dtype=torch.float64, seed=42
):
    """
    Generate synthetic data for softmax classification problem.

    Args:
        n_samples: Number of data points N
        input_dim: Dimension of input vectors (n)
        n_classes: Number of classes (m)
        noise_std: Standard deviation of noise added to logits
        dtype: Data type for tensors
        seed: Random seed

    Returns:
        X: Input data matrix of shape (N, input_dim)
        y: Target class labels of shape (N,) - integers 0 to n_classes-1
        W_true: True weight matrix of shape (n_classes, input_dim)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate random input data X
    X = torch.randn(n_samples, input_dim, dtype=dtype)

    # Generate random true weight matrix W_true
    W_true = torch.randn(n_classes, input_dim, dtype=dtype)

    # Generate logits: logits = W_true @ X.T + noise
    logits_clean = torch.matmul(W_true, X.T).T  # (N, n_classes)

    # Add noise to logits
    noise = torch.randn_like(logits_clean) * noise_std
    logits = logits_clean + noise

    # Get class labels by taking argmax
    y = torch.argmax(logits, dim=1)  # (N,)

    return X, y, W_true


class SoftmaxCrossEntropyLoss(nn.Module):
    """
    Softmax cross-entropy loss for classification.
    """

    def __init__(self, reduction="mean"):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw logits of shape (batch_size, n_classes)
            targets: Class indices of shape (batch_size,)
        """
        # Use PyTorch's cross entropy which applies softmax + log + nll
        loss = nn.functional.cross_entropy(logits, targets, reduction="none")

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def libsvm_prepocess(args):
    g, seed_worker = utils.set_global_seed(args.seed)

    if args.dataset == "synthetic_classification":
        # Generate synthetic classification data
        n_samples = getattr(args, "n_samples", 100)
        input_dim = getattr(args, "input_dim", 10)
        n_classes = getattr(args, "n_classes", 5)
        noise_std = getattr(args, "noise_std", 0)

        X, y, W_true = generate_synthetic_classification_data(
            n_samples=n_samples,
            input_dim=input_dim,
            n_classes=n_classes,
            noise_std=noise_std,
            dtype=getattr(torch, args.dtype),
            seed=args.seed,
        )

    elif args.dataset == "mushrooms":
        if not os.path.exists(f"./{args.data_path}/mushrooms"):
            os.system(
                f"cd ./{args.data_path}\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms \n cd .."
            )
        X, y = load_svmlight_file(f"./{args.data_path}/mushrooms")
        y = y - 1
        X = X.toarray()
        X = torch.tensor(X)
        if args.n_samples is not None:
            X = X[: args.n_samples]
            y = y[: args.n_samples]
        y = torch.tensor(y, dtype=X.dtype)

    elif args.dataset == "binary":
        if not os.path.exists(f"./{args.data_path}/covtype.libsvm.binary.scale.bz2"):
            os.system(
                f"cd ./{args.data_path}\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2 \n cd .."
            )
        X, y = load_svmlight_file(f"./{args.data_path}/covtype.libsvm.binary.scale.bz2")
        y = y - 1
        X = X.toarray()
        X = torch.tensor(X)
        y = torch.tensor(y, dtype=X.dtype)

    # Apply transformations only for non-synthetic datasets
    if args.dataset != "synthetic_quadratic" and args.scale:
        A = np.diag(np.exp(np.random.uniform(0, args.scale_bound, X.shape[1])))
        X = X @ A
    if args.dataset != "synthetic_quadratic" and args.rotate:
        B = np.random.random([X.shape[1], X.shape[1]])
        A, _ = np.linalg.qr(B.T @ B)
        X = X @ A

    ds = TensorDataset(X, y)
    ds_train, ds_val, ds_test = torch.utils.data.random_split(
        ds, [0.7, 0.2, 0.1], generator=g
    )
    train_dataloader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    # Choose loss function based on dataset type
    if args.dataset == "synthetic_classification":
        loss_fn = SoftmaxCrossEntropyLoss(reduction="none")
    else:
        loss_fn = nn.CrossEntropyLoss(reduction="none")

    if args.model == "linear-classifier":
        model = models_libsvm.LinearClassifier(
            input_dim=X.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=len(np.unique(y)),
            dtype=X.dtype,
            bias=not args.no_bias,
            weight_init=args.weight_init,
            X=X,
        )
    elif args.model == "softmax-linear":
        if args.dataset != "synthetic_classification":
            raise ValueError(
                f"Model {args.model} is only compatible with synthetic_classification dataset"
            )
        model = models_libsvm.SoftmaxLinearModel(
            input_dim=X.shape[1],
            output_dim=n_classes,
            dtype=X.dtype,
            bias=not args.no_bias,
            X=X,
        )
    else:
        raise ValueError(f"Wrong model name: {args.model} for dataset {args.dataset}")

    return train_dataloader, val_dataloader, test_dataloader, loss_fn, model
