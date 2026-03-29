import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import numpy as np
from sklearn.model_selection import train_test_split

import utils
import models_cv

_CIFAR10_RGB_MEANS = (0.491, 0.482, 0.447)
_CIFAR10_RGB_STDS = (0.247, 0.243, 0.262)

def cifar_prepocess(args):
    g, seed_worker = utils.set_global_seed(args.seed)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS)
    ])
    if args.not_augment:
        transform_train = transform_test
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS),
        ])

    ds = torchvision.datasets.CIFAR10(args.data_path, train=True, 
                                      transform=transform_train, download=True)
    train_idx, val_idx = train_test_split(np.arange(len(ds)), test_size=0.2, 
                                          stratify=ds.targets, random_state=args.seed)
    ds_train = Subset(ds, train_idx)
    ds_val = Subset(ds, val_idx)
    # ds_train = utils.TransformdDataset(ds_train, transform=transform_train)
    # ds_val = utils.TransformdDataset(ds_val, transform=transform_test)
    ds_test = torchvision.datasets.CIFAR10(args.data_path, train=False, 
                                           transform=transform_test)
    # ds_test = utils.TransformdDataset(ds_test, transform=transform_test)
    
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
        shuffle=False,
    )
    test_dataloader = DataLoader(
        ds_test, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=False,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    if args.model == "resnet20":
        model = models_cv.resnet20()
    else:
        raise ValueError(f"Wrong model name: {args.model}")

    return train_dataloader, val_dataloader, test_dataloader, loss_fn, model