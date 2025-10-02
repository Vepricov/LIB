"""
This module contains functions for training and evaluating PyTorch models, including support for logging and metric calculations.

In our library we use it for CV and libsvm problems. For fine-tuning we utilize HF trainer
"""

import torch
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm.auto import tqdm
import wandb

from collections import defaultdict


def train_step(
    model,
    optimizer,
    dataloader,
    loss_fn,
    device,
    args,
    tuning=False,
    epoch=0,
    verbose=False,
    train_config=None,
):
    model.train()
    total_loss = 0
    for t, (X, y) in enumerate(dataloader):
        X = X.to(device)
        if args.dataset in ["mushrooms", "binary"]:
            y = y.type(torch.LongTensor)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(X)

        # Проверка, что loss является скаляром
        # Для многих выходов нам нужно убедиться, что мы получаем скалярное значение
        if preds.ndim > 1 and y.ndim > 0:
            # Убедимся, что функция потерь возвращает скаляр
            loss = loss_fn(preds, y)
            # Если loss не скаляр, сделаем его скаляром
            if loss.dim() > 0:
                loss = loss.mean()
        else:
            loss = loss_fn(preds, y)

        # Теперь loss точно скаляр, можно вызывать backward()
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        if hasattr(args, "report_fisher_diff") and args.report_fisher_diff and not tuning:
            hess = (
                model.compute_hessian() if hasattr(model, "compute_hessian") else None
            )
        else:
            hess = None
        try:
            optimizer.step(hess=hess)
        except TypeError:
            optimizer.step()

        if args.wandb and not tuning:
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_step": train_config["train_step"],
                    "gradient_norm": total_norm,
                }
            )
        if (
            verbose
            and not tuning
            and round((t + 1) / len(dataloader), 1) - round(t / len(dataloader), 1) > 0
        ):
            line = f"[TRAIN {epoch+t/len(dataloader):.1f}/{args.n_epoches_train}] train_loss {loss.item():.4f} grad_norm {total_norm:.4f}"
            print(f"{line}")
        train_config["train_step"] += 1
        total_loss += loss.item()

    return total_loss / (t + 1)  # Исправлен делитель, чтобы избежать деления на ноль


@torch.no_grad()  # Исправлено: было @torch.no_grad, должно быть @torch.no_grad()
def eval_step(model, dataloader, loss_fn, device, args, validation=True):
    model.eval()
    total_loss = 0
    total_true = np.array([])
    total_pred = np.array([])
    for t, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.type(torch.LongTensor).to(device)
        preds = model(X).to(device)
        losses = loss_fn(preds, y)
        # Убедимся, что losses - скаляр для вычисления среднего
        if losses.dim() > 0:
            loss = losses.mean()
        else:
            loss = losses

        y_pred = preds.argmax(dim=-1)
        total_true = np.append(total_true, y.cpu().detach().numpy())
        total_pred = np.append(total_pred, y_pred.cpu().detach().numpy())

        total_loss += loss.item()

    prefix = "val" if validation else "test"
    if args.dataset in ["cifar10"]:
        average = "weighted" if len(np.unique(total_true)) > 2 else "binary"
        f1 = f1_score(total_true, total_pred, average=average)
        precision = precision_score(
            total_true, total_pred, zero_division=0.0, average=average
        )
        recall = recall_score(total_true, total_pred, average=average)
        results = {
            f"{prefix}_f1": f1,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
        }
    else:
        accuracy = accuracy_score(total_true, total_pred)
        results = {f"{prefix}_accuracy": accuracy}
    results[f"{prefix}_loss"] = total_loss / (t + 1)  # Защита от деления на ноль

    return results


def train(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    loss_fn,
    device,
    args,
    tuning=False,
    verbose=False,
):
    train_config = {}
    train_config["train_step"] = 0
    n_epoches = args.n_epoches_tune if tuning else args.n_epoches_train
    e_list = range(n_epoches) if tuning or args.verbose else tqdm(range(n_epoches))
    val_metrics = defaultdict(list)
    test_metrics = defaultdict(list)

    # Добавляем сохранение лучшей модели
    best_metric = float("-inf")  # или float('inf') для метрик, где меньше - лучше
    best_model_state = None

    for e in e_list:
        _ = train_step(
            model,
            optimizer,
            train_dataloader,
            loss_fn,
            device,
            args,
            tuning=tuning,
            verbose=verbose,
            train_config=train_config,
            epoch=e,
        )
        val_results = eval_step(
            model, val_dataloader, loss_fn, device, args, validation=True
        )

        test_results = eval_step(
            model, test_dataloader, loss_fn, device, args, validation=False
        )

        # Сохраняем метрики
        for key in val_results.keys():
            val_metrics[key].append(val_results[key])
        for key in test_results.keys():
            test_metrics[key].append(test_results[key])

        # Определяем метрику для отслеживания лучшей модели
        # Предполагается, что если есть accuracy, используем её, иначе берём первую доступную метрику
        current_metric = None
        if f"val_accuracy" in val_results:
            current_metric = val_results["val_accuracy"]
        elif f"val_f1" in val_results:
            current_metric = val_results["val_f1"]
        else:
            # Если нет конкретной метрики, берем первую, которая не потеря
            for key in val_results:
                if "loss" not in key:
                    current_metric = val_results[key]
                    break

        # Если нашли метрику и она лучше предыдущей, сохраняем модель
        if current_metric is not None and current_metric > best_metric:
            best_metric = current_metric
            # Сохраняем состояние модели
            best_model_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": e,
                "best_metric": best_metric,
            }

            # Опционально можем сохранять модель на диск
            if hasattr(args, "save_best_model") and args.save_best_model:
                save_path = os.path.join(args.results_path, "best_model.pth")
                torch.save(best_model_state, save_path)
                if verbose:
                    print(
                        f"Saved best model with {list(val_results.keys())[0].split('_')[1]} = {best_metric:.4f}"
                    )

        if args.wandb and not tuning:
            wandb_config = val_results | test_results
            wandb_config["epoch"] = e + 1
            wandb.log(wandb_config)

        if verbose and not tuning:
            line = f">>>[VAL  {e+1}/{args.n_epoches_train}] "
            for key in val_results.keys():
                line += f"{key.split('_')[1]} {val_results[key]:.4f} "
            print(f"{line}")
            line = f">>>[TEST {e+1}/{args.n_epoches_train}] "
            for key in test_results.keys():
                line += f"{key.split('_')[1]} {test_results[key]:.4f} "
            print(f"{line}")

    # Если мы сохраняли лучшую модель, можем загрузить её перед возвратом
    if (
        best_model_state is not None
        and hasattr(args, "return_best_model")
        and args.return_best_model
    ):
        model.load_state_dict(best_model_state["model_state_dict"])

    return model, val_metrics, test_metrics
