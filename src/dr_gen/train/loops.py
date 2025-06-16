import time
from datetime import timedelta
from typing import Any

import torch
import torch.nn as nn
from dr_util.metrics import BATCH_KEY, Metrics
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

import dr_gen.train.evaluate as eu
import dr_gen.train.model as mu


def log_metrics(md: Metrics | None, group_name: str, **kwargs: Any) -> None:  # noqa: ANN401
    assert md is not None, "There should be a metrics obj"

    loss = kwargs.get("loss")
    output = kwargs.get("output")
    target = kwargs.get("target")
    mean_grad_norm = kwargs.get("mean_grad_norm")
    lr = kwargs.get("lr")

    if output is not None:
        md.log_data((BATCH_KEY, output.shape[0]), group_name)

    if not (output is None or target is None):
        acc1, acc5 = eu.accuracy(output, target, topk=(1, 5))
        md.log_data(("acc1", acc1.item()), group_name)
        md.log_data(("acc5", acc5.item()), group_name)

    if loss is not None:
        md.log_data(("loss", loss.item()), group_name)

    if mean_grad_norm is not None:
        md.log_data(("grad_norm", mean_grad_norm), group_name)

    if lr is not None:
        md.log_data(("lr", lr), group_name)


def train_epoch(cfg, epoch, model, dataloader, criterion, optimizer, md=None):  # noqa: ARG001
    model.train()
    for _i, (image, target) in enumerate(dataloader):
        # if i % 10 == 0:
        #    md.log(f">> Sample: {i * image.shape[0]} / {len(dataloader.dataset)}")  # noqa: ERA001,E501
        image, target = image.to(cfg.device), target.to(cfg.device)  # noqa: PLW2901
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if cfg.optim.get("clip_grad_norm", None) is not None:
            nn.utils.clip_grad_norm_(
                model.paramteres(),
                cfg.optim.clip_grad_norm,
            )
        mean_grad_norm = None
        with torch.no_grad():
            total_norm = 0
            num_parameters = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)  # L2 norm
                    total_norm += param_norm.item() ** 2  # Sum of squares
                    num_parameters += 1

            if num_parameters > 0:
                mean_grad_norm = (
                    total_norm / num_parameters
                ) ** 0.5  # Root of mean of squares

        optimizer.step()
        log_metrics(
            md,
            "train",
            loss=loss,
            output=output,
            target=target,
            grad_norm=mean_grad_norm,
            lr=optimizer.param_groups[0]["lr"],
        )


def eval_model(cfg, model, dataloader, criterion, name="val", md=None):
    model.eval()
    with torch.inference_mode():
        for image, target in dataloader:
            image = image.to(cfg.device, non_blocking=True)  # noqa: PLW2901
            target = target.to(cfg.device, non_blocking=True)  # noqa: PLW2901
            output = model(image)
            loss = criterion(output, target)
            log_metrics(
                md,
                name,
                loss=loss,
                output=output,
                target=target,
            )


def train_loop(
    cfg: Any,  # noqa: ANN401
    train_dl: DataLoader[Any],
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    lr_sched: LRScheduler | None,
    val_dl: DataLoader[Any] | None = None,
    eval_dl: DataLoader[Any] | None = None,
    md: Metrics | None = None,
) -> None:
    assert md is not None  # Temporarily
    criterion = mu.get_criterion(cfg)
    mu.checkpoint_model(cfg, model, "init_model", md=md)
    lr_sched.step(epoch=0)

    start_time = time.time()
    for epoch in range(cfg.epochs):
        md.log(f">> Start Epoch: {epoch}")

        # Train
        train_epoch(cfg, epoch, model, train_dl, criterion, optim, md=md)
        if lr_sched is not None:
            lr_sched.step(epoch=epoch + 1)
        md.agg_log("train")

        # Val
        if val_dl is not None:
            eval_model(cfg, model, val_dl, criterion, "val", md=md)
            md.agg_log("val")

        # Eval
        if eval_dl is not None:
            eval_model(cfg, model, eval_dl, criterion, "eval", md=md)
            md.agg_log("eval")

        # TODO: Hacky
        if epoch + 1 in [5, 10, 25]:
            mu.checkpoint_model(cfg, model, f"epoch_{epoch}", md=md)
        md.clear_data()
        md.log("")
    mu.checkpoint_model(cfg, model, f"final_model_epoch{cfg.epochs - 1}", md=md)

    total_time = time.time() - start_time
    total_ts = str(timedelta(seconds=int(total_time)))
    md.log(f"Training time {total_ts}")
