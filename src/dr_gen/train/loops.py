from datetime import timedelta
import time

import torch
import torch.nn as nn

from dr_util.metrics import BATCH_KEY

import dr_gen.train.evaluate as eu
import dr_gen.train.model as mu


def log_metrics(md, group_name, **kwargs):
    assert md is not None, "There should be a metrics obj"

    loss = kwargs.get("loss", None)
    output = kwargs.get("output", None)
    target = kwargs.get("target", None)

    if output is not None:
        md.log_data((BATCH_KEY, output.shape[0]), group_name)

    if not (output is None or target is None):
        acc1, acc5 = eu.accuracy(output, target, topk=(1, 5))
        md.log_data(("acc1", acc1.item()), group_name)
        md.log_data(("acc5", acc5.item()), group_name)

    if loss is not None:
        md.log_data(("loss", loss.item()), group_name)


def train_epoch(cfg, epoch, model, dataloader, criterion, optimizer, md=None):
    model.train()
    for i, (image, target) in enumerate(dataloader):
        # if i % 10 == 0:
        #    md.log(f">> Sample: {i * image.shape[0]} / {len(dataloader.dataset)}")
        image, target = image.to(cfg.device), target.to(cfg.device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if cfg.optim.get("clip_grad_norm", None) is not None:
            nn.utils.clip_grad_norm_(
                model.paramteres(),
                cfg.optim.clip_grad_norm,
            )
        optimizer.step()
        log_metrics(
            md,
            "train",
            loss=loss,
            output=output,
            target=target,
        )


def eval_model(cfg, model, dataloader, criterion, name="val", md=None):
    model.eval()
    with torch.inference_mode():
        for image, target in dataloader:
            image = image.to(cfg.device, non_blocking=True)
            target = target.to(cfg.device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            log_metrics(
                md,
                name,
                loss=loss,
                output=output,
                target=target,
            )


def train_loop(cfg, train_dl, val_dl=None, eval_dl=None, md=None):
    assert md is not None  # Temporarily
    model, optim, lr_sched = mu.get_model_optim_lrsched(
        cfg,
        cfg.data.num_classes,
        md=md,
    )
    criterion = mu.get_criterion(cfg)
    mu.checkpoint_model(cfg, model, "init_model", md=md)

    start_time = time.time()
    for epoch in range(cfg.epochs):
        md.log(f">> Start Epoch: {epoch}")

        # Train
        train_epoch(cfg, epoch, model, train_dl, criterion, optim, md=md)
        if lr_sched is not None:
            lr_sched.step()
        md.agg_log("train")

        # Val
        if val_dl is not None:
            eval_model(cfg, model, val_dl, criterion, "val", md=md)
            md.agg_log("val")

        # Eval
        if eval_dl is not None:
            eval_model(cfg, model, eval_dl, criterion, "eval", md=md)
            md.agg_log("eval")

        # mu.checkpoint_model(cfg, model, f"epoch_{epoch}", md=md)
        md.clear_data()
        md.log("")

    total_time = time.time() - start_time
    total_ts = str(timedelta(seconds=int(total_time)))
    md.log(f"Training time {total_ts}")
