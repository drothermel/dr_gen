from datetime import datetime
import time

import torch
import torch.nn as nn

from dr_util.metrics import BATCH_KEY

import dr_gen.utils.evaluate as eu
import dr_gen.utils.model as mu

CRITERIONS = {"cross_entropy"}
OPTIMIZERS = {"sgd", "rmsprop", "adamw"}
LR_SCHEDULERS = {"steplr", "cosineannealinglr", "exponentiallr"}

# md.clear_data() optional group_name=None
# md.log_data(data, group_name, ns=1)


def log_metrics(cfg, group_name, **kwargs):
    assert cfg.md is not None, "There should be a metrics obj"

    loss = kwargs.get("loss", None)
    output = kwargs.get("output", None)
    target = kwargs.get("target", None)

    if output is not None:
        cfg.md.log_data((BATCH_KEY, output.shape[0]))

    if not (output is None or target is None):
        acc1, acc5 = eu.accuracy(output, target, topk=(1, 5))
        cfg.md.log_data(("acc1", acc1))
        cfg.md.log_data(("acc5", acc5))

    if loss is not None:
        cfg.md.log_data(("loss", loss))


def train_epoch(cfg, epoch, model, dataloader, criterion, optimizer):
    model.train()
    for i, (image, target) in enumerate(dataloader):
        image, target = image.to(cfg.device), target.to(cfg.device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if cfg.optim.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                model.paramteres(),
                cfg.optim.clip_grad_norm,
            )
        optimizer.step()
        log_metrics(
            cfg,
            "train",
            loss=loss,
            output=output,
            target=target,
        )


def eval_model(cfg, model, dataloader, criterion, metrics):
    model.eval()
    with torch.inference_mode():
        for image, target in dataloader:
            image = image.to(cfg.device, non_blocking=True)
            target = target.to(cfg.device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            log_metrics(
                cfg,
                "val",
                loss=loss,
                output=output,
                target=target,
            )
    return metrics


def train_loop(cfg, model, train_dl, val_dl=None):
    assert cfg.md is not None
    criterion = mu.get_criterion(cfg)
    optim, lr_sched = mu.get_optim(cfg, model)
    mu.checkpoint_model(cfg, model, "init_model")

    start_time = time.time()
    for epoch in range(cfg.epochs):
        cfg.md.log(f">> Start Epoch: {epoch}")

        # Train
        train_epoch(cfg, epoch, model, train_dl, criterion, optim)
        lr_sched.step()
        cfg.md.agg_log("train")

        # Val
        if val_dl is not None:
            eval_model(cfg, model, val_dl, criterion)
            cfg.md.agg_log("val")

        mu.checkpoint_model(cfg, model, f"epoch_{epoch}")
        cfg.md.clear_data()
        cfg.md.log("")

    total_time = time.time() - start_time
    total_ts = str(datetime.timedelta(seconds=int(total_time)))
    cfg.md.log(f"Training time {total_ts}")
