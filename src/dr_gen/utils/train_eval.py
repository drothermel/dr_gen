from collections import defaultdict
from enum import Enum
from datetime import datetime
import time

import torch
import torch.nn as nn

from dr_util.metrics import BATCH_KEY

import dr_gen.utils.evaluate as eu
import dr_gen.utils.model as mu

CRITERIONS = {"cross_entropy"}
OPTIMIZERS = {"sgd", "rmsprop", "adamw"}
LR_SCHEDULERS = "steplr", "cosineannealinglr", "exponentiallr"}

# md.clear_data() optional group_name=None
# md.log_data(data, group_name, ns=1)

def log_metrics(cfg, group_name, **kwargs):
    if cfg.md is None:
        print(">> WARNING: No metrics object for logging")
        return

    output = kwargs.get("output", None)
    target = kwargs.get("target", None)
    loss = kwargs.get("loss", None)

    if output is not None:
        cfg.md.log_data((BATCH_KEY, output.shape[0]))

    if not (output is None or target is None):
        acc1, acc5 = eu.accuracy(output, target, topk=(1, 5))
        cfg.md.log_data(('acc1', acc1))
        cfg.md.log_data(('acc5', acc5))

    if loss is not None:
        cfg.md.log_data(('loss', loss))


def train_epoch(cfg, epoch, model, dataloader, criterion, optimizer, metrics):
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
        acc1, acc5 = eu.accuracy(output, target, topk=(1, 5))
        metrics.train(
            {
                'loss': loss.item(),
                'acc1': acc1.item(), 
                'acc5': acc5.item(),
            },
            ns=image.shape[0],
        )
    return metrics


def eval_model(cfg, model, dataloader, criterion, metrics):
    model.eval()
    with torch.inference_mode():
        for image, target in dataloader:
            image = image.to(cfg.device, non_blocking=True)
            target = target.to(cfg.device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = eu.accuracy(output, target, topk=(1, 5))
            metrics.val(
                {
                    'loss': loss.item(),
                    'acc1': acc1.item(), 
                    'acc5': acc5.item(),
                },
                ns=image.shape[0],
            )
    return metrics


def train_loop(cfg, model, train_dl, val_dl=None):
    criterion = mu.get_criterion(cfg)
    optim, lr_sched = mu.get_optim(cfg, model)
    mu.checkpoint_model(cfg, model, "init_model")

    start_time = time.time()
    for epoch in range(cfg.epochs):
        md = GenMetrics(cfg)
        md.log(f">> Start Epoch: {epoch}")

        # Train
        md = train_epoch(cfg, epoch, model, train_dl, criterion, optim, md)
        lr_sched.step()
        md.agg_log("train")

        # Val
        if val_dl is not None:
            md = eval_model(cfg, model, val_dl, criterion, md)
            md.agg_log("val")

        mu.checkpoint_model(cfg, model, f"epoch_{epoch}")
        print()

    total_time = time.time() - start_time
    total_ts = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_ts}")
