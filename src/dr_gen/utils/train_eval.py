from collections import defaultdict
from enum import Enum
from datetime import datetime
import time

import torch
import torch.nn as nn
import dr_gen.utils.evaluate as eu
import dr_gen.utils.model as mu

from dr_util.metrics import (
    Metrics, MetricsSubgroup,
    add_sum, add_list, agg_passthrough,
    agg_none, agg_batch_weighted_list_avg,
    create_logger,
)

CRITERIONS = {"cross_entropy"}
OPTIMIZERS = {"sgd", "rmsprop", "adamw"}
LR_SCHEDULERS = {"steplr", "cosineannealinglr", "exponentiallr"}

class GenMetricType(Enum):
    INT = "int"
    LIST = "list"
    BATCH_WEIGHTED_AVG_LIST = "batch_weighted_avg_list"
    AVG_LIST = "avg_list"

def agg_avg_list(data, key):
    data_sum = sum(data[key])
    return data_sum * 1.0 / len(data[key])


class GenMetricsSubgroup(MetricsSubgroup):
    def _init_data(self):
        if self.data_structure is None:
            return

        for key, data_type in self.data_structure.items():
            match data_type:
                case GenMetricType.INT.value:
                    self.data[key] = 0
                    self.add_fxns[key] = add_sum
                    self.agg_fxns[key] = agg_passthrough
                case GenMetricType.LIST.value:
                    self.data[key] = []
                    self.add_fxns[key] = add_list
                    self.agg_fxns[key] = agg_none
                case GenMetricType.BATCH_WEIGHTED_AVG_LIST.value:
                    self.data[key] = []
                    self.add_fxns[key] = add_list
                    self.agg_fxns[key] = agg_batch_weighted_list_avg
                case GenMetricType.AVG_LIST.value:
                    self.data[key] = []
                    self.add_fxns[key] = add_list
                    self.agg_fxns[key] = agg_avg_list

class GenMetrics(Metrics):
    def __init__(self, cfg):
        self.cfg = cfg
        self.group_names = ["train", "val"]

        # Initialize subgroups and loggers
        self.groups = {name: GenMetricsSubgroup(cfg, name) for name in self.group_names}
        self.loggers = [create_logger(cfg, lt) for lt in cfg.metrics.loggers]


def update_metrics(metrics, batch_size, loss, output, target, prefix=""):
    # Calc and Add Metrics
    acc1, acc5 = eu.accuracy(output, target, topk=(1, 5))
    img_ps = -1
    t_key = f"{prefix}_update_time"
    metrics[t_key].append(time.time())
    if len(metrics[t_key]) > 1:
        diff_t = metrics[t_key][-1] - metrics[t_key][-2]
        img_ps = batch_size * 1.0 / diff_t

    metrics[f"{prefix}_batch_size"].append(batch_size)
    metrics[f"{prefix}_loss"].append(loss.item())
    metrics[f"{prefix}_acc1"].append(acc1.item())
    metrics[f"{prefix}_acc5"].append(acc5.item())
    metrics[f"{prefix}_img_per_s"].append(img_ps)

    # Logging
    n = sum(metrics[f"{prefix}_batch_size"])
    print(f">> {prefix} {n}| {loss=} {acc1=} {acc5=} {img_ps=}")


def agg_metrics(metrics):
    agg_m = {}

    # Scale and sum
    bs_key = [k for k in metrics.keys() if "batch_size" in k][0]
    for k, v in metrics.items():
        agg_m[k] = 0
        for i, vv in enumerate(v):
            agg_m[k] += vv * metrics[bs_key][i]

    # Normalize
    total_bs = sum(metrics[bs_key])
    for k in agg_m.keys():
        agg_m[k] = agg_m[k] / total_bs

    return agg_m


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
