from collections import defaultdict
import time

import torch
import dr_gen.utils.evaluate as eu

CRITERIONS = {"cross_entropy"}
OPTIMIZERS = {"sgd", "rmsprop", "adamw"}
LR_SCHEDULERS = {"steplr", "cosineannealinglr", "exponentiallr"}


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
    bs_key = [k for k in metrics.keys() if 'batch_size' in k][0]
    for k, v in metrics.items():
        agg_m[k] = 0
        for i, vv in enumerate(v):
            agg_m[k] += vv * metrics[bs_key][i]

    # Normalize
    total_bs = sum(metrics[bs_key])
    for k in agg_m.keys():
        agg_m[k] = agg_m[k] / total_bs

    return agg_m


def train_epoch(cfg, epoch, model, dataloader, criterion, optimizer):
    model.train()

    metrics = defaultdict(list)
    for i, (image, target) in enumerate(dataloader):
        start_time = time.time()
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
        update_metrics(
            metrics, image.shape[0], loss, output, target, "t",
        )
    return metrics


def eval_model(cfg, model, dataloader, criterion):
    model.eval()

    metrics = defaultdict(list)
    with torch.inference_mode():
        num_processed_samples = 0

        for image, target in dataloader:
            image = image.to(cfg.device, non_blocking=True)
            target = target.to(cfg.device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            update_metrics(
                metrics, image.shape[0], loss, output, target,
            )

            acc1, acc5 = eu.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            update_metrics(
                metrics, batch_size, loss, output, target,
            )


    return metrics


def train_loop(cfg, model, train_dl, val_dl=None):
    # Things in main
    # Setup logging here or elsewhere?
    # cfg.device = torch.device(cfg.device)
    # generator = ru.set_deterministic(cfg.seed)
    # split_dls = du.get_dataloaders(cfg, generator)
    # model = create_model(cfg, len(split_dls['train'].dataset.classes))
    # train_loop(cfg, model, split_dls['train'], val_dl=split_dls['val'])

    criterion = get_criterion(cfg)
    optim, lr_sched = get_optim(cfg, model)
    mu.checkpoint_model(cfg, model, "init_model")

    start_time = time.time()
    for epoch in range(cfg.epochs):
        print(f">> Start Epoch: {epoch}")
        
        # Train
        tm = train_epoch(cfg, epoch, model, train_dl, criterion, optim)
        lr_sched.step()
        tm_str = ' '.join([f"{k}={v}" for k, v in agg_metrics(tm))
        print(f":: TRAIN :: {tm_str}")

        # Val
        if val_dl is not None:
            vm = eval_model(cfg, model, val_dl, criterion)
            vm_str = ' '.join([f"{k}={v}" for k, v in agg_metrics(vm))
            print(f":: VAL :: {vm_str}")
        
        mu.checkpoint_model(cfg, model, f"epoch_{epoch}")
        print()

    total_time = time.time() - start_time
    total_ts = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_ts}")
    
