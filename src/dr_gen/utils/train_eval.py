from collections import defaultdict

import torch

CRITERIONS = {"cross_entropy"}


def get_criterion(cfg):
    crit = cfg.optim.loss.lower()
    assert crit in CRITERIONS
    return torch.nn.CrossEntropyLoss(
        label_smoothing=cfg.optim.label_smoothing,
    )


def get_optim(cfg, model):
    params = utils.set_weight_decay(model, cfg.optim.weight_decay)

    opt_name = cfg.optim.name.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.optim.lr,
            momentum=cfg.optim.momentum,
            weight_decay=cfg.optim.weight_decay,
            nesterov=cfg.optim.nesterov,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params,
            lr=cfg.optim.lr,
            momentum=cfg.optim.momentum,
            weight_decay=cfg.optim.weight_decay,
            eps=cfg.optim.eps,
            alpha=cfg.optim.alpha,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=cfg.optim.lr, weight_decay=args.weight_decay
        )
    else:
        assert False, f"Invalid optimizer {cfg.optim.name}.")

    assert cfg.optim.lr_scheduler.lower() == cfg.optim.lr_scheduler
    if cfg.optim.lr_scheduler == "steplr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.optim.lr_step_size,
            gamma=cfg.optim.lr_gamma,
        )
    elif cfg.optim.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs - cfg.optim.lr_warmup_epochs,
            eta_min=cfg.optim.lr_min
        )
    elif cfg.optim.lr_scheduler == "exponentiallr":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.optim.lr_gamma
        )
    else:
        assert False, f"Invalid LRSched {cfg.optim.lr_scheduler}."

    # TODO: Handle resume (line 341)
    # TODO: Handle test_only (line 353)

    return optimizer, lr_scheduler


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
            batch_size = image.shape[0]

            # Logging
            metrics["batch_size"].append(batch_size)
            metrics["loss"].append(loss.item())
            num_processed_samples += batch_size
            print(f">> {num_processed_samples}| {loss=}")

    return metrics

# Simplified from https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L405
def set_weight_decay(
    model,
    weight_decay,
):
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )
    params = []
    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            params.append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups

