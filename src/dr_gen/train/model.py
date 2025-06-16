from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import timm
import torch
import torchvision
from timm.optim import create_optimizer
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn.parameter import Parameter

import dr_gen.schemas as vu
from dr_gen.schemas import (
    CriterionTypes,
    LRSchedTypes,
    OptimizerTypes,
)

# Match the torch defaults

OPTIM_DEFAULTS = {
    "momentum": 0,
    "weight_decay": 0,
    "nesterov": False,
    "eps": 1e-08,
    "alpha": 0.99,
    "step_size": 30,
}

# These match the torch defaults
LRSCHED_DEFAULTS = {
    "gamma": 0.1,
    "warmup_epochs": 0,
    "warmup_start_lr": 0.0,
}

CRITERION_DEFAULTS = {
    "label_smoothing": 0.0,
}

# ================== cfg free ==================


def create_optim(
    name: str, model_params: Iterator[Parameter], optim_params: dict[str, Any]
) -> torch.optim.Optimizer:
    assert "lr" in optim_params
    match name:
        case "timm_sgd":
            args = SimpleNamespace()
            args.weight_decay = optim_params["weight_decay"]
            args.lr = optim_params["lr"]
            args.opt = "sgd"
            args.momentum = optim_params["momentum"]
            return create_optimizer(args, model_params)
        case OptimizerTypes.SGD.value:
            return torch.optim.SGD(
                model_params,
                **{
                    k: v
                    for k, v in optim_params.items()
                    if k
                    in [
                        "lr",
                        "momentum",
                        "dampening",
                        "weight_decay",
                        "nesterov",
                    ]
                },
            )
        case OptimizerTypes.RMSPROP.value:
            return torch.optim.RMSprop(
                model_params,
                **{
                    k: v
                    for k, v in optim_params.items()
                    if k
                    in [
                        "lr",
                        "alpha",
                        "eps",
                        "weight_decay",
                        "momentum",
                    ]
                },
            )
        case OptimizerTypes.ADAMW.value:
            return torch.optim.AdamW(
                model_params,
                **{
                    k: v
                    for k, v in optim_params.items()
                    if k
                    in [
                        "lr",
                        "betas",
                        "eps",
                        "weight_decay",
                        "amsgrad",
                    ]
                },
            )


def create_lrsched(cfg: Any, optimizer: torch.optim.Optimizer) -> Any:  # noqa: ANN401
    match cfg.optim.lr_scheduler:
        case None:
            return None
        case "timm_cosine":
            return CosineLRScheduler(
                optimizer,
                t_initial=cfg.epochs,
                warmup_t=cfg.optim.warmup_epochs,
                warmup_lr_init=cfg.optim.warmup_start_lr,
                lr_min=cfg.optim.lr_min,
                cycle_limit=cfg.optim.cycle_limit,
            )
        case LRSchedTypes.STEP_LR.value:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.optim.get("step_size", 30),
                gamma=cfg.optim.gamma,
            )


# ================== cfg ==================


# Config Req: cfg.model.name
def create_model(cfg: Any, num_classes: int) -> torch.nn.Module:  # noqa: ANN401
    assert "resnet" in cfg.model.name
    if cfg.model.source == "torchvision":
        weights_name = cfg.model.get("weights", None)
        if weights_name == "None":
            weights_name = None
        if weights_name is None:
            assert cfg.weight_type != "pretrained"
        else:
            assert cfg.weight_type == "pretrained"
        model = torchvision.models.get_model(
            cfg.model.name,
            weights=weights_name,
        )
        if model.fc.out_features != num_classes:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif cfg.model.source == "timm":
        if cfg.weight_type == "pretrained":
            model_name = cfg.model.weights
            pretrained = True
        else:
            model_name = cfg.model.name
            pretrained = False
        model = timm.create_model(
            model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    else:
        assert False, f"{cfg.model.source} {cfg.model.weights}"
    return model


# Config Req: cfg.optim.name, cfg.optim.lr
def create_optim_lrsched(
    cfg: Any, model: torch.nn.Module  # noqa: ANN401
) -> tuple[torch.optim.Optimizer, Any]:
    model_params = model.parameters()

    # ---------- Optim -----------
    optim_params = {k: cfg.optim.get(k, v) for k, v in OPTIM_DEFAULTS.items()}
    optim_params["lr"] = cfg.optim.lr
    optimizer = create_optim(cfg.optim.name, model_params, optim_params)

    # ---------- LR Sched -----------
    lr_scheduler = create_lrsched(
        cfg,
        optimizer,
    )
    return optimizer, lr_scheduler


def get_model_optim_lrsched(
    cfg: Any, num_classes: int, md: Any = None  # noqa: ANN401
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, Any]:
    model = create_model(cfg, num_classes)
    optimizer = None
    lr_scheduler = None
    if cfg.get("load_checkpoint", None) is not None:
        if md is not None:
            md.log(f">> Loading checkpoint: {cfg.load_checkpoint}")
        checkpoint = torch.load(
            cfg.load_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer = checkpoint.get("optimizer", None)
        lr_scheduler = checkpoint.get("lr_scheduler", None)

    model.to(cfg.device)
    if optimizer is None or lr_scheduler is None:
        optimizer, lr_scheduler = create_optim_lrsched(cfg, model)
    return model, optimizer, lr_scheduler


def get_criterion(cfg):
    vu.validate_criterion(cfg.optim.loss)
    crit_params = {k: cfg.optim.get(k, v) for k, v in CRITERION_DEFAULTS.items()}

    match cfg.optim.loss:
        case CriterionTypes.CROSS_ENTROPY.value:
            return torch.nn.CrossEntropyLoss(**crit_params)


def checkpoint_model(cfg, model, checkpoint_name, optim=None, lrsched=None, md=None):
    if cfg.get("write_checkpoint", None) is None:
        return

    chpt_dir = Path(cfg.write_checkpoint)
    chpt_dir.mkdir(parents=True, exist_ok=True)
    chpt_path = chpt_dir / f"{checkpoint_name}.pt"
    chpt = {"model": model.state_dict(), "optimizer": optim, "lr_scheduler": lrsched}
    torch.save(chpt, chpt_path)
    if md is not None:
        md.log(f">> Saved checkpoint to: {chpt_path}")
