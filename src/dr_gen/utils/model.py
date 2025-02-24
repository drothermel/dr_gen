from pathlib import Path
import torch
import torchvision

import dr_gen.schemas as vu
from dr_gen.schemas import (
    OptimizerTypes,
    LRSchedTypes,
    CriterionTypes,
)

# Match the torch defaults

OPTIM_DEFAULTS = {
    "momentum": 0,
    "weight_decay": 0,
    "nesterov": False,
    "eps": 1e-08,
    "alpha": 0.99,
}

# These match the torch defaults
LRSCHED_DEFAULTS = {
    "gamma": 0.1,
}

CRITERION_DEFAULTS = {
    "label_smoothing": 0.0,
}

# ================== cfg free ==================


def create_optim(name, model_params, optim_params):
    assert "lr" in optim_params
    match name:
        case OptimizerTypes.SGD.value:
            return torch.optim.SGD(
                model_params,
                **{ k: v for k, v in optim_params.items() if k in [
                    'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov',
                ]},
            )
        case OptimizerTypes.RMSPROP.value:
            return torch.optim.RMSprop(
                model_params,
                **{ k:v for k, v in optim_params.items() if k in [
                    'lr', 'alpha', 'eps', 'weight_decay', 'momentum',
                ]},
            )
        case OptimizerTypes.ADAMW.value:
            return torch.optim.AdamW(
                model_params,
                **{ k:v for k, v in optim_params.items() if k in [
                    'lr', 'betas', 'eps', 'weight_decay', 'amsgrad',
                ]},
            )


def create_lrsched(name, optimizer, lrsched_params):
    match name:
        case None:
            return None
        case LRSchedTypes.STEP_LR.value:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                **lrsched_params,
            )
        case LRSchedTypes.EXPONENTIAL_LR.value:
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                **lrsched_params,
            )


# ================== cfg ==================


# Config Req: cfg.model.name
def create_model(cfg, num_classes):
    model = torchvision.models.get_model(
        cfg.model.name,
        weights=cfg.model.get("weights", None),
        num_classes=num_classes,
    )
    return model


# Config Req: cfg.optim.name, cfg.optim.lr
def create_optim_lrsched(cfg, model):
    vu.validate_optimizer(cfg.optim.name)
    vu.validate_lrsched(cfg.optim.get("lr_scheduler", None))
    model_params = model.parameters()

    # ---------- Optim -----------
    optim_params = {k: cfg.optim.get(k, v) for k, v in OPTIM_DEFAULTS.items()}
    optim_params["lr"] = cfg.optim.lr
    optimizer = create_optim(cfg.optim.name, model_params, optim_params)

    # ---------- LR Sched -----------
    lrsched_params = {k: cfg.optim.get(k, v) for k, v in LRSCHED_DEFAULTS.items()}
    if "step_size" in cfg.optim:
        lrsched_params["step_size"] = cfg.optim.step_size
    lr_scheduler = create_lrsched(
        cfg.optim.get("lr_scheduler", None),
        optimizer,
        lrsched_params,
    )
    return optimizer, lr_scheduler


def get_model_optim_lrsched(cfg, num_classes):
    model = create_model(cfg, num_classes)
    optimizer = None
    lr_scheduler = None
    if cfg.get("load_checkpoint", None) is not None:
        if cfg.md is not None:
            cfg.md.log(f">> Loading checkpoint: {cfg.load_checkpoint}")
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


def checkpoint_model(cfg, model, checkpoint_name, optim=None, lrsched=None):
    if cfg.get("write_checkpoint", None) is None:
        return

    chpt_dir = Path(cfg.write_checkpoint)
    chpt_dir.mkdir(parents=True, exist_ok=True)
    chpt_path = chpt_dir / f"{checkpoint_name}.pt"
    chpt = {
        k: v
        for k, v in [
            ("model", model.state_dict()),
            ("optimizer", optim),
            ("lr_scheduler", lrsched),
        ]
    }
    torch.save(chpt, chpt_path)
    if cfg.md is not None:
        cfg.md.log(f">> Saved checkpoint to: {chpt_path}")
