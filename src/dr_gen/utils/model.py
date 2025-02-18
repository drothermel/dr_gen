from pathlib import Path
import torch
import torchvision

CRITERIONS = {"cross_entropy"}
OPTIMIZERS = {"sgd", "rmsprop", "adamw"}
LR_SCHEDULERS = {"steplr", "cosineannealinglr", "exponentiallr"}


def create_model(cfg, num_classes):
    model = torchvision.models.get_model(
        cfg.model.name,
        weights=cfg.model.weights,
        num_classes=num_classes,
    )
    return model


def create_optim_lrsched(cfg, model):
    opt_name = cfg.optim.name.lower()
    assert opt_name in OPTIMIZERS, f"Invalid optimizer {cfg.optim.name}."

    params = model.parameters()

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
            params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
    else:
        assert False, f"Invalid optimizer {cfg.optim.name}"

    assert cfg.optim.lr_scheduler in LR_SCHEDULERS
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
            eta_min=cfg.optim.lr_min,
        )
    elif cfg.optim.lr_scheduler == "exponentiallr":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.optim.lr_gamma
        )
    else:
        assert False, f"Invalid LRSched {cfg.optim.lr_scheduler}."
    return optimizer, lr_scheduler


def get_model_optim_lrsched(cfg, num_classes):
    model = create_model(cfg, num_classes)
    if cfg.load_checkpoint is not None:
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
    crit = cfg.optim.loss.lower()
    assert crit in CRITERIONS
    return torch.nn.CrossEntropyLoss(
        label_smoothing=cfg.optim.label_smoothing,
    )


def checkpoint_model(cfg, model, checkpoint_name, optim=None, lrsched=None):
    if cfg.write_checkpoint is None:
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
