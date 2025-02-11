from pathlib import Path
import torch
import torchvision


def create_model(cfg, num_classes):
    model = torchvision.models.get_model(
        cfg.model.name,
        weights=cfg.model.weights,
        num_classes=num_classes,
    )
    if cfg.model.load_checkpoint is not None:
        print(f">> Loading checkpoint: {cfg.model.load_checkpoint}")
        checkpoint = torch.load(
            cfg.model.load_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
    model.to(cfg.device)
    return model


def checkpoint_model(cfg, model, checkpoint_name):
    chpt_dir = Path(cfg.model.write_checkpoint)
    chpt_dir.mkdir(parents=True, exist_ok=True)
    chpt_path = chpt_dir / f"{checkpoint_name}.pt"
    torch.save(
        {"model": model.state_dict()},
        chpt_path,
    )
    print(f">> Saved checkpoint to: {chpt_path}")
