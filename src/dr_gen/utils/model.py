from pathlib import Path
import torch
import torchvision

def create_model(cfg, num_classes):
    model = torchvision.models.get_model(
        cfg.model.name,
        weights=cfg.model.weights,
        num_classes=num_classes,
    )
    model.to(cfg.device)
    return model

    
    
