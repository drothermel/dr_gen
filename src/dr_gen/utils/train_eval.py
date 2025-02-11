from collections import defaultdict

import torch

CRITERIONS = {"cross_entropy"}


def get_criterion(cfg):
    crit = cfg.optim.loss.lower()
    assert crit in CRITERIONS
    return torch.nn.CrossEntropyLoss(
        label_smoothing=cfg.optim.label_smoothing,
    )


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
