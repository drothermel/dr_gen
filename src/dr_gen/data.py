"""Simplified data loading for train/val/eval splits with transforms."""

from collections.abc import Callable
from typing import Any

import dr_util.data_utils as du
import dr_util.determinism_utils as dtu
import timm
import timm.data
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import v2 as transforms_v2

from dr_gen.config import AVAIL_DATASETS

# Default values
DEFAULT_DATASET_CACHE_ROOT = "../data/"
DEFAULT_DOWNLOAD = True


def get_dataset(
    dataset_name: str,
    source_split: str,
    root: str = DEFAULT_DATASET_CACHE_ROOT,
    transform: Callable[[Any], Any] | None = None,
    download: bool = DEFAULT_DOWNLOAD,
) -> Dataset[Any]:
    """Load a dataset by name and split."""
    if dataset_name in ["cifar10", "cifar100"]:
        return du.get_cifar_dataset(
            dataset_name,
            source_split,
            root,
            transform=transform,
            download=download,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_transforms(xfm_cfg: DictConfig | None) -> Callable[[Any], Any] | None:
    """Build transform pipeline from configuration."""
    if xfm_cfg is None:
        return None

    transforms = [
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
    ]

    if xfm_cfg.get("random_crop", False):
        transforms.append(
            transforms_v2.RandomCrop(
                xfm_cfg.crop_size,
                padding=xfm_cfg.crop_padding,
            )
        )

    if xfm_cfg.get("random_horizontal_flip", False):
        transforms.append(
            transforms_v2.RandomHorizontalFlip(
                p=xfm_cfg.random_horizontal_flip_prob,
            )
        )

    if xfm_cfg.get("color_jitter", False):
        transforms.append(
            transforms_v2.ColorJitter(
                brightness=xfm_cfg.jitter_brightness,
            )
        )

    if xfm_cfg.get("normalize", False):
        transforms.append(
            transforms_v2.Normalize(
                mean=xfm_cfg.normalize_mean,
                std=xfm_cfg.normalize_std,
            )
        )

    return transforms_v2.Compose(transforms)


def create_train_transform(
    cfg: DictConfig, model: torch.nn.Module
) -> Callable[[Any], Any] | None:
    """Create training transforms (with data augmentation)."""
    if cfg.data.transform_type == "timm":
        data_config = timm.data.resolve_model_data_config(model)
        return timm.data.create_transform(**data_config, is_training=True)
    if cfg.data.transform_type == "pycil":
        return build_transforms(cfg.data.get("train_transform"))
    raise ValueError(f"Invalid transform type: {cfg.data.transform_type}")


def create_eval_transform(
    cfg: DictConfig, model: torch.nn.Module
) -> Callable[[Any], Any] | None:
    """Create evaluation transforms (no data augmentation)."""
    if cfg.data.transform_type == "timm":
        data_config = timm.data.resolve_model_data_config(model)
        return timm.data.create_transform(**data_config, is_training=False)
    if cfg.data.transform_type == "pycil":
        return build_transforms(cfg.data.get("eval_transform"))
    raise ValueError(f"Invalid transform type: {cfg.data.transform_type}")


def get_dataloaders(
    cfg: DictConfig, generator: torch.Generator, model: torch.nn.Module
) -> dict[str, DataLoader[Any]]:
    """Create train/val/eval dataloaders with simplified logic.

    1. Load dataset's train and test splits
    2. Split train into train/val using configured ratio
    3. Apply training transforms to train split
    4. Apply eval transforms to val and eval splits
    5. Create dataloaders with appropriate settings
    """
    # Validate dataset
    if cfg.data.name not in AVAIL_DATASETS:
        raise ValueError(f"Dataset {cfg.data.name} should be in {AVAIL_DATASETS}")

    # 1. Load raw datasets
    train_dataset = get_dataset(
        cfg.data.name, "train", cfg.paths.dataset_cache_root, download=cfg.data.download
    )

    # Use "test" for CIFAR datasets, but could be "eval" for others
    eval_split_name = "test" if cfg.data.name in ["cifar10", "cifar100"] else "eval"
    eval_dataset = get_dataset(
        cfg.data.name,
        eval_split_name,
        cfg.paths.dataset_cache_root,
        download=cfg.data.download,
    )

    # 2. Split train into train/val
    train_ratio = cfg.data.get("train_val_ratio", 0.8)  # Default 80/20 split
    train_split, val_split = du.split_data(
        train_dataset, train_ratio, data_split_seed=cfg.data.get("split_seed", 42)
    )

    # 3. Create transforms
    train_transform = create_train_transform(cfg, model)
    eval_transform = create_eval_transform(cfg, model)

    # 4. Apply transforms to datasets
    datasets = {}
    if train_transform:
        datasets["train"] = du.TransformedSubset(train_split, train_transform)
    else:
        datasets["train"] = train_split

    if eval_transform:
        datasets["val"] = du.TransformedSubset(val_split, eval_transform)
        datasets["eval"] = du.TransformedSubset(eval_dataset, eval_transform)
    else:
        datasets["val"] = val_split
        datasets["eval"] = eval_dataset

    # 5. Create dataloaders
    dataloaders = {}

    # Training dataloader (shuffled)
    dataloaders["train"] = DataLoader(
        datasets["train"],
        batch_size=cfg.train.batch_size,
        sampler=RandomSampler(datasets["train"], generator=generator),
        num_workers=cfg.data.num_workers,
        collate_fn=default_collate,
        pin_memory=False,
        worker_init_fn=dtu.seed_worker,
    )

    # Validation dataloader (not shuffled)
    dataloaders["val"] = DataLoader(
        datasets["val"],
        batch_size=cfg.val.batch_size,
        sampler=SequentialSampler(datasets["val"]),
        num_workers=cfg.data.num_workers,
        collate_fn=default_collate,
        pin_memory=False,
        worker_init_fn=dtu.seed_worker,
    )

    # Evaluation dataloader (not shuffled)
    dataloaders["eval"] = DataLoader(
        datasets["eval"],
        batch_size=cfg.eval.batch_size,
        sampler=SequentialSampler(datasets["eval"]),
        num_workers=cfg.data.num_workers,
        collate_fn=default_collate,
        pin_memory=False,
        worker_init_fn=dtu.seed_worker,
    )

    return dataloaders


# Keep the refactored function name for backward compatibility
get_dataloaders_refactored = get_dataloaders
