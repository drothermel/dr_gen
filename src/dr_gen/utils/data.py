import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2

AVAIL_DATASETS = {"cifar10", "cifar100"}


def get_dataset(data_cfg, paths_cfg, train=True):
    data_name_lower = data_cfg.name.lower()
    assert data_name_lower in AVAIL_DATASETS

    if data_name_lower == "cifar10":
        return datasets.CIFAR10(
            root=paths_cfg.dataset_cache_root,
            train=train,
            download=data_cfg.download,
        )
    elif data_name_lower == "cifar100":
        return datasets.CIFAR100(
            root=paths_cfg.dataset_cache_root,
            train=train,
            download=data_cfg.download,
        )
