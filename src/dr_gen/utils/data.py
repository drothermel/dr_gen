import torch
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2


AVAIL_DATASETS = {"cifar10", "cifar100"}


def get_dataset(data_cfg, paths_cfg, train=True, as_pil=False):
    data_name_lower = data_cfg.name.lower()
    assert data_name_lower in AVAIL_DATASETS

    # Generally load image as tensor not PIL
    tensor_transforms = []
    if not as_pil:
        tensor_transform = [
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
        ]


    # Return the correct dataset
    if data_name_lower == "cifar10":
        extra_transforms = []
        if train:
            extra_transforms = [
                transforms_v2.RandomCrop(32, padding=4),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ColorJitter(brightness=63 / 255),
            ]

        txs = transforms_v2.Compose([
            *tensor_transform,
            *extra_transforms,
            transforms_v2.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ])
        return datasets.CIFAR10(
            root=paths_cfg.dataset_cache_root,
            train=train,
            transform=txs,
            target_transform=None,
            download=data_cfg.download,
        )
    elif data_name_lower == "cifar100":
        extra_transforms = []
        if train:
            extra_transforms = [
                transforms_v2.RandomCrop(32, padding=4),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.ColorJitter(brightness=63 / 255),
            ]

        txs = transforms_v2.Compose([
            *tensor_transform,
            *extra_transforms,
            transforms_v2.Normalize(
                mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            ),
        ])
        return datasets.CIFAR100(
            root=paths_cfg.dataset_cache_root,
            train=train,
            transform=txs,
            target_transform=None,
            download=data_cfg.download,
        )
