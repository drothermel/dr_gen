import torch
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2


AVAIL_DATASETS = {"cifar10", "cifar100"}


def get_dataset(data_cfg, paths_cfg, train=True, as_pil=False):
    data_name_lower = data_cfg.name.lower()
    assert data_name_lower in AVAIL_DATASETS

    # Generally load image as tensor not PIL
    tensor_transform = None
    if not as_pil:
        tensor_transform = transforms_v2.Compose([
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
        ])

    # Return the correct dataset
    if data_name_lower == "cifar10":
        return datasets.CIFAR10(
            root=paths_cfg.dataset_cache_root,
            train=train,
            transform=tensor_transform,
            target_transform=None,
            download=data_cfg.download,
        )
    elif data_name_lower == "cifar100":
        return datasets.CIFAR100(
            root=paths_cfg.dataset_cache_root,
            train=train,
            transform=tensor_transform,
            target_transform=None,
            download=data_cfg.download,
        )
