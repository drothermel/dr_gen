import math
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2

from dr_gen.utils.run import seed_worker


AVAIL_DATASETS = {"cifar10", "cifar100"}

def get_transforms(augment_cfg):
    xfs = [
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
    ]

    if augment_cfg.random_crop:
        xfs.append(
            transforms_v2.RandomCrop(
                augment_cfg.crop_size,
                padding=augment_cfg.crop_padding,
            )
        )

    if augment_cfg.random_horizontal_flip:
        xfs.append(
            transforms_v2.RandomHorizontalFlip(
                p=augment_cfg.random_horizontal_flip_prob,
            )
        )

    if augment_cfg.color_jitter:
        xfs.append(
            transforms_v2.ColorJitter(
                brightness=augment_cfg.jitter_brightness,
            )
        )

    if augment_cfg.normalize:
        xfs.append(
            transforms_v2.Normalize(
                mean=augment_cfg.normalize_mean,
                std=augment_cfg.normalize_std,
            )
        )
    return transforms_v2.Compose([xfs])
    

def get_dataset(cfg, split):
    data_name_lower = cfg.data.name.lower()
    assert data_name_lower in AVAIL_DATASETS

    xf_type = split if split == "train" else "eval"
    xfs = get_transforms(cfg.data.transform[xf_type])
    print(xfs)

    if data_name_lower == "cifar10":
        use_train_data = (
            (split == "train") or
            (split == "val" and cfg.val.source == "train")
        )
        return datasets.CIFAR10(
            root=cfg.paths.dataset_cache_root,
            train=use_train_data,
            #transform=xfs,
            transform=None,
            target_transform=None,
            download=cfg.data.download,
        )
    elif data_name_lower == "cifar100":
        use_train_data = (
            (split == "train") or
            (split == "val" and cfg.val.source == "train")
        )
        return datasets.CIFAR100(
            root=cfg.paths.dataset_cache_root,
            train=use_train_data,
            #transform=xfs,
            transform=None,
            target_transform=None,
            download=cfg.data.download,
        )

def split_size(dataset, percent):
    return math.floor(len(dataset) * percent)
    
def setup_train_val_datasets(cfg, generator):
    train_ds = get_dataset(cfg, "train")
    val_ds = get_dataset(cfg, "val")

    #train_size = split_size(train_ds, cfg.train.source_percent)
    #train_indices = torch.randperm(len(train_ds))
    #if cfg.val.source == "train":
        #assert len(train_ds) == len(val_ds)
        #assert cfg.train.source_percent + cfg.val.source_percent == 1.0
        #train_ds = torch.utils.data.Subset(
            #train_ds, train_indices[:train_size],
        #)
        #val_ds = torch.utils.data.Subset(
            #val_ds, train_indices[train_size:],
        #)
    #elif cfg.train.source_percent != 1.0:
        #train_ds = torch.utils.data.Subset(
            #train_ds, train_indices[:train_size],
        #)
    #elif cfg.val.source_percent != 1.0:
        #val_indices = torch.randperm(len(val_ds))
        #val_size = split_size(val_ds, cfg.val.source_percent)
        #val_ds = torch.utils.data.Subset(
            #val_ds, val_indices[:val_size]
        #)
    return train_ds, val_ds


def setup_train_val_dataloaders(cfg, generator):
    train_ds, val_ds = setup_train_val_datasets(cfg, generator)
    train_sampler = torch.utils.data.RandomSampler()
    val_sampler = torch.utils.data.SequentialSampler()

    train_size = split_size(train_ds, cfg.train.source_percent)
    train_indices = torch.randperm(len(train_ds))
    if cfg.val.source == "train":
        assert len(train_ds) == len(val_ds)
        assert cfg.train.source_percent + cfg.val.source_percent == 1.0
        train_sampler = SubsetRandomSampler(train_indices[:train_size])
        val_sampler = SubsetRandomSampler(train_indices[train_size:])
    elif cfg.train.source_percent != 1.0:
        train_sampler = SubsetRandomSampler(train_indices[:train_size])
    elif cfg.val.source_percent != 1.0:
        val_indices = torch.randperm(len(val_ds))
        val_size = split_size(val_ds, cfg.val.source_percent)
        val_sampler = SubsetRandomSampler(val_indices[:val_size])


    train_dl = get_dataloader(cfg, train_ds, sampler, generator, "train")
    val_dl = get_dataloader(cfg, val_ds, sampler, generator, "val")
    return train_dl, val_dl


def get_dataloader(cfg, dataset, sampler, generator, split):
    # assumes determinism has been set
    # assumes dataset is tensors not pil images
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg[split].batch_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=default_collate,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    
    
