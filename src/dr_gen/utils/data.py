import math
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2

from dr_gen.utils.run import seed_worker


SPLIT_NAMES = ['train', 'val', 'eval']
AVAIL_DATASETS = {"cifar10", "cifar100"}


def get_transforms(augment_cfg):
    xfs_list = [
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
    ]
    if augment_cfg.random_crop:
        xfs_list.append(
            transforms_v2.RandomCrop(
                augment_cfg.crop_size,
                padding=augment_cfg.crop_padding,
            )
        )

    if augment_cfg.random_horizontal_flip:
        xfs_list.append(
            transforms_v2.RandomHorizontalFlip(
                p=augment_cfg.random_horizontal_flip_prob,
            )
        )

    if augment_cfg.color_jitter:
        xfs_list.append(
            transforms_v2.ColorJitter(
                brightness=augment_cfg.jitter_brightness,
            )
        )

    if augment_cfg.normalize:
        xfs_list.append(
            transforms_v2.Normalize(
                mean=augment_cfg.normalize_mean,
                std=augment_cfg.normalize_std,
            )
        )
    xfs = transforms_v2.Compose(xfs_list)
    return xfs


def prep_dataset_split_sources(cfg):
    # Process the source info for data into usable data struct
    source_percents = {}
    for spl in SPLIT_NAMES:
        spl_source = cfg.data[spl].source
        if spl_source not in source_percents:
            source_percents[spl_source] = []
        source_percents[spl_source].append((spl, cfg.data[spl].source_percent))

    # Validate source distribution
    for source, usages in source_percents.items():
        if sum([use[1] for use in usages]) > 1.0:
            assert False, "Cannot use more than 100% of a data source"

    return source_percents


def get_source_range(cfg, source_percents, split):
    assert split in SPLIT_NAMES, f"Split {split} should be in {SPLIT_NAMES}"
    my_source = cfg.data[split].source
    my_source_percents = source_percents[my_source]
    start_percent = 0
    end_percent = None
    for spl, perc in my_source_percents:
        end_percent = start_percent + perc
        if split == spl:
            return start_percent, end_percent
        start_percent = end_percent
        end_percent = None
    assert False, f"Split {split} should be in {my_source_percents}"

def get_source_dataset(cfg, split):
    assert split in SPLIT_NAMES, f"Split {split} should be in {SPLIT_NAMES}"
    my_source = cfg.data[split].source
    data_name_lower = cfg.data.name.lower()
    assert data_name_lower in AVAIL_DATASETS
    data_source = cfg.data[split].source
    xfs = get_transforms(cfg.data[split].transform)
    if data_name_lower == "cifar10":
        ds = datasets.CIFAR10(
            root=cfg.paths.dataset_cache_root,
            train=(data_source == "train"),
            transform=xfs,
            target_transform=None,
            download=cfg.data.download,
        )
    elif data_name_lower == "cifar100":
        ds = datasets.CIFAR100(
            root=cfg.paths.dataset_cache_root,
            train=(data_source == "train"),
            transform=xfs,
            target_transform=None,
            download=cfg.data.download,
        )
    return ds


def get_dataloaders(cfg, generator):
    # Prepare for splitting source datasets
    source_percents = prep_dataset_split_sources(cfg)
    source_indices = {}

    # For each split select the portion of the dataset specified
    split_dls = {}
    for split in SPLIT_NAMES:
        source = cfg.data[split].source
        shuffle = cfg.data[split].shuffle
        source_ds = get_source_dataset(cfg, split)
        # Shuffle once and then keep indices order fixed for split calc
        if source not in source_indices:
            source_indices[source] = torch.randperm(len(source_ds))
        indices = source_indices[source]

        start_perc, end_perc = get_source_range(
            cfg, source_percents, split
        )
        ns = len(source_ds)
        start_i = math.floor(ns * start_perc)
        end_i = ns if end_perc == 1.0 else math.floor(ns * end_perc)
        spl_indices = indices[start_i:end_i]
        if shuffle and (start_perc == 0  and end_perc == 1.0):
            ds = source_ds
            sampler = torch.utils.data.RandonmSampler()
        elif shuffle:
            ds = source_ds
            sampler = torch.utils.data.SubsetRandomSampler(spl_indices)
        else:
            ds = torch.utils.data.Subset(source_ds, spl_indices)
            sampler = torch.utils.data.SequentialSampler(ds)
        split_dls[split] = get_dataloader(
            cfg, ds, sampler, generator, split
        )
    return split_dls
    

def get_dataloader(cfg, dataset, sampler, generator, split):
    assert split in SPLIT_NAMES, f"Split {split} should be in {SPLIT_NAMES}"
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
