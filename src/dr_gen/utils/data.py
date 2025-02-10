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


def split_size(dataset, percent):
    return math.floor(len(dataset) * percent)


def prep_dataset_split_sources(cfg):
    # Process the source info for data into usable data struct
    source_percents = {}
    for spl in ["train", "val", "eval"]:
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


def get_dataset_and_sampler(cfg, split):
    data_name_lower = cfg.data.name.lower()
    assert data_name_lower in AVAIL_DATASETS

    # For now repeat this (deterministically) each time we
    # get a dataset to avoid state, in future maybe do once
    data_source = cfg.data[split].source
    source_percents = prep_dataset_split_sources(cfg)
    start_perc, end_perc = get_source_range(cfg, source_percents, split)

    # Make the dataset based on the source with expected transforms
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

    # Setup the sampler for the dataset
    shuffle = cfg.data[split].shuffle
    ns = len(ds)
    indices = torch.randperm(ns)
    start_i = math.floor(ns * start_perc)
    end_i = ns if end_perc == 1.0 else math.floor(ns * end_perc)
    ds_indices = indices[start_i:end_i]
    if shuffle and (start_perc == 0 and end_perc == 1.0):
        print(">> Shuffle full dataset")
        sampler = torch.utils.data.RandomSampler()
    elif shuffle:
        print(f">> Shuffle subset dataset: {start_perc} {start_i} - {end_perc} {end_i}")
        sampler = torch.utils.data.SubsetRandomSampler(ds_indices)
    else:
        print(
            ">> Sequential subset dataset: {start_perc} {start_i} - {end_perc} {end_i}"
        )
        ds = torch.utils.data.Subset(ds, ds_indices)
        sampler = torch.utils.data.SequentialSampler()
    return ds, sampler


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
