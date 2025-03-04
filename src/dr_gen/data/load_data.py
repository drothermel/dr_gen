from collections import defaultdict
import math
import torch
from torch.utils.data import (
    Subset,
    SequentialSampler,
    RandomSampler,
    SubsetRandomSampler,
)
from torch.utils.data.dataloader import default_collate
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2

import dr_gen.schemas as vu
from dr_gen.utils.run import seed_worker

# ---------------- Default and Config Utils ---------------

DEFAULT_DATASET_CACHE_ROOT = "../data/"
DEFAULT_NUM_WORKERS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_DOWNLOAD = True
DEFAULT_SOURCE_PERCENT = 1.0
DEFAULT_SHUFFLE = True


# Source is usually the split itself, but sometimes we need to split
# a source into multiple splits (eg "train" becomes train and val).
# If not specified, use the split as the source.
def get_source(split, cfg=None):
    if cfg is None or split not in cfg.data or "source" not in cfg.data[split]:
        return split
    return cfg.data[split].source


def get_source_percent(split=None, cfg=None):
    source_p = DEFAULT_SOURCE_PERCENT
    if cfg is not None and split is not None:
        source_p = (
            cfg.get("data", {})
            .get(split, {})
            .get(
                "source_percent",
                DEFAULT_SOURCE_PERCENT,
            )
        )
    return source_p


# Use a default dataset location if not provided
def get_ds_root(cfg=None):
    ds_root = DEFAULT_DATASET_CACHE_ROOT
    if cfg is not None:
        ds_root = cfg.get("paths", {}).get(
            "dataset_cache_root", DEFAULT_DATASET_CACHE_ROOT
        )
    return ds_root


# Transforms aren't required, so any of these can be None
def get_transform_cfg(split=None, cfg=None):
    if cfg is None or split is None:
        return None
    return cfg.get("data", {}).get(split, {}).get("transform", None)


# Download param isn't required so any can be None
def get_download(cfg=None):
    if cfg is None:
        return None
    return cfg.get("data", {}).get("download", DEFAULT_DOWNLOAD)


def get_shuffle(split=None, cfg=None):
    if cfg is None or split is None:
        return None
    return cfg.get("data", {}).get(split, {}).get("shuffle", DEFAULT_SHUFFLE)


# -------------------- Loader Utils -------------------


# Config Reqs: None
# If a transform is selected, its hpms must be included.
def build_transforms(xfm_cfg):
    if xfm_cfg is None:
        return None

    xfs_list = [
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
    ]
    if xfm_cfg.get("random_crop", False):
        xfs_list.append(
            transforms_v2.RandomCrop(
                xfm_cfg.crop_size,
                padding=xfm_cfg.crop_padding,
            )
        )

    if xfm_cfg.get("random_horizontal_flip", False):
        xfs_list.append(
            transforms_v2.RandomHorizontalFlip(
                p=xfm_cfg.random_horizontal_flip_prob,
            )
        )

    if xfm_cfg.get("color_jitter", False):
        xfs_list.append(
            transforms_v2.ColorJitter(
                brightness=xfm_cfg.jitter_brightness,
            )
        )

    if xfm_cfg.get("normalize", False):
        xfs_list.append(
            transforms_v2.Normalize(
                mean=xfm_cfg.normalize_mean,
                std=xfm_cfg.normalize_std,
            )
        )
    xfs = transforms_v2.Compose(xfs_list)
    return xfs


# Config Reqs: None, default is source=split, percent=1.0
def get_split_source_config(cfg):
    split_source_range_dict = {}
    source_usage = defaultdict(int)
    for split in vu.SPLIT_NAMES:
        # Validate source cfg and save split source usage range
        source = get_source(split, cfg=cfg)
        source_p = get_source_percent(split=split, cfg=cfg)
        source_start = source_usage[source]
        source_end = source_usage[source] + source_p
        split_source_range_dict[split] = (source_start, source_end)
        if source_end > 1.0:
            assert False, f">> Using more than 100% of {source}"
        source_usage[source] = source_end
    sources_used = list(source_usage.keys())
    return sources_used, split_source_range_dict


# -------------------- Config Based Loaders -------------------


# Config Req: cfg.data.name
# Select transforms based on split, data based on source
def get_source_dataset(cfg, split, source):
    assert vu.validate_dataset(cfg.data.name)

    # Use defaults to get dataset config info to make more general
    return get_dataset(
        cfg.data.name,
        source,
        root=get_ds_root(cfg=cfg),
        transform=build_transforms(get_transform_cfg(split=split, cfg=cfg)),
        download=get_download(cfg=cfg),
    )


# Config Reqs: cfg.data.name, and cfg.data must contain the
#    name of any desired splits.
def get_dataloaders(cfg, generator):
    vu.validate_dataset(cfg.data.name)

    # Each split comes from a single source, but each source can
    # supply multiple splits so fix the source range percents
    # before shuffling based on random seed
    splits = [k for k in vu.SPLIT_NAMES if k in cfg.data]
    sources_used, split_source_rs = get_split_source_config(cfg)

    # For each source used, shuffle the indices once to have diff
    # data splits per random seed.  Then fix to ensure the splits are
    # non-overlapping even if they come from the same source.
    ds_root = get_ds_root(cfg=cfg)
    source_indices = {
        source: torch.randperm(len(get_dataset(cfg.data.name, source, root=ds_root)))
        for source in sources_used
    }

    # For each split select the portion of the dataset specified
    split_dls = {}
    for split in splits:
        vu.validate_split(split)
        shuffle = get_shuffle(split=split, cfg=cfg)

        # Select the indices for this split
        source = get_source(split, cfg=cfg)
        num_source_samples = len(source_indices[source])
        start_perc, end_perc = split_source_rs[split]
        ds = get_source_dataset(cfg, split, source)
        if (start_perc, end_perc) == (0.0, 1.0):
            # For full dataset, only the sampler changes with shuffle
            sampler = RandomSampler() if shuffle else SequentialSampler(ds)
        else:
            # For partial dataset, have to select indices of the subest
            start_i = math.floor(num_source_samples * start_perc)
            end_i = math.floor(num_source_samples * end_perc)
            indices = source_indices[source][start_i:end_i]
            if shuffle:
                # Then select those indices via sampler if we want shuffle
                sampler = SubsetRandomSampler(indices)
            else:
                # Or just subset the data if we don't want shuffle
                ds = Subset(ds, indices)
                sampler = SequentialSampler(ds)
        split_dls[split] = get_dataloader(ds, sampler, generator, split, cfg=cfg)
    return split_dls


# -------------------- General Purpose Loaders -------------------


# Config Reqs: None
def get_dataset(
    dataset_name,
    source_split,
    root=DEFAULT_DATASET_CACHE_ROOT,
    transform=None,
    download=DEFAULT_DOWNLOAD,
):
    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(
            root=root,
            train=(source_split == "train"),
            transform=transform,
            target_transform=None,
            download=download,
        )
    elif dataset_name == "cifar100":
        ds = datasets.CIFAR100(
            root=root,
            train=(source_split == "train"),
            transform=transform,
            target_transform=None,
            download=download,
        )
    else:
        assert False
    return ds


# Config Reqs: None
def get_dataloader(dataset, sampler, generator, split, cfg=None):
    assert vu.validate_split(split)

    # Set some defaults to make this more broadly usable
    cfg = cfg if cfg is not None else {}
    batch_size = cfg.get(split, {}).get("batch_size", DEFAULT_BATCH_SIZE)
    num_workers = cfg.get("data", {}).get("num_workers", DEFAULT_NUM_WORKERS)

    # assumes determinism has been set
    # assumes dataset is tensors not pil images
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=default_collate,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
