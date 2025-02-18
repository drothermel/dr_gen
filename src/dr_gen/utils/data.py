from collections import defaultdict
import math
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2

import dr_gen.schemas as vu
from dr_gen.utils.run import seed_worker

# -------------------- Loader Utils -------------------

DEFAULT_DATASET_CACHE_ROOT = "../data/"
DEFAULT_NUM_WORKERS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_DOWNLOAD = True
DEFAULT_SOURCE_PERCENT = 1.0


def get_transforms(augment_cfg):
    if augment_cfg is None:
        return None

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

# Config Reqs: cfg.data required, but can be empty
def get_split_source_config(cfg):
    split_source_range = {}
    source_usage = defaultdict(int)
    for split in vu.SPLIT_NAMES:
        if split not in cfg.data:
            continue

        # Validate split source availablility
        source = cfg.data[split].source
        source_p = cfg.data[split].get(
            "source_percent", DEFAULT_SOURCE_PERCENT,
        )
        source_start = source_usage[source]
        source_end = source_usage[source] + source_p
        split_source_range[split] = (source_start, source_end)
        if source_end > 1.0:
            assert False, f">> Using more than 100% of {source}"
        source_usage[source] = source_end
    sources_used = list(source_usage.keys())
    return sources_used, split_source_range


# -------------------- Config Based Loaders -------------------

def get_source_dataset(cfg, split):
    assert vu.validate_split(split)
    assert vu.validate_dataset(cfg.data.name)
    assert split in cfg.data

    # Use defaults to get dataset config info to make more general
    dataset_cache_root = cfg.get("paths", {}).get(
        "dataset_cache_root", DEFAULT_DATASET_CACHE_ROOT
    )
    transform_cfg = cfg.data[split].get('transform', None)
    transform = get_transforms(transform_cfg)
    return get_dataset(
        cfg.data.name,
        cfg.data[split].source,
        root=dataset_cache_root,
        transform=transform,
        download=cfg.data.get("download", DEFAULT_DOWNLOAD),
    )

# Config Reqs: cfg.data required, but can be empty
def get_dataloaders(cfg, generator):
    # Each split comes from a single source, but each source can
    # supply multiple splits so fix the source range percents
    # before shuffling based on random seed
    splits = list(cfg.data.keys())
    sources_used, split_source_ranges = get_split_source_config(cfg)
    ds_root = cfg.data[splits[0]].get(
        "dataset_cache_root",
        DEFAULT_DATASET_CACHE_ROOT,
    )

    # For each source used, shuffle the indices once to have diff 
    # data splits per random seed.  Then fix to ensure the splits are
    # non-overlapping even if they come from the same source.
    source_indices = {
        torch.randperm(len(get_dataset(cfg.data.name, source, root=ds_root)))
        for source in sources_used
    }

    # For each split select the portion of the dataset specified
    split_dls = {}
    for split in splits:
        vu.validate_split(split)

        # Select the indices for this split
        source = cfg.data[split].source
        num_source_samples = len(source_indices[source])
        start_perc, end_perc = split_source_ranges(split)
        start_i = math.floor(num_source_samples * start_perc)
        end_i = math.floor(num_source_samples * end_perc)
        indices = source_indices[source][start_i : end_i]

        # Use the indices to create a split sampler
        if shuffle and (start_perc == 0 and end_perc == 1.0):
            ds = get_source_dataset(cfg, split)
            sampler = torch.utils.data.RandomSampler()
        elif shuffle:
            ds = get_source_dataset(cfg, split)
            sampler = torch.utils.data.SubsetRandomSampler(indices)
        else:
            ds = torch.utils.data.Subset(
                get_source_dataset(cfg, split),
                indices,
            )
            sampler = torch.utils.data.SequentialSampler(ds)
        split_dls[split] = get_dataloader(cfg, ds, sampler, generator, split)
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
            root=cfg.paths.dataset_cache_root,
            train=(data_source == "train"),
            transform=xfs,
            target_transform=None,
            download=cfg.data.download,
        )
    else:
        assert False
    return ds

# Config Reqs: cfg can be None
def get_dataloader(cfg, dataset, sampler, generator, split):
    assert vu.validate_split(split)

    # Set some defaults to make this more broadly usable
    cfg = cfg if cfg is not None else {}
    cfg_split = cfg.get(split, {})
    cfg_data = cfg.get("data", {})
    batch_size = cfg_split.get("batch_size", DEFAULT_BATCH_SIZE)
    num_workers = cfg_data.get("num_workers", DEFAULT_NUM_WORKERS)
    if "batch_size" not in cfg_split or "num_workers" not in cfg_data:
        warn_msg = ">> Using defaults in get_dataloader, intentional?"
        cfg.md.log(warn_msg) if "md" in cfg else print(warn_msg)

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
