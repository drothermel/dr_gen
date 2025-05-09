from collections import defaultdict
import math
import torch
from torch.utils.data import (
    Subset,
    SequentialSampler,
    RandomSampler,
)
from torch.utils.data.dataloader import default_collate
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2

import dr_util
import dr_gen.schemas as vu

DEFAULT_DATASET_CACHE_ROOT = "../data/"
DEFAULT_DOWNLOAD = True

# TODO: replace with timm
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

def get_dataset(
    dataset_name,
    source_split,
    root=DEFAULT_DATASET_CACHE_ROOT,
    transform=None,
    download=DEFAULT_DOWNLOAD,
):
    if dataset_name in ["cifar10", "cifar100"]:
        ds = dr_util.data_utils.get_cifar_dataset(
            dataset_name,
            source_split,
            root,
            transform=transform,
            download=download,
        )
    else:
        assert False
    return ds

def get_dataloaders(cfg, generator):
    vu.validate_dataset(cfg.data.name)

    # Extract the split source and ratio info
    splits = []
    split_by_source = {}
    for split in vu.SPLIT_NAMES:
        if split not in cfg.data:
            continue
        splits.append(split)
        source = cfg.data[split].source
        ratio = cfg.data[split].source_percent
        if source not in split_by_source:
            split_by_source[source] = []
        elif len(split_by_source) > 1:
            assert False, "Only two splits can share a source"
        split_by_source[source].append((split, ratio))

    # Get the split data, dividing a input data split if needed
    # using the data_split_seed
    split_data = {}
    for source, split_ratio_list in split_by_source.items():
        # Include transforms in the dataset if just one source
        if len(split_ratio_list == 1):
            split, _ = split_ratio_list[0]
            transform_cfg = cfg.data[split].transform
            split_data[split] = get_dataset(
                cfg.data.name,
                source,
                root=cfg.paths.dataset_cache_root,
                transform=build_transforms(transfomr_cfg),
                download=cfg.data.download,
            )
            continue

        # If source needs to be split, create dataset without
        # transforms and add them to the subsets instead
        dataset = get_dataset(
            cfg.data.name,
            source,
            root=cfg.paths.dataset_cache_root,
            transform=None,
            download=cfg.data.download,
        )
        subsets = dr_util.data_utils.split_data(
            dataset, ratio1, data_split_seed=cfg.data.split_seed,
        )
        for (split, _), subset in zip(split_ratio_list, subsets):
            transform_cfg = cfg.data[split].transform
            split_data[split] = dr_util.data_utils.TransformedSubset(
                subset, build_transforms(transform_cfg),
            )
            
    # For each split select the portion of the dataset specified
    split_dls = {}
    for split, ds in split_data.items():
        shuffle = cfg.data[split].shuffle
        sampler = RandomSampler() if shuffle else SequentialSampler(ds)
        # Note: SubsetRandomSampler(indices) might help in future
        split_dls[split] = torch.utils.data.DataLoader(
            ds, 
            battch_size=cfg[split].batch_size,
            sampler=sampler,
            num_workers=cfg.data.num_workers,
            collate_fn=default_collate,
            pin_memory=True,
            worker_init_fn=dr_util.determinism_utils.seed_worker,
            generator=generator,
        )
    return split_dls


