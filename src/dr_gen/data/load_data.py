from typing import Any

import dr_util.data_utils as du

# Constants for data splitting validation
MAX_SUPPORTED_SPLITS = 2
import dr_util.determinism_utils as dtu
import timm
import timm.data
import torch
from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
    Subset,
)
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import v2 as transforms_v2

import dr_gen.schemas as vu

DEFAULT_DATASET_CACHE_ROOT = "../data/"
DEFAULT_DOWNLOAD = True


# TODO: replace with timm
def build_transforms(xfm_cfg: Any) -> Any:  # noqa: ANN401
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
    return transforms_v2.Compose(xfs_list)


def get_dataset(
    dataset_name: str,
    source_split: str,
    root: str = DEFAULT_DATASET_CACHE_ROOT,
    transform: Any = None,
    download: bool = DEFAULT_DOWNLOAD,
) -> Any:
    if dataset_name in ["cifar10", "cifar100"]:
        ds = du.get_cifar_dataset(
            dataset_name,
            source_split,
            root,
            transform=transform,
            download=download,
        )
    else:
        assert False
    return ds


def _parse_and_validate_config(
    cfg: Any,  # noqa: ANN401
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    """Parses data configuration, identifies sources, and prepares for splitting."""
    vu.validate_dataset(cfg.data.name)

    parsed_configs = {}
    source_usage_info: dict[str, Any] = {}  # Tracks how original sources are utilized

    for split_name_key in vu.SPLIT_NAMES:  # e.g., 'train', 'val', 'eval'
        if split_name_key not in cfg.data:
            continue

        split_config_from_file = cfg.data[split_name_key]
        # Assuming batch_size is directly under cfg.train, cfg.val etc.
        batch_size = cfg[split_name_key].batch_size

        current_split_params = {
            "source_dataset_name": split_config_from_file.source,
            "source_percent": split_config_from_file.source_percent,
            "use_percent": split_config_from_file.use_percent,
            "dataloader_shuffle": split_config_from_file.shuffle,
            "transform_config": split_config_from_file.transform,
            "batch_size": batch_size,
        }
        parsed_configs[split_name_key] = current_split_params

        # Track usage of source datasets (e.g., CIFAR10 'train' source)
        source_name = split_config_from_file.source
        if source_name not in source_usage_info:
            source_usage_info[source_name] = []
        source_usage_info[source_name].append(
            {
                # e.g. 'train' (the key for the final dataloader)
                "target_split_key": split_name_key,
                "source_percent_allocation": split_config_from_file.source_percent,
            }
        )

    # Validate that percentages for a shared source sum to 1.0
    for source_name, usages in source_usage_info.items():
        if len(usages) > 1:  # Source is being split
            total_percent = sum(u["source_percent_allocation"] for u in usages)
            total_tensor = torch.tensor(total_percent, dtype=torch.float32)
            one_tensor = torch.tensor(1.0, dtype=torch.float32)
            if not torch.isclose(total_tensor, one_tensor):
                raise ValueError(
                    f"Source '{source_name}' has split percentages that do not "
                    f"sum to 1.0. Usages: {usages}, Sum: {total_percent}"
                )
            # Sort by target_split_key for consistent processing order
            # (e.g., if split_data relies on order of ratios)
            source_usage_info[source_name] = sorted(
                usages, key=lambda x: x["target_split_key"]
            )
            if len(usages) > MAX_SUPPORTED_SPLITS:  # Based on original code's assertion
                raise ValueError(
                    "Configuration implies a source is split into more than "
                    "two parts, which is not supported by the current logic."
                )

    unique_source_dataset_names = list(source_usage_info.keys())
    return parsed_configs, source_usage_info, unique_source_dataset_names


def _load_source_datasets(
    cfg: Any, unique_source_names_to_load: list[str]  # noqa: ANN401
) -> dict[str, Any]:
    """Loads raw datasets for each unique source. Transforms are NOT applied here."""
    loaded_raw_datasets = {}
    for source_name in unique_source_names_to_load:
        loaded_raw_datasets[source_name] = get_dataset(
            dataset_name=cfg.data.name,
            source_split=source_name,  # e.g., 'train' or 'eval' for CIFAR10 source
            root=cfg.paths.dataset_cache_root,
            transform=None,  # Load raw data; transforms applied later
            download=cfg.data.download,
        )
    return loaded_raw_datasets


def _perform_source_splitting(
    raw_datasets: dict[str, Any],
    source_usage_details: dict[str, Any],
    data_split_seed: int,
) -> dict[str, Any]:
    """Splits raw datasets based on 'source_percent' using 'data_split_seed'.
    Output dict maps target_split_key (e.g., 'train', 'val') to its dataset.
    """
    datasets_after_source_split = {}

    for source_name, usages in source_usage_details.items():
        original_dataset_for_source = raw_datasets[source_name]

        if len(usages) == 1:
            # Source is used by only one final split (no splitting)
            usage_info = usages[0]
            target_key = usage_info["target_split_key"]
            # If source_percent is < 1.0, it implies a non-splitting subset,
            # but the prompt's `split_data` is for splitting one source into two.
            # We assume if len(usages)==1, the source_percent should be 1.0 or
            # it means "take this much of the source".
            # The current interpretation of source_percent is primarily for
            # train/val style splitting. If source_percent < 1.0 for a single
            # user, it's ambiguous with use_percent. For now, assume single
            # usage means it takes the whole identified source_dataset.
            # `use_percent` will handle taking a portion of this.
            source_percent = usage_info["source_percent_allocation"]
            if not torch.isclose(torch.tensor(source_percent), torch.tensor(1.0)):
                pass
            datasets_after_source_split[target_key] = original_dataset_for_source

        # Source is split into two target splits
        elif len(usages) == MAX_SUPPORTED_SPLITS:
            target1_info = usages[0]  # Assumes sorted order from parsing
            target2_info = usages[1]

            ratio_for_target1 = target1_info["source_percent_allocation"]

            # print(
            #     f"Splitting source '{source_name}' for "
            #     f"'{target1_info['target_split_key']}' "
            #     f"({ratio_for_target1*100}%) and "
            #     f"'{target2_info['target_split_key']}' "
            #     f"({(1-ratio_for_target1)*100}%) using seed {data_split_seed}."
            # )

            dataset_for_target1, dataset_for_target2 = du.split_data(
                original_dataset_for_source,
                ratio_for_target1,
                data_split_seed=data_split_seed,
            )
            key1 = target1_info["target_split_key"]
            key2 = target2_info["target_split_key"]
            datasets_after_source_split[key1] = dataset_for_target1
            datasets_after_source_split[key2] = dataset_for_target2
        # Else: >2 usages, already handled by validation in _parse_and_validate_config

    return datasets_after_source_split


def _apply_use_percent(datasets_after_source_splitting, parsed_configs):
    """Applies 'use_percent' to further subset the datasets.
    Currently takes the first N elements of the (potentially shuffled by
    data_split_seed) input dataset.
    """
    final_subsetted_datasets = {}
    for target_key, dataset_obj in datasets_after_source_splitting.items():
        use_p = parsed_configs[target_key]["use_percent"]

        if use_p < 1.0:
            original_len = len(dataset_obj)
            num_to_use = int(original_len * use_p)
            if (
                num_to_use == 0 and use_p > 0 and original_len > 0
            ):  # Ensure at least one sample if percent > 0
                num_to_use = 1

            # print(
            #     f"Applying use_percent={use_p} to '{target_key}'. "
            #     f"Original size: {original_len}, new size: {num_to_use}"
            # )

            # Takes the first N elements. If the dataset_obj is already a result of
            # a seeded shuffle (from split_data),
            # this selection is deterministic on a consistently shuffled set.
            indices_to_keep = list(range(num_to_use))
            final_subsetted_datasets[target_key] = Subset(dataset_obj, indices_to_keep)
        else:
            final_subsetted_datasets[target_key] = dataset_obj  # No change, use 100%

    return final_subsetted_datasets


def _apply_transforms(cfg, datasets_to_be_transformed, parsed_configs, model):
    """Applies transforms to the datasets."""
    datasets_with_transforms = {}
    for target_key, dataset_obj in datasets_to_be_transformed.items():
        data_config = timm.data.resolve_model_data_config(model)
        if cfg.data.transform_type == "timm":
            is_training = target_key == "train"
            transform_function = timm.data.create_transform(
                **data_config, is_training=is_training
            )
        elif cfg.data.transform_type == "pycil":
            transform_config_details = parsed_configs[target_key]["transform_config"]
            # Returns None if no transforms
            transform_function = build_transforms(transform_config_details)
        else:
            assert False, f">> Invalid transform type: {cfg.data.transform_type}"

        if transform_function:
            datasets_with_transforms[target_key] = du.TransformedSubset(
                dataset_obj, transform_function
            )
        else:
            datasets_with_transforms[target_key] = dataset_obj  # No transforms to apply

    return datasets_with_transforms


def _create_dataloaders_from_final_datasets(
    final_datasets_for_loaders: dict[str, Any],
    parsed_configs: dict[str, dict[str, Any]],
    num_workers_global: int,
    main_torch_generator: torch.Generator,
) -> dict[str, torch.utils.data.DataLoader[Any]]:
    """Creates DataLoaders for each processed dataset."""
    data_loaders_map = {}
    for target_key, final_dataset in final_datasets_for_loaders.items():
        config_for_this_loader = parsed_configs[target_key]

        use_shuffle_in_dl = config_for_this_loader["dataloader_shuffle"]
        current_batch_size = config_for_this_loader["batch_size"]

        # For reproducible shuffling in DataLoader, RandomSampler can
        # take a generator.
        if use_shuffle_in_dl:
            sampler = RandomSampler(final_dataset, generator=main_torch_generator)
        else:
            sampler = SequentialSampler(final_dataset)

        data_loaders_map[target_key] = torch.utils.data.DataLoader(
            final_dataset,
            batch_size=current_batch_size,
            sampler=sampler,
            num_workers=num_workers_global,
            collate_fn=default_collate,
            # pin_memory=True, # Common optimization
            pin_memory=False,
            worker_init_fn=dtu.seed_worker,
            # generator for DataLoader (>=1.9) can also be set for workers
            # if worker_init_fn isn't covering all needs.
        )
        # print(
        #     f"Created DataLoader for '{target_key}': "
        #     f"batch_size={current_batch_size}, shuffle={use_shuffle_in_dl}, "
        #     f"num_samples={len(final_dataset)}"
        # )

    return data_loaders_map


def get_dataloaders_refactored(cfg, main_torch_generator, model):
    """Refactored function to create dataloaders, incorporating source_percent splits
    (with data_split_seed) and use_percent subsampling, with a modular design.

    Args:
        cfg: The main configuration object.
        main_torch_generator: A torch.Generator for reproducible random operations
                              (e.g., shuffling in DataLoaders).
    """
    # 1. Parse and Validate Configuration
    (
        parsed_split_level_configs,
        source_dataset_usage_info,
        unique_raw_source_names,
    ) = _parse_and_validate_config(cfg)

    # 2. Load Raw Source Datasets (e.g., full CIFAR10 'train' or 'eval' splits)
    raw_source_datasets_map = _load_source_datasets(cfg, unique_raw_source_names)

    # 3. Perform Source Splitting (using source_percent and data_split_seed)
    # This creates the initial datasets for your model's 'train', 'val'
    # phases from the raw source datasets. E.g., splits CIFAR10 'train'
    # into model 'train' & 'val'.
    datasets_after_source_split = _perform_source_splitting(
        raw_source_datasets_map, source_dataset_usage_info, cfg.data.data_split_seed
    )

    # 4. Apply 'use_percent' Sub-sampling
    # Takes a percentage of the datasets resulting from step 3.
    datasets_after_use_percent = _apply_use_percent(
        datasets_after_source_split, parsed_split_level_configs
    )

    # 5. Apply Transforms
    # Applies data augmentation and preprocessing.
    final_datasets_ready_for_loader = _apply_transforms(
        cfg,
        datasets_after_use_percent,
        parsed_split_level_configs,
        model,
    )

    # 6. Create DataLoaders
    return _create_dataloaders_from_final_datasets(
        final_datasets_ready_for_loader,
        parsed_split_level_configs,
        cfg.data.num_workers,
        main_torch_generator,
    )


"""
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
        subsets = du.split_data(
            dataset, ratio1, data_split_seed=cfg.data.split_seed,
        )
        for (split, _), subset in zip(split_ratio_list, subsets):
            transform_cfg = cfg.data[split].transform
            split_data[split] = du.TransformedSubset(
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
            worker_init_fn=dtu.seed_worker,
            generator=generator,
        )
    return split_dls
"""
