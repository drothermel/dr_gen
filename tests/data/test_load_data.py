from unittest.mock import Mock

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torchvision.transforms import v2 as transforms_v2

import dr_gen.data.load_data as du
import dr_gen.schemas as vu

# ---------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------


@pytest.fixture
def base_cfg():
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="base_config", overrides=[])


@pytest.fixture
def transform_cfg():
    """Returns a base configuration mock with all transformations disabled."""
    cfg = Mock()
    cfg.random_crop = False
    cfg.crop_size = (32, 32)
    cfg.crop_padding = 4
    cfg.random_horizontal_flip = False
    cfg.random_horizontal_flip_prob = 0.5
    cfg.color_jitter = False
    cfg.jitter_brightness = 0.2
    cfg.normalize = False
    cfg.normalize_mean = [0.5, 0.5, 0.5]
    cfg.normalize_std = [0.5, 0.5, 0.5]
    return cfg


class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Fake __init__ functions for CIFAR datasets ---
def fake_cifar10_init(self, root, train, transform, target_transform, download):
    # Simply assign attributes without file checks.
    self.train = train
    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.download = download


def fake_cifar100_init(self, root, train, transform, target_transform, download):
    # Simply assign attributes without file checks.
    self.train = train
    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.download = download


# Dummy implementations to override various helpers used in get_dataloaders


def dummy_get_dataset(dataset_name, source_split, root, transform=None, download=False):
    # Always return a DummyDataset of length 20
    return DummyDataset(list(range(20)))


def dummy_get_ds_root(cfg):
    return "dummy_root"


def dummy_build_transforms(transform_cfg):
    # Return an identity transform (or simply None)
    return None


def dummy_get_transform_cfg(split, cfg):
    return {}


def dummy_get_download(cfg):
    return False


def dummy_get_source(split, cfg):
    # Return the source as specified in the configuration for that split,
    # or default to the split name.
    return cfg.data.get(split, {}).get("source", split)


def dummy_get_split_source_config(cfg):
    """For our dummy configuration assume:

      - "train" uses source "source1" with 80% of the data (range: 0.0 to 0.8)
      - "val" uses the same source "source1" for the remaining 20% (range: 0.8 to 1.0)
      - "eval" uses source "eval" with full data (range: 0.0 to 1.0)
    We return a tuple of:
      sources_used, and a callable that returns the range for a given split.
    """
    sources_used = ["source1", "eval"]

    def range_for_split(split):
        mapping = {"train": (0.0, 0.8), "val": (0.8, 1.0), "eval": (0.0, 1.0)}
        return mapping[split]

    return sources_used, range_for_split


def dummy_get_shuffle(split, cfg):
    # For testing purposes, fix shuffle to False
    return False


def dummy_validate_dataset(name):
    # Always validate successfully
    return True


def dummy_validate_split(split):
    # Allow only our known splits
    return split in ["train", "val", "eval"]


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

# ------------------- Cfg Getters -------------------------


@pytest.mark.parametrize(
    "split, cfg, expected",
    [
        # No configuration provided: should return the split.
        ("train", None, "train"),
        # Config provided but no data for the split: returns the split.
        ("train", OmegaConf.create({"data": {}}), "train"),
        # Split exists in config but without a "source" key: returns the split.
        ("train", OmegaConf.create({"data": {"train": {}}}), "train"),
        # Split exists and contains a "source": returns the custom source.
        (
            "train",
            OmegaConf.create({"data": {"train": {"source": "custom_train"}}}),
            "custom_train",
        ),
        # Another split missing from the config: returns the split.
        (
            "val",
            OmegaConf.create({"data": {"train": {"source": "custom_train"}}}),
            "val",
        ),
    ],
)
def test_get_source(split, cfg, expected) -> None:
    assert du.get_source(split, cfg) == expected


@pytest.mark.parametrize(
    "split, cfg, expected",
    [
        # If no config is provided, default is returned.
        (None, None, du.DEFAULT_SOURCE_PERCENT),
        # If config is None, default is returned even if split is given.
        ("train", None, du.DEFAULT_SOURCE_PERCENT),
        # If split is None, default is returned even if config is provided.
        (
            None,
            OmegaConf.create({"data": {"train": {"source_percent": 0.8}}}),
            du.DEFAULT_SOURCE_PERCENT,
        ),
        # If cfg is an empty dict, default is returned.
        ("train", OmegaConf.create({}), du.DEFAULT_SOURCE_PERCENT),
        # If cfg has no "data" key, default is returned.
        ("train", OmegaConf.create({"other_key": {}}), du.DEFAULT_SOURCE_PERCENT),
        # If "data" exists but split is missing, default is returned.
        ("train", OmegaConf.create({"data": {}}), du.DEFAULT_SOURCE_PERCENT),
        # If the split exists but without a "source_percent" key, default is returned.
        ("train", OmegaConf.create({"data": {"train": {}}}), du.DEFAULT_SOURCE_PERCENT),
        # If the split exists and has a custom "source_percent" value, return that.
        ("train", OmegaConf.create({"data": {"train": {"source_percent": 0.8}}}), 0.8),
        # Test with a different split (e.g., val) having a different source percent.
        ("val", OmegaConf.create({"data": {"val": {"source_percent": 0.2}}}), 0.2),
        # Test with eval split having a custom source percent.
        ("eval", OmegaConf.create({"data": {"eval": {"source_percent": 1.0}}}), 1.0),
    ],
)
def test_get_source_percent(split, cfg, expected) -> None:
    result = du.get_source_percent(split, cfg)
    assert result == expected


# ---------------------- Test Transforms -----------------------------


def test_basic_transforms(transform_cfg) -> None:
    """Test if base transforms (ToImage and ToDtype) are always included."""
    transforms = du.build_transforms(transform_cfg)
    assert isinstance(transforms, transforms_v2.Compose)
    assert isinstance(transforms.transforms[0], transforms_v2.ToImage)
    assert isinstance(transforms.transforms[1], transforms_v2.ToDtype)


def test_random_crop(transform_cfg) -> None:
    """Test if RandomCrop is included when enabled."""
    transform_cfg.random_crop = True
    transforms = du.build_transforms(transform_cfg)
    assert any(isinstance(t, transforms_v2.RandomCrop) for t in transforms.transforms)


def test_random_horizontal_flip(transform_cfg) -> None:
    """Test if RandomHorizontalFlip is included when enabled."""
    transform_cfg.random_horizontal_flip = True
    transforms = du.build_transforms(transform_cfg)
    assert any(
        isinstance(t, transforms_v2.RandomHorizontalFlip) for t in transforms.transforms
    )


def test_color_jitter(transform_cfg) -> None:
    """Test if ColorJitter is included when enabled."""
    transform_cfg.color_jitter = True
    transforms = du.build_transforms(transform_cfg)
    assert any(isinstance(t, transforms_v2.ColorJitter) for t in transforms.transforms)


def test_normalize(transform_cfg) -> None:
    """Test if Normalize is included when enabled."""
    transform_cfg.normalize = True
    transforms = du.build_transforms(transform_cfg)
    assert any(isinstance(t, transforms_v2.Normalize) for t in transforms.transforms)


def test_full_pipeline(transform_cfg) -> None:
    """Test if all transformations are included when enabled."""
    transform_cfg.random_crop = True
    transform_cfg.random_horizontal_flip = True
    transform_cfg.color_jitter = True
    transform_cfg.normalize = True

    transforms = du.build_transforms(transform_cfg)
    transform_types = {type(t) for t in transforms.transforms}

    expected_types = {
        transforms_v2.ToImage,
        transforms_v2.ToDtype,
        transforms_v2.RandomCrop,
        transforms_v2.RandomHorizontalFlip,
        transforms_v2.ColorJitter,
        transforms_v2.Normalize,
    }

    assert expected_types.issubset(transform_types)


# ------------ Test Simple Dataset and Dataloader Creation -------------


def test_get_dataset_cifar10(tmp_path, monkeypatch) -> None:
    """Test that get_dataset returns a CIFAR10 dataset when requested.

    We override the __init__ to bypass data validation.
    """
    from torchvision import datasets

    monkeypatch.setattr(datasets.CIFAR10, "__init__", fake_cifar10_init)
    root = str(tmp_path)
    dataset = du.get_dataset("cifar10", "train", root=root, download=False)
    assert isinstance(dataset, datasets.CIFAR10)
    # For CIFAR10, when source_split=="train", train should be True.
    assert dataset.train is True


def test_get_dataset_cifar100(tmp_path, monkeypatch) -> None:
    """Test that get_dataset returns a CIFAR100 dataset.

    We override the __init__ to bypass data validation.
    """
    from torchvision import datasets

    monkeypatch.setattr(datasets.CIFAR100, "__init__", fake_cifar100_init)
    root = str(tmp_path)
    dataset = du.get_dataset("cifar100", "val", root=root, download=False)
    assert isinstance(dataset, datasets.CIFAR100)
    # Since source_split is "val", the train flag should be False.
    assert dataset.train is False


def test_get_dataset_invalid() -> None:
    """Test that an invalid dataset name raises an assertion error.
    """
    with pytest.raises(AssertionError):
        du.get_dataset("unknown_dataset", "train")


# --- Tests for get_dataloader ---


def test_get_dataloader_default_config() -> None:
    """Test get_dataloader with no custom configuration.

    It should use the default batch size and number of workers.
    """
    dummy_data = DummyDataset(list(range(10)))
    sampler = SequentialSampler(dummy_data)
    generator = torch.Generator()
    valid_split = "train"
    # Assume vu.validate_split(valid_split) returns True.
    dataloader = du.get_dataloader(
        dummy_data, sampler, generator, valid_split, cfg=None
    )
    assert dataloader.batch_size == du.DEFAULT_BATCH_SIZE
    assert dataloader.num_workers == du.DEFAULT_NUM_WORKERS


def test_get_dataloader_custom_config() -> None:
    """Test get_dataloader when passing a custom OmegaConf configuration.

    The batch size and num_workers should be taken from the configuration.
    """
    dummy_data = DummyDataset(list(range(10)))
    sampler = SequentialSampler(dummy_data)
    generator = torch.Generator()
    valid_split = "train"
    # Create an OmegaConf configuration with custom values.
    cfg = OmegaConf.create(
        {
            "train": {"batch_size": 4},
            "data": {"num_workers": 2},
        }
    )
    dataloader = du.get_dataloader(dummy_data, sampler, generator, valid_split, cfg=cfg)
    assert dataloader.batch_size == 4
    assert dataloader.num_workers == 2


def test_get_dataloader_invalid_split(monkeypatch) -> None:
    """Test that get_dataloader raises an assertion error when an invalid split is provided.

    We patch vu.validate_split to only consider 'train', 'val', and 'eval' as valid.
    """
    dummy_data = DummyDataset(list(range(10)))
    sampler = SequentialSampler(dummy_data)
    generator = torch.Generator()
    invalid_split = "invalid_split"
    # Patch vu.validate_split for the purpose of this test.
    monkeypatch.setattr(vu, "validate_split", lambda s: s in ["train", "val", "eval"])
    with pytest.raises(AssertionError):
        du.get_dataloader(dummy_data, sampler, generator, invalid_split, cfg=None)


# ---------------------- Test Source Calcs -----------------------------


def test_get_split_source_config_defaults() -> None:
    """When no configuration is provided, each split should use itself as the source
    with the default percentage.

    """
    sources, ranges = du.get_split_source_config(cfg=None)

    # Each split uses itself as the source.
    expected_sources = list(vu.SPLIT_NAMES)
    # For each split, the range is (0, du.DEFAULT_SOURCE_PERCENT) i.e. (0, 1.0)
    expected_ranges = {
        split: (0, du.DEFAULT_SOURCE_PERCENT) for split in vu.SPLIT_NAMES
    }

    # Order may depend on SPLIT_NAMES; we compare sorted lists for safety.
    assert sorted(sources) == sorted(expected_sources)
    assert ranges == expected_ranges


def test_get_split_source_config_custom() -> None:
    """Test when a custom configuration maps multiple splits to the same source.

    For instance, both 'train' and 'val' use "official_train" with 0.8 and 0.2 respectively,
    while 'eval' uses the default (its own name and 1.0).
    """
    cfg = OmegaConf.create(
        {
            "data": {
                "train": {"source": "official_train", "source_percent": 0.8},
                "val": {"source": "official_train", "source_percent": 0.2},
                # "eval" is not specified, so it uses defaults.
            }
        }
    )
    sources, ranges = du.get_split_source_config(cfg)

    # For train: range is (0, 0.8); for val: since it uses the same source "official_train",
    # its range is (0.8, 1.0). For eval: default is used (source "eval", range (0, 1.0)).
    expected_ranges = {
        "train": (0, 0.8),
        "val": (0.8, 1.0),
        "eval": (0, du.DEFAULT_SOURCE_PERCENT),
    }
    # Since 'train' and 'val' share the same source, sources used
    # should be ["official_train", "eval"].
    expected_sources = ["official_train", "eval"]

    assert sorted(sources) == sorted(expected_sources)
    assert ranges == expected_ranges


def test_get_split_source_config_over_usage() -> None:
    """Test that an assertion error is raised if the total allocated percentage
    for a shared source exceeds 100% (i.e. > 1.0).

    """
    cfg = OmegaConf.create(
        {
            "data": {
                "train": {"source": "official_train", "source_percent": 0.6},
                "val": {"source": "official_train", "source_percent": 0.6},
                # "eval" remains default.
            }
        }
    )
    # Since train (0.6) and val (0.6) sum to 1.2 (> 1.0),
    # an AssertionError should be raised.
    with pytest.raises(
        AssertionError, match=">> Using more than 100% of official_train"
    ):
        du.get_split_source_config(cfg)


# ---------------------- Test Combination Utils -----------------------------


# Test for get_dataloaders
def test_get_dataloaders(monkeypatch) -> None:
    """This test constructs a dummy configuration (via OmegaConf) and then patches
    out helper functions so that get_dataloaders returns predictable DataLoaders.

    We then check that each returned DataLoader has the expected batch_size and
    that dataloaders are provided for the correct splits.
    """
    # Import the module that contains get_dataloaders and its helpers.
    import dr_gen.data.load_data as utils

    # Patch helper functions with our dummy implementations.
    monkeypatch.setattr(utils, "get_dataset", dummy_get_dataset)
    monkeypatch.setattr(utils, "get_ds_root", dummy_get_ds_root)
    monkeypatch.setattr(utils, "build_transforms", dummy_build_transforms)
    monkeypatch.setattr(utils, "get_transform_cfg", dummy_get_transform_cfg)
    monkeypatch.setattr(utils, "get_download", dummy_get_download)
    monkeypatch.setattr(utils, "get_source", dummy_get_source)
    monkeypatch.setattr(utils, "get_shuffle", dummy_get_shuffle)
    monkeypatch.setattr(utils.vu, "validate_dataset", dummy_validate_dataset)
    monkeypatch.setattr(utils.vu, "validate_split", dummy_validate_split)
    # Force vu.SPLIT_NAMES to be predictable.
    monkeypatch.setattr(utils.vu, "SPLIT_NAMES", ["train", "val", "eval"])

    # Create a dummy OmegaConf configuration.
    cfg = OmegaConf.create(
        {
            "train": {"batch_size": 4},
            "val": {"batch_size": 3},
            "eval": {"batch_size": 2},
            "data": {
                "name": "dummy",  # Dummy dataset name
                # For "train" and "val" we use the same source "source1"
                # but different percentages.
                "train": {"source_percent": 0.8, "source": "source1"},
                "val": {"source_percent": 0.2, "source": "source1"},
                # For "eval" use a separate source and default full percent.
                "eval": {"source_percent": 1.0, "source": "eval"},
                "num_workers": 0,
            },
        }
    )

    generator = torch.Generator()

    # Call the function under test.
    dls = utils.get_dataloaders(cfg, generator)

    # Check that we have DataLoaders for each expected split.
    assert set(dls.keys()) == {"train", "val", "eval"}

    # For each split, check that the DataLoader is an instance of
    # torch.utils.data.DataLoader
    # and that its batch_size matches what is specified in the configuration.
    for split, dl in dls.items():
        assert isinstance(dl, DataLoader)
        expected_bs = (
            OmegaConf.to_container(cfg)
            .get(split, {})
            .get("batch_size", utils.DEFAULT_BATCH_SIZE)
        )
        assert dl.batch_size == expected_bs

    # Optionally, verify that the dataset lengths (after subsetting) are as expected.
    # For "train" and "val", the dummy dataset length is 20 so:
    #   "train" should use approximately 80% (i.e. 16 samples) and
    #   "val" the remaining 20% (i.e. 4 samples).
    # For "eval", since the full dataset is used, length should be 20.
    # Note that due to floor rounding, you may have minor differences.
    train_dl = dls["train"]
    val_dl = dls["val"]
    eval_dl = dls["eval"]

    # Access the sampler length by checking the length of the dataset
    # that the sampler iterates over.
    # For "train" and "val", our dummy_get_dataloader calls
    # get_dataloader with a subset.
    train_indices = list(train_dl.sampler)
    val_indices = list(val_dl.sampler)
    eval_indices = list(eval_dl.sampler)  # full dataset sampler

    # Since our dummy dataset always has 20 elements:
    # For train, expected count ~ math.floor(20 * 0.8) = 16
    # For val, expected count ~ math.floor(20 * 0.2) = 4
    # For eval, expected count is 20.
    assert len(train_indices) in (15, 16)  # rounding may vary
    assert len(val_indices) in (4, 5)
    assert len(eval_indices) == 20
