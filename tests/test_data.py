"""Tests for simplified data loading functionality."""

from typing import Any
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms_v2

import dr_gen.data as du


class DummyDataset(Dataset):
    """Helper class for testing data loading functionality."""

    def __init__(self, data) -> None:
        """Initialize DummyDataset with provided data."""
        self.data = data

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:  # noqa: ANN401
        """Get item at the specified index."""
        return self.data[idx]


@pytest.fixture
def transform_cfg():
    """Returns a base configuration mock with all transformations disabled."""
    cfg = OmegaConf.create({
        "random_crop": False,
        "crop_size": (32, 32),
        "crop_padding": 4,
        "random_horizontal_flip": False,
        "random_horizontal_flip_prob": 0.5,
        "color_jitter": False,
        "jitter_brightness": 0.2,
        "normalize": False,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
    })
    return cfg


# Test dataset loading
def test_get_dataset_cifar10(tmp_path, monkeypatch):
    """Test that get_dataset returns a CIFAR10 dataset when requested."""
    from torchvision import datasets

    def fake_cifar10_init(
        self, root, train, transform, target_transform, download
    ) -> None:
        self.train = train
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

    monkeypatch.setattr(datasets.CIFAR10, "__init__", fake_cifar10_init)
    root = str(tmp_path)
    dataset = du.get_dataset("cifar10", "train", root=root, download=False)
    assert isinstance(dataset, datasets.CIFAR10)
    assert dataset.train is True


def test_get_dataset_invalid():
    """Test that an invalid dataset name raises an error."""
    with pytest.raises(ValueError, match="Unsupported dataset"):
        du.get_dataset("unknown_dataset", "train")


# Test transforms
def test_basic_transforms(transform_cfg):
    """Test if base transforms (ToImage and ToDtype) are always included."""
    transforms = du.build_transforms(transform_cfg)
    assert isinstance(transforms, transforms_v2.Compose)
    assert isinstance(transforms.transforms[0], transforms_v2.ToImage)
    assert isinstance(transforms.transforms[1], transforms_v2.ToDtype)


def test_random_crop(transform_cfg):
    """Test if RandomCrop is included when enabled."""
    transform_cfg.random_crop = True
    transforms = du.build_transforms(transform_cfg)
    assert any(isinstance(t, transforms_v2.RandomCrop) for t in transforms.transforms)


def test_random_horizontal_flip(transform_cfg):
    """Test if RandomHorizontalFlip is included when enabled."""
    transform_cfg.random_horizontal_flip = True
    transforms = du.build_transforms(transform_cfg)
    assert any(
        isinstance(t, transforms_v2.RandomHorizontalFlip) for t in transforms.transforms
    )


def test_color_jitter(transform_cfg):
    """Test if ColorJitter is included when enabled."""
    transform_cfg.color_jitter = True
    transforms = du.build_transforms(transform_cfg)
    assert any(isinstance(t, transforms_v2.ColorJitter) for t in transforms.transforms)


def test_normalize(transform_cfg):
    """Test if Normalize is included when enabled."""
    transform_cfg.normalize = True
    transforms = du.build_transforms(transform_cfg)
    assert any(isinstance(t, transforms_v2.Normalize) for t in transforms.transforms)


def test_full_pipeline(transform_cfg):
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


# Test simplified dataloader creation
def test_get_dataloaders_simple(monkeypatch):
    """Test the simplified get_dataloaders function."""

    # Mock the dataset loading
    def mock_get_dataset(
        dataset_name, source_split, root, download=True
    ) -> DummyDataset:
        return DummyDataset(list(range(100)))  # 100 samples

    # Mock the split_data function
    def mock_split_data(
        dataset, ratio, data_split_seed
    ) -> tuple[DummyDataset, DummyDataset]:
        data = list(range(100))
        split_point = int(len(data) * ratio)
        train_data = data[:split_point]
        val_data = data[split_point:]
        return DummyDataset(train_data), DummyDataset(val_data)

    # Mock timm functions
    def mock_resolve_model_data_config(model) -> dict[str, Any]:
        return {}

    def mock_create_transform(**kwargs: Any) -> transforms_v2.Compose:  # noqa: ANN401
        return transforms_v2.Compose([transforms_v2.ToTensor()])

    # Apply patches
    monkeypatch.setattr(du, "get_dataset", mock_get_dataset)
    monkeypatch.setattr("dr_util.data_utils.split_data", mock_split_data)
    monkeypatch.setattr(
        "timm.data.resolve_model_data_config", mock_resolve_model_data_config
    )
    monkeypatch.setattr("timm.data.create_transform", mock_create_transform)

    # Create test configuration
    cfg = OmegaConf.create(
        {
            "data": {
                "name": "cifar10",
                "download": True,
                "num_workers": 0,
                "transform_type": "timm",
                "train_val_ratio": 0.8,
                "split_seed": 42,
            },
            "paths": {"dataset_cache_root": "/tmp"},  # noqa: S108
            "train": {"batch_size": 32},
            "val": {"batch_size": 64},
            "eval": {"batch_size": 64},
        }
    )

    # Create dummy model
    model = torch.nn.Linear(10, 1)
    generator = torch.Generator()

    # Test the function
    dataloaders = du.get_dataloaders(cfg, generator, model)

    # Verify we get the expected dataloaders
    assert set(dataloaders.keys()) == {"train", "val", "eval"}

    # Verify all are DataLoader instances
    for dl in dataloaders.values():
        assert isinstance(dl, DataLoader)

    # Verify batch sizes
    assert dataloaders["train"].batch_size == 32
    assert dataloaders["val"].batch_size == 64
    assert dataloaders["eval"].batch_size == 64


def test_get_dataloaders_invalid_dataset():
    """Test that invalid dataset raises error."""
    cfg = OmegaConf.create({"data": {"name": "invalid_dataset"}})

    model = torch.nn.Linear(10, 1)
    generator = torch.Generator()

    with pytest.raises(ValueError, match="Dataset invalid_dataset should be in"):
        du.get_dataloaders(cfg, generator, model)


def test_create_transforms():
    """Test transform creation functions."""
    cfg = OmegaConf.create(
        {
            "data": {
                "transform_type": "pycil",
                "train_transform": {
                    "random_crop": True,
                    "crop_size": (32, 32),
                    "crop_padding": 4,
                },
                "eval_transform": {
                    "normalize": True,
                    "normalize_mean": [0.5, 0.5, 0.5],
                    "normalize_std": [0.5, 0.5, 0.5],
                },
            }
        }
    )

    model = torch.nn.Linear(10, 1)

    train_transform = du.create_train_transform(cfg, model)
    eval_transform = du.create_eval_transform(cfg, model)

    assert train_transform is not None
    assert eval_transform is not None
    assert isinstance(train_transform, transforms_v2.Compose)
    assert isinstance(eval_transform, transforms_v2.Compose)


def test_create_transforms_invalid_type():
    """Test that invalid transform type raises error."""
    cfg = OmegaConf.create({"data": {"transform_type": "invalid"}})

    model = torch.nn.Linear(10, 1)

    with pytest.raises(ValueError, match="Invalid transform type"):
        du.create_train_transform(cfg, model)

    with pytest.raises(ValueError, match="Invalid transform type"):
        du.create_eval_transform(cfg, model)
