import pytest
import torch
from unittest.mock import Mock
from torchvision.transforms import v2 as transforms_v2

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose

import dr_gen.utils.data as du
import dr_gen.utils.run as ru


@pytest.fixture
def augment_cfg():
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


@pytest.fixture
def data_cfg():
    """Mocked config with deterministic seed and batch sizes."""
    cfg = Mock()
    cfg.seed = 101
    cfg.data = {
        "train": Mock(source="train", source_percent=0.6, shuffle=True),
        "val": Mock(source="train", source_percent=0.2, shuffle=False),
        "eval": Mock(source="eval", source_percent=0.2, shuffle=False),
    }
    return cfg


@pytest.fixture
def hydra_cfg():
    with initialize(config_path=f"../scripts/conf/", version_base=None):
        cfg = compose(
            config_name="config.yaml",
        )
    return cfg


def test_basic_transforms(augment_cfg):
    """Test if base transforms (ToImage and ToDtype) are always included."""
    transforms = du.get_transforms(augment_cfg)
    assert isinstance(transforms, transforms_v2.Compose)
    assert isinstance(transforms.transforms[0], transforms_v2.ToImage)
    assert isinstance(transforms.transforms[1], transforms_v2.ToDtype)


def test_random_crop(augment_cfg):
    """Test if RandomCrop is included when enabled."""
    augment_cfg.random_crop = True
    transforms = du.get_transforms(augment_cfg)
    assert any(isinstance(t, transforms_v2.RandomCrop) for t in transforms.transforms)


def test_random_horizontal_flip(augment_cfg):
    """Test if RandomHorizontalFlip is included when enabled."""
    augment_cfg.random_horizontal_flip = True
    transforms = du.get_transforms(augment_cfg)
    assert any(
        isinstance(t, transforms_v2.RandomHorizontalFlip) for t in transforms.transforms
    )


def test_color_jitter(augment_cfg):
    """Test if ColorJitter is included when enabled."""
    augment_cfg.color_jitter = True
    transforms = du.get_transforms(augment_cfg)
    assert any(isinstance(t, transforms_v2.ColorJitter) for t in transforms.transforms)


def test_normalize(augment_cfg):
    """Test if Normalize is included when enabled."""
    augment_cfg.normalize = True
    transforms = du.get_transforms(augment_cfg)
    assert any(isinstance(t, transforms_v2.Normalize) for t in transforms.transforms)


def test_full_pipeline(augment_cfg):
    """Test if all transformations are included when enabled."""
    augment_cfg.random_crop = True
    augment_cfg.random_horizontal_flip = True
    augment_cfg.color_jitter = True
    augment_cfg.normalize = True

    transforms = du.get_transforms(augment_cfg)
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


def test_prep_dataset_split_sources(data_cfg):
    """Test if prep_dataset_split_sources correctly groups sources and their percentages."""
    result = du.prep_dataset_split_sources(data_cfg)

    expected = {
        "train": [("train", 0.6), ("val", 0.2)],
        "eval": [("eval", 0.2)],
    }

    assert result == expected


def test_prep_dataset_split_sources_invalid(data_cfg):
    """Test if prep_dataset_split_sources raises assertion error when percentages exceed 1.0."""
    data_cfg.data["train"].source_percent = 0.7
    data_cfg.data["val"].source_percent = 0.5  # Exceeds 1.0

    with pytest.raises(
        AssertionError, match="Cannot use more than 100% of a data source"
    ):
        du.prep_dataset_split_sources(data_cfg)


@pytest.fixture
def source_percents():
    """Mocked source_percents structure returned by prep_dataset_split_sources."""
    return {
        "train": [("train", 0.6), ("val", 0.2)],
        "eval": [("eval", 0.2)],
    }


def test_get_source_range(data_cfg, source_percents):
    """Test if get_source_range returns correct percentage range for each split."""
    assert du.get_source_range(data_cfg, source_percents, "train") == (0.0, 0.6)
    assert du.get_source_range(data_cfg, source_percents, "val") == (0.6, 0.8)
    assert du.get_source_range(data_cfg, source_percents, "eval") == (0.0, 0.2)


def test_get_source_range_invalid(data_cfg, source_percents):
    """Test if get_source_range raises assertion error when an invalid split is provided."""
    with pytest.raises(AssertionError, match="Split test should be in .*"):
        du.get_source_range(data_cfg, source_percents, "test")


# def test_determinism(hydra_cfg):
# hydra_cfg.seed = 101
# generator = ru.set_deterministic(hydra_cfg.seed)
# dls = du.get_dataloaders(hydra_cfg, generator)
# data_out = {}
# for split in du.SPLIT_NAMES:
# spl_iter = iter(dls[split])
# for ind in range(3):
# feats, labels = next(spl_iter)
# assert feats.shape[0] == hydra_cfg[split].batch_size
# data_out[f"{split}_{ind}_features"] = feats
# data_out[f"{split}_{ind}_labels"] = labels
#
# del generator, dls
# hydra_cfg.seed = 202
# generator = ru.set_deterministic(hydra_cfg.seed)
# dls = du.get_dataloaders(hydra_cfg, generator)
# for split in du.SPLIT_NAMES:
# spl_iter = iter(dls[split])
# for ind in range(3):
# feats, labels = next(spl_iter)
# assert feats.shape[0] == hydra_cfg[split].batch_size
# assert torch.isclose(feats, data_out[f"{split}_{ind}_features"]).all().item()
# assert torch.isclose(labels, data_out[f"{split}_{ind}_labels"]).all().item()

# def test_shuffle(hydra_cfg):
# generator = ru.set_deterministic(hydra_cfg.seed)
# dls = du.get_dataloaders(hydra_cfg, generator)
#
# for split in du.SPLIT_NAMES:
# spl_iter = iter(dls[split])
# feats, labels = next(spl_iter)
# for _ in spl_iter:
# continue
#
# spl_iter = iter(dls[split])
# feats2, labels2 = next(spl_iter)
#
# if hydra_cfg.data[split].shuffle:
# assert not torch.isclose(labels, labels2).all().item()
# else:
# assert torch.isclose(labels, labels2).all().item()
#
# if (
# hydra_cfg.data[split].shuffle or
# hydra_cfg.data[split].transform.random_crop or
# hydra_cfg.data[split].transform.random_horizontal_flip or
# hydra_cfg.data[split].transform.color_jitter
# ):
# assert torch.isclose(feats, feats2).all().item()
#
# else:
# assert not torch.isclose(feats, feats2).all().item()
