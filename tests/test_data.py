import pytest
import torch
from unittest.mock import Mock
from torchvision.transforms import v2 as transforms_v2

import dr_gen.utils.data as du



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
    assert any(isinstance(t, transforms_v2.RandomHorizontalFlip) for t in transforms.transforms)

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
