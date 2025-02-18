from dataclasses import MISSING, dataclass
from enum import Enum
import logging

import dr_util.schema_utils as drutil_schemas
from dr_util.schema_utils import lenient_validate

# -------------- Validate Enums --------------

SPLIT_NAMES = ["train", "val", "eval"]
AVAIL_DATASETS = ["cifar10", "cifar100"]

def validate_split(split):
    if split not in SPLIT_NAMES:
        logging.error(f"Split {split} should be in {SPLIT_NAMES}")
        return False
    return True

def validate_dataset(dataset):
    if dataset not in AVAIL_DATASETS:
        logging.error(f"Dataset {dataset} should be in {AVAIL_DATASETS}")
        return False
    return True
        
# -------------- Validate Configs --------------


class ConfigType(Enum):
    USES_METRICS = "uses_metrics"

def get_schema(config_type):
    match config_type:
        case ConfigType.USES_METRICS.value:
            return drutil_schemas.UsingMetricsConfig
    return None

## -- Don't require, but prefer --

## For Data
# paths.dataset_cache_root
# cfg.data.num_workers
# cfg.data[split]
# .download
# .transform
# cfg[split].batch_size

## -- Required --

## For data
# cfg.data[split]
# .source
# .source_percent
# .name
# .shuffle

## Required for paths
