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
    USES_DATA = "uses_data"

def get_schema(config_type):
    match config_type:
        case ConfigType.USES_METRICS.value:
            return drutil_schemas.UsingMetricsConfig
        case ConfigType.USES_DATA.value:
            return UsingDataConfig
    return None


#########################################################
#                  Config Definitions
#########################################################

@lenient_validate
@dataclass
class DataConfig:
    name: str = ???

#########################################################
#             Config Interface Definitions
#########################################################

@lenient_validate
@dataclass
class UsingDataConfig:
    data: type = DataConfig

## Using Data is setup to also use the following (optionally)
#
# split.batch_size 
#
# paths.dataset_cache_root
#
# data:
#   download
#   num_workers
#   split:
#     source
#     source_percent
#     shuffle
#     transform
## ----------------------------------------------------------
