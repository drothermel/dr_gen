from dataclasses import MISSING, dataclass
from enum import Enum
import logging

import dr_util.schema_utils as drutil_schemas
from dr_util.schema_utils import lenient_validate

# -------------- Validate Enums --------------

SPLIT_NAMES = ["train", "val", "eval"]
AVAIL_DATASETS = ["cifar10", "cifar100"]

# Do string in enum checking
class StringMetaEnum(type):
    def __contains__(cls, val):
        try:
            cls(item)
        except ValueError:
            return False
        return True

class OptimizerTypes(Enum, metaclass=StringMetaEnum):
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"

class LRSchedTypes(Enum, metaclass=StringMetaEnum):
    STEP_LR = "steplr"
    EXPONENTIAL_LR = "exponentiallr"

class CriterionTypes(Enum, metaclass=StringMetaEnum):
    CROSS_ENTROPY = "cross_entropy"


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

def validate_optimizer(optim):
    if optim not in OptimizerTypes:
        logging.error(f">> Invalid Optimizer: {optim}")
        return False
    return True

def validate_lrsched(lrsched):
    if lrsched not in LR_SCHEDULERS and lrsched is not None:
        logging.error(f">> Invalid LR Scheduler Type: {lrsched}")
        return False
    return True

def validate_criterion(criterion):
    if criterion not in CriterionTypes:
        logging.error(f">> Invalid criterion: {criterion}")
        
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


@lenient_validate
@dataclass
class ModelConfig:
    name: str = ???

@lenient_validate
@dataclass
class OptimConfig:
    name: str = ???
    loss: str = ???
    lr: float = ???

@lenient_validate
@dataclass
class RunConfig:
    run: bool = ???


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

@lenient_validate
@dataclass
class UsingModelConfig:
    device: str = ???
    md: any = None
    model: type = ModelConfig
    optim: type = OptimConfig


@lenient_validate
@dataclass
class UsingRun:
    seed: int = ???
    epochs: int = ???
    train: RunConfig
    val: RunConfig
