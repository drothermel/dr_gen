from dataclasses import MISSING, dataclass
from enum import Enum
import logging

import dr_util.schema_utils as drutil_schemas
from dr_util.schema_utils import lenient_validate

# -------------- Validate Enums --------------

SPLIT_NAMES = ["train", "val", "eval"]
AVAIL_DATASETS = ["cifar10", "cifar100"]


def check_contains(cls, val):
    try:
        cls(val)
    except ValueError:
        return False
    return True


# ---- Implemented Type Enums


class OptimizerTypes(Enum):
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"

    def __contains__(self, val):
        return check_contains(self, val)


class LRSchedTypes(Enum):
    STEP_LR = "steplr"
    EXPONENTIAL_LR = "exponentiallr"

    def __contains__(self, val):
        return check_contains(self, val)


class CriterionTypes(Enum):
    CROSS_ENTROPY = "cross_entropy"

    def __contains__(self, val):
        return check_contains(self, val)


# ---- Val Fxns


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
    if lrsched not in LRSchedTypes and lrsched is not None:
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
    USES_MODEL = "uses_model"
    USES_OPTIM = "uses_optim"
    PERFORMS_RUN = "performs_run"


def get_schema(config_type):
    match config_type:
        case ConfigType.USES_METRICS.value:
            return drutil_schemas.UsingMetricsConfig
        case ConfigType.USES_DATA.value:
            return UsingDataConfig
        case ConfigType.USES_MODEL.value:
            return UsingModelConfig
        case ConfigType.USES_OPTIM.value:
            return UsingOptimConfig
        case ConfigType.PERFORMS_RUN.value:
            return PerformingRun
    return None


#########################################################
#                  Config Definitions
#########################################################


@lenient_validate
@dataclass
class DataConfig:
    name: str = MISSING


@lenient_validate
@dataclass
class ModelConfig:
    name: str = MISSING


@lenient_validate
@dataclass
class OptimConfig:
    name: str = MISSING
    loss: str = MISSING
    lr: float = MISSING


@lenient_validate
@dataclass
class RunConfig:
    run: bool = MISSING


#########################################################
#             Config Interface Definitions
#########################################################


@lenient_validate
@dataclass
class UsingDataConfig:
    data: type = DataConfig


"""
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
"""


@lenient_validate
@dataclass
class UsingModelConfig:
    device: str = MISSING
    md: any = None
    model: type = ModelConfig


@lenient_validate
@dataclass
class UsingOptimConfig:
    device: str = MISSING
    md: any = None
    model: type = ModelConfig
    optim: type = OptimConfig


@lenient_validate
@dataclass
class PerformingRun:
    seed: int = MISSING
    epochs: int = MISSING
    train: RunConfig
    val: RunConfig
