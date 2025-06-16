import logging
from dataclasses import MISSING, dataclass, field
from enum import Enum
from typing import Any, cast

import dr_util.schemas as drutil_schemas
from dr_util.schema_utils import lenient_validate

# -------------- Validate Enums --------------

SPLIT_NAMES = ["train", "val", "eval"]
AVAIL_DATASETS = ["cifar10", "cifar100"]


def check_contains(cls: Any, val: Any) -> bool:
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

    def __contains__(self, val: Any) -> bool:
        return check_contains(self, val)


class LRSchedTypes(Enum):
    STEP_LR = "steplr"
    EXPONENTIAL_LR = "exponentiallr"

    def __contains__(self, val: Any) -> bool:
        return check_contains(self, val)


class CriterionTypes(Enum):
    CROSS_ENTROPY = "cross_entropy"

    def __contains__(self, val: Any) -> bool:
        return check_contains(self, val)


# ---- Val Fxns


def validate_split(split: str) -> bool:
    if split not in SPLIT_NAMES:
        logging.error(f"Split {split} should be in {SPLIT_NAMES}")
        return False
    return True


def validate_dataset(dataset: str) -> bool:
    if dataset not in AVAIL_DATASETS:
        logging.error(f"Dataset {dataset} should be in {AVAIL_DATASETS}")
        return False
    return True


def validate_optimizer(optim: str) -> bool:
    if optim not in OptimizerTypes:
        logging.error(f">> Invalid Optimizer: {optim}")
        return False
    return True


def validate_lrsched(lrsched: str | None) -> bool:
    if lrsched not in LRSchedTypes and lrsched is not None:
        logging.error(f">> Invalid LR Scheduler Type: {lrsched}")
        return False
    return True


def validate_criterion(criterion: str) -> bool:
    if criterion not in CriterionTypes:
        logging.error(f">> Invalid criterion: {criterion}")
        return False
    return True


# -------------- Validate Configs --------------


class ConfigType(Enum):
    USES_METRICS = "uses_metrics"
    USES_DATA = "uses_data"
    USES_MODEL = "uses_model"
    USES_OPTIM = "uses_optim"
    PERFORMS_RUN = "performs_run"


def get_schema(config_type: str) -> type | None:
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
    name: str = field(default=cast("str", MISSING))


@lenient_validate
@dataclass
class ModelConfig:
    name: str = field(default=cast("str", MISSING))


@lenient_validate
@dataclass
class OptimConfig:
    name: str = field(default=cast("str", MISSING))
    loss: str = field(default=cast("str", MISSING))
    lr: float = field(default=cast("float", MISSING))


@lenient_validate
@dataclass
class RunConfig:
    run: bool = field(default=cast("bool", MISSING))


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
    device: str = field(default=cast("str", MISSING))
    model: type = ModelConfig


@lenient_validate
@dataclass
class UsingOptimConfig:
    device: str = field(default=cast("str", MISSING))
    model: type = ModelConfig
    optim: type = OptimConfig


@lenient_validate
@dataclass
class PerformingRun:
    train: RunConfig
    val: RunConfig
    seed: int = field(default=cast("int", MISSING))
    epochs: int = field(default=cast("int", MISSING))
