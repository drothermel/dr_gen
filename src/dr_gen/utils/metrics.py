from enum import Enum
from functools import singledispatchmethod

from dr_util.metrics import (
    BATCH_KEY,
    Metrics,
    MetricsSubgroup,
    add_sum,
    add_list,
    agg_passthrough,
    agg_none,
    agg_batch_weighted_list_avg,
    create_logger,
)

from dr_gen.schemas import SPLIT_NAMES


class GenMetricType(Enum):
    INT = "int"
    LIST = "list"
    BATCH_WEIGHTED_AVG_LIST = "batch_weighted_avg_list"
    AVG_LIST = "avg_list"


def agg_avg_list(data, key):
    data_sum = sum(data[key])
    return data_sum * 1.0 / len(data[key])


class GenMetricsSubgroup(MetricsSubgroup):
    def _init_data(self):
        self._init_data_values()
        self._init_data_fxns()

    def _init_data_values(self):
        if self.data_structure is None:
            return

        for key, data_type in self.data_structure.items():
            match data_type:
                case GenMetricType.INT.value:
                    self.data[key] = 0
                case GenMetricType.LIST.value:
                    self.data[key] = []
                case GenMetricType.BATCH_WEIGHTED_AVG_LIST.value:
                    self.data[key] = []
                case GenMetricType.AVG_LIST.value:
                    self.data[key] = []

    def _init_data_fxns(self):
        if self.data_structure is None:
            return

        for key, data_type in self.data_structure.items():
            match data_type:
                case GenMetricType.INT.value:
                    self.add_fxns[key] = add_sum
                    self.agg_fxns[key] = agg_passthrough
                case GenMetricType.LIST.value:
                    self.add_fxns[key] = add_list
                    self.agg_fxns[key] = agg_none
                case GenMetricType.BATCH_WEIGHTED_AVG_LIST.value:
                    self.add_fxns[key] = add_list
                    self.agg_fxns[key] = agg_batch_weighted_list_avg
                case GenMetricType.AVG_LIST.value:
                    self.add_fxns[key] = add_list
                    self.agg_fxns[key] = agg_avg_list

    def clear_data(self):
        self._init_data_values()

    ## Override these to handle batch size better ##
    @singledispatchmethod
    def add(self, data, ns=None):  # noqa: ARG002 (unused args)
        assert False, f">> Unexpected data type: {type(data)}"  # noqa

    @add.register(tuple)
    def _(self, data, ns=None):
        assert len(data) == len(("key", "val"))
        self._add_tuple(*data)
        if ns is not None:
            self._add_tuple(BATCH_KEY, ns)

    @add.register(dict)
    def _(self, data, ns=None):
        for key, val in data.items():
            self._add_tuple(key, val)
        if ns is not None:
            self._add_tuple(BATCH_KEY, ns)


class GenMetrics(Metrics):
    def __init__(self, cfg):
        self.cfg = cfg
        self.group_names = [
            s for s in SPLIT_NAMES if cfg.get(s, {}).get('run', False)
        ]

        # Initialize subgroups and loggers
        self.groups = {name: GenMetricsSubgroup(cfg, name) for name in self.group_names}
        self.loggers = [create_logger(cfg, lt) for lt in cfg.metrics.loggers]

    def log_data(self, data, group_name, ns=None):
        if group_name not in self.groups:
            assert False, f">> Invalid group name: {group_name}"
        self.groups[group_name].add(data, ns=ns)

    def clear_data(self, group_name=None):
        for gname, group in self.groups.items():
            if group_name == gname or group_name is None:
                group.clear_data()

    def agg_log(self, data_name):
        try:
            super().agg_log(data_name)
        except:
            self.log(f">> Aggregation failed for {data_name}")
