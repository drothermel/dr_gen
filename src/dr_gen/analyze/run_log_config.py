from dr_gen.utils.utils import flatten_dict_tuple_keys

class RunLogConfig:
    def __init__(
        self,
        cfg_json, # raw line from log file
        remap_keys={},
        epochs_key="epochs",
    ):
        self.raw_cfg_json = cfg_json
        self.remap_keys = remap_keys
        self.epochs_key = epochs_key

        self.cfg = None # dict
        self.parse_errors = []
        self.parse_cfg_str()


    @property
    def flat_cfg(self):
        flat_cfg = flatten_dict_tuple_keys(self.cfg)
        return {".".join(k): v for k, v in flat_cfg.items()}

    def parse_cfg_str(self):
        if self.raw_cfg_json.get("type", None) != "dict_config":
            self.parse_errors.append(">> Config json doesn't have {type: dict_config}")
        if "value" not in self.raw_cfg_json:
            self.parse_errors.append(">> Config 'value' not set")
        elif not isinstance(self.raw_cfg_json['value'], dict):
            self.parse_errors.append(">> Config type isn't dict")
        elif len(self.raw_cfg_json['value']) == 0:
            self.parse_errors.append(">> The config is empty")
        if len(self.parse_errors) > 0:
            return
        self.cfg = self.raw_cfg_json['value']

    def get_sweep_cfg(
        keys,
        pretty=True,
    ):
        sweep_cfg = {}
        for k, v in self.flat_cfg:
            if k not in keys:
                continue
            if remap_keys:
                sweep_cfg[remap_keys.get(k, k)] = v
        return sweep_cfg
