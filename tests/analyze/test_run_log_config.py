import pytest
from dr_gen.analyze.run_log_config import RunLogConfig 

# A valid config similar to your example
VALID_CONFIG = {
    "type": "dict_config",
    "value": {
        "paths": {
            "root": "/scratch/ddr8143",
            "proj_dir_name": "cifar10_scratch"
        },
        "data": {
            "name": "cifar10",
            "num_workers": 8,
        },
        "epochs": 270
    }
}

def test_valid_config_parsing():
    config = RunLogConfig(VALID_CONFIG)
    # No errors should have been reported.
    assert config.parse_errors == []
    # cfg should be set to the "value" part.
    assert config.cfg == VALID_CONFIG["value"]

    # Test flat_cfg property: it should flatten nested keys with dot notation.
    flat = config.flat_cfg
    # Expected keys: "paths.root", "paths.proj_dir_name", "data.name", "data.num_workers", "epochs"
    expected_keys = {"paths.root", "paths.proj_dir_name", "data.name", "data.num_workers", "epochs"}
    assert expected_keys.issubset(flat.keys())

def test_get_sweep_cfg_with_remap():
    # Use a remap so that a specific key is renamed in the output.
    remap = {"data.name": "dataset"}
    config = RunLogConfig(VALID_CONFIG, remap_keys=remap)
    # Let's say we want only keys related to "data" (and "epochs")
    keys = {"data.name", "data.num_workers", "epochs"}
    sweep_cfg = config.get_sweep_cfg(keys)
    
    # Check that the remapped key for "data.name" is used.
    assert "dataset" in sweep_cfg
    assert sweep_cfg["dataset"] == VALID_CONFIG["value"]["data"]["name"]
    # And check other keys remain as-is.
    assert sweep_cfg.get("data.num_workers") == VALID_CONFIG["value"]["data"]["num_workers"]
    assert sweep_cfg.get("epochs") == VALID_CONFIG["value"]["epochs"]

def test_error_invalid_type():
    # Missing the proper type ("dict_config")
    bad_config = {
        "type": "str",
        "value": VALID_CONFIG["value"]
    }
    cfg_instance = RunLogConfig(bad_config)
    # Expect an error message about the type.
    assert ">> Config json doesn't have {type: dict_config}" in cfg_instance.parse_errors
    # Because errors exist, self.cfg should remain None.
    assert cfg_instance.cfg is None

def test_error_missing_value():
    # Config without a "value" key.
    bad_config = {"type": "dict_config"}
    cfg_instance = RunLogConfig(bad_config)
    assert ">> Config 'value' not set" in cfg_instance.parse_errors
    assert cfg_instance.cfg is None

def test_error_non_dict_value():
    # Config where "value" is not a dict.
    bad_config = {"type": "dict_config", "value": "this should be a dict"}
    cfg_instance = RunLogConfig(bad_config)
    assert ">> Config type isn't dict" in cfg_instance.parse_errors
    assert cfg_instance.cfg is None

def test_error_empty_value():
    # Config where "value" is an empty dict.
    bad_config = {"type": "dict_config", "value": {}}
    cfg_instance = RunLogConfig(bad_config)
    assert ">> The config is empty" in cfg_instance.parse_errors
    assert cfg_instance.cfg is None

