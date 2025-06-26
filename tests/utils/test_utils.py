import json

from dr_gen.analyze.metrics import (
    _flatten_dict_tuple_keys as flatten_dict_tuple_keys,
)


def test_flatten_dict_tuple_keys() -> None:
    # This sample dict is inspired by the "dict_config" in your file example.
    sample_input = {
        "type": "dict_config",
        "value": {
            "paths": {
                "root": "/scratch/ddr8143",
                "proj_dir_name": "cifar10_scratch",
            },
            "data": {
                "name": "cifar10",
                "num_workers": 8,
            },
        },
    }

    # Expected output: keys are tuples of nested keys.
    expected_output = {
        ("type",): "dict_config",
        ("value", "paths", "root"): "/scratch/ddr8143",
        ("value", "paths", "proj_dir_name"): "cifar10_scratch",
        ("value", "data", "name"): "cifar10",
        ("value", "data", "num_workers"): 8,
    }

    flattened = flatten_dict_tuple_keys(sample_input)
    assert flattened == expected_output


def test_flatten_empty_dict() -> None:
    # Test edge-case: an empty dictionary should return an empty dictionary.
    assert flatten_dict_tuple_keys({}) == {}


def test_flatten_single_level_dict() -> None:
    # Test a dictionary that is not nested.
    sample_input = {"a": 1, "b": 2}
    expected_output = {("a",): 1, ("b",): 2}
    assert flatten_dict_tuple_keys(sample_input) == expected_output


# You could also simulate reading one of your file's JSON lines.
def test_flatten_from_json_line() -> None:
    # Suppose this is one line from your file (e.g., line 1)
    json_line = (
        '{"type": "dict_config", "value": {"paths": {"root": "/scratch/ddr8143", '
        '"proj_dir_name": "cifar10_scratch"}}}'
    )
    data = json.loads(json_line)

    expected_output = {
        ("type",): "dict_config",
        ("value", "paths", "root"): "/scratch/ddr8143",
        ("value", "paths", "proj_dir_name"): "cifar10_scratch",
    }

    flattened = flatten_dict_tuple_keys(data)
    assert flattened == expected_output
