import hashlib
from datetime import UTC, datetime
from typing import Any


def make_list(in_val: Any) -> list[Any]:  # noqa: ANN401
    return in_val if isinstance(in_val, list) else [in_val]


def make_list_of_lists(in_val: Any, dim: int = 0) -> list[Any]:  # noqa: ANN401
    in_val = make_list(in_val)
    if isinstance(in_val[0], list):
        return in_val  # type: ignore[no-any-return]

    if dim == 0:
        return [in_val]
    return [[iv] for iv in in_val]


def add_dim(inlist: list[Any], dim: int = 0) -> list[Any]:
    if dim == 0:
        return [inlist]
    return [[v] for v in inlist]


def make_list_of_lols(in_val: Any, dim: int = 0) -> list[Any]:  # noqa: ANN401
    in_val = make_list(in_val)
    if isinstance(in_val[0], list) and isinstance(in_val[0][0], list):
        return in_val

    # If its just a list
    if not isinstance(in_val[0], list):
        if dim == 0:
            return [[in_val]]
        if dim == 1:
            return [[[v] for v in in_val]]
        return [[[v]] for v in in_val]

    # If its just a list of lists
    if not isinstance(in_val[0][0], list):
        if dim == 0:
            return [in_val]
        if dim == 1:
            return [[vs] for vs in in_val]
        return [[[v] for v in vs] for vs in in_val]

    # Default fallback (should not reach here in normal cases)
    return in_val  # type: ignore[no-any-return]


def flatten_dict_tuple_keys(
    d: dict[Any, Any], parent_key: tuple[Any, ...] = ()
) -> dict[tuple[Any, ...], Any]:
    items = {}
    for k, v in d.items():
        new_key = (*parent_key, k)
        if isinstance(v, dict):
            items.update(flatten_dict_tuple_keys(v, new_key))
        else:
            items[new_key] = v
    return items


def flatten_dict(in_dict: dict[Any, Any]) -> dict[str, Any]:
    flat_tuple_keys = flatten_dict_tuple_keys(in_dict)
    return {".".join(k): v for k, v in flat_tuple_keys.items()}


def hash_string_to_length(s: str, length: int) -> str:
    # Encode the string to bytes and compute the SHA-256 hash
    hash_obj = hashlib.sha256(s.encode("utf-8"))
    # Get the hexadecimal digest of the hash
    hex_digest = hash_obj.hexdigest()
    # Return the hash truncated to the specified length
    return hex_digest[:length]


def hash_from_time(length: int) -> str:
    return hash_string_to_length(str(datetime.now(UTC)), length)


def dict_to_tupledict(in_dict: dict[Any, Any]) -> tuple[tuple[Any, Any], ...]:
    return tuple(sorted(in_dict.items()))


def tupledict_to_dict(in_tupledict: tuple[tuple[Any, Any], ...]) -> dict[Any, Any]:
    return dict(in_tupledict)


def check_prefix_exclude(check_string: str, excluded_prefixes: list[str]) -> bool:
    return any(check_string.startswith(pre) for pre in excluded_prefixes)
