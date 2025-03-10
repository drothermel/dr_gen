import hashlib
from datetime import datetime


def make_list(in_val):
    return in_val if isinstance(in_val, list) else [in_val]


def make_list_of_lists(in_val, dim=0):
    in_val = make_list(in_val)
    if isinstance(in_val[0], list):
        return in_val

    if dim == 0:
        return [in_val]
    else:
        return [[iv] for iv in in_val]


def add_dim(inlist, dim=0):
    if dim == 0:
        return [inlist]
    else:
        return [[v] for v in inlist]


def make_list_of_lols(in_val, dim=0):
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
        elif dim == 1:
            return [[vs] for vs in in_val]
        else:
            return [[[v] for v in vs] for vs in in_val]


def flatten_dict_tuple_keys(d, parent_key=()):
    items = {}
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.update(flatten_dict_tuple_keys(v, new_key))
        else:
            items[new_key] = v
    return items


def flatten_dict(in_dict):
    flat_tuple_keys = flatten_dict_tuple_keys(in_dict)
    return {".".join(k): v for k, v in flat_tuple_keys.items()}


def hash_string_to_length(s, length):
    # Encode the string to bytes and compute the SHA-256 hash
    hash_obj = hashlib.sha256(s.encode("utf-8"))
    # Get the hexadecimal digest of the hash
    hex_digest = hash_obj.hexdigest()
    # Return the hash truncated to the specified length
    return hex_digest[:length]


def hash_from_time(length):
    return hash_string_to_length(str(datetime.now()), length)


def dict_to_tupledict(in_dict):
    return tuple(sorted(list(in_dict.items())))


def tupledict_to_dict(in_tupledict):
    return {k: v for k, v in in_tupledict}


def check_prefix_exclude(check_string, excluded_prefixes):
    for pre in excluded_prefixes:
        if check_string.startswith(pre):
            return True
    return False
