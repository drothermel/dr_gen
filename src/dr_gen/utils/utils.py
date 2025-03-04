def flatten_dict_tuple_keys(d, parent_key=()):
    items = {}
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items
