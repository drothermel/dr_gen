import hashlib
from datetime import datetime

def flatten_dict_tuple_keys(d, parent_key=()):
    items = {}
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.update(flatten_dict_tuple_keys(v, new_key))
        else:
            items[new_key] = v
    return items

def hash_string_to_length(s, length):
    # Encode the string to bytes and compute the SHA-256 hash
    hash_obj = hashlib.sha256(s.encode("utf-8"))
    # Get the hexadecimal digest of the hash
    hex_digest = hash_obj.hexdigest()
    # Return the hash truncated to the specified length
    return hex_digest[:length]

def hash_from_time(length):
    return hash_string_to_length(str(datetime.now()), length)

