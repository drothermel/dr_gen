import json
import pickle

import numpy  # Make sure numpy is installed


def convert_numpy_to_list(item: object) -> object:
    """Recursively converts numpy arrays in a data structure to lists."""
    if isinstance(item, dict):
        return {k: convert_numpy_to_list(v) for k, v in item.items()}
    if isinstance(item, list):
        return [convert_numpy_to_list(i) for i in item]
    if isinstance(item, numpy.ndarray): # Check for numpy array
        return item.tolist()
    return item

def convert_pkl_to_json(pkl_filepath: str, json_filepath: str) -> None:
    """Converts a .pkl file (with potential numpy arrays) to a .json file."""
    try:
        with open(pkl_filepath, "rb") as f_pkl:
            data = pickle.load(f_pkl)

        json_compatible_data = convert_numpy_to_list(data)

        with open(json_filepath, "w") as f_json:
            json.dump(json_compatible_data, f_json, indent=4)
        print(f"Successfully converted '{pkl_filepath}' to '{json_filepath}'")
    except FileNotFoundError:
        print(f"Error: The file '{pkl_filepath}' was not found.")
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle '{pkl_filepath}'. It might be corrupted or not a pickle file.")
    except (OSError, ValueError, TypeError) as e:
        print(f"An error occurred during conversion: {e}")

# Example usage:
# convert_pkl_to_json("your_aggregated_results.pkl", "plotter_data.json")
if __name__ == "__main__":
    convert_pkl_to_json(
        "/Users/daniellerothermel/drotherm/data/dr_gen/run_data_v1/lr_wd_init_v0_t2.pkl",
        "/Users/daniellerothermel/drotherm/data/dr_gen/run_data_v1/lr_wd_init_v0_t2.json",
    )
