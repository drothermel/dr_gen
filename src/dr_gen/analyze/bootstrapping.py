import numpy as np

def bootstrap_samples_batched(dataset, num_bootstraps=None):
    """
    Generate bootstrap samples from a batched dataset.

    Performs bootstrapping with replacement independently for each
    row along the first dimension (B).

    Args:
        dataset (np.ndarray): Input data of shape (B, N), where H is the
                              batch dimension and N is the feature/sample dimension.
        b (int, optional): Number of bootstrap samples to generate for each row (B).
                           If None, it returns the original dataset expanded
                           to shape (B, 1, N), representing one sample (the original)
                           for each row. Defaults to None.

    Returns:
        np.ndarray: Array of bootstrap samples.
                    Shape is (B, b, N) if 'b' is a positive integer.
                    Shape is (B, 1, N) if 'b' is None.
    """
    assert isinstance(dataset, np.ndarray)
    if dataset.ndim == 1:
        dataset = dataset[np.newaxis, :]

    B, N = dataset.shape # B = batch size, N = number of features/samples per batch
    if num_bootstraps is None:
        return np.expand_dims(dataset, axis=1) # Shape becomes (B, 1, N)

    # Select each of the batches (B)
    row_indices = np.arange(B)[:, np.newaxis, np.newaxis]          # Shape (B, 1, 1)
    # Select num_bootstraps (b) random indices from each batch
    indices = np.random.randint(0, N, size=(B, num_bootstraps, N)) # Shape (B, b, N)

    # Perform the selection using advanced indexing.
    #    `dataset[row_indices, indices]` uses the broadcasted `row_indices`
    #    to select the correct row (h) and the `indices` array to select
    #    the correct column index within that row for each element in the
    #    (B, b, N) output shape.
    bootstrap_result = dataset[row_indices, indices] # Shape (B, b, N)
    return bootstrap_result

if __name__ == "__main__":
    # Parameters
    H = 3  # Batch size
    N = 5  # Number of features/data points per batch element
    B = 10 # Number of bootstrap samples per batch element

    # Create a sample dataset (H x N)
    # Using arange for easy visual verification
    data = np.arange(H * N).reshape(H, N)
    print("Original Dataset (HxN):")
    print(data)
    print(f"Shape: {data.shape}\n")

    # --- Test Case 1: Generate B bootstrap samples ---
    try:
        bootstrapped_data = bootstrap_samples_batched(data, num_bootstraps=B)
        print(f"Bootstrapped Dataset (HxBxN) with B={B}:")
        # Printing the whole thing might be large, show first few samples per batch
        for h in range(H):
            print(f"  Batch {h} (first 3 samples):")
            print(bootstrapped_data[h, :3, :])
        print(f"Shape: {bootstrapped_data.shape}\n")

        # Verification for the first batch element (h=0)
        print("Verification for first batch element (h=0):")
        original_row_0 = data[0, :]
        bootstrapped_batch_0 = bootstrapped_data[0, :, :]
        print(f"  Original row (h=0): {original_row_0}")
        # Check if all elements in the bootstrapped samples for h=0 came from the original row 0
        is_subset = np.all(np.isin(bootstrapped_batch_0, original_row_0))
        print(f"  Are all elements in bootstrapped[0,:,:] from original data[0,:]? {is_subset}")
        # Check if elements were repeated (expected with replacement)
        has_repeats = any(np.unique(row).size < N for row in bootstrapped_batch_0)
        print(f"  Do samples for h=0 show repeats (expected)? {has_repeats}\n")

    except ValueError as e:
        print(f"Error during bootstrapping: {e}")


    # --- Test Case 2: b = None ---
    try:
        bootstrapped_none = bootstrap_samples_batched(data, num_bootstraps=None)
        print(f"Bootstrapped Dataset (Hx1xN) with b=None:")
        print(bootstrapped_none)
        print(f"Shape: {bootstrapped_none.shape}")
        # Verify it's the same as manually expanding dimensions
        is_equal = np.array_equal(bootstrapped_none, np.expand_dims(data, axis=1))
        print(f"Is it equal to data expanded at axis 1? {is_equal}\n")
    except ValueError as e:
        print(f"Error during bootstrapping (b=None): {e}")


    # --- Test Case 3: Invalid Input - 1D array ---
    try:
        print("Testing invalid input (1D array):")
        bootstrap_samples_batched(np.arange(10), num_bootstraps=5)
    except ValueError as e:
        print(f"  Caught expected error: {e}\n")

    # --- Test Case 4: Invalid Input - b=0 ---
    try:
        print("Testing invalid input (b=0):")
        bootstrap_samples_batched(data, num_bootstraps=0)
    except ValueError as e:
        print(f"  Caught expected error: {e}\n")

    # --- Test Case 5: Invalid Input - b=-1 ---
    try:
        print("Testing invalid input (b=-1):")
        bootstrap_samples_batched(data, num_bootstraps=-1)
    except ValueError as e:
        print(f"  Caught expected error: {e}\n")

