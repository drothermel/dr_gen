import random
import numpy as np
import torch

import dr_gen.utils.run as ru


def test_set_deterministic_torch_random_np():
    # Use a fixed seed.
    seed = 42
    generator1 = ru.set_deterministic(seed)
    # Verify deterministic algorithms are enabled.
    assert torch.are_deterministic_algorithms_enabled() is True

    # Generate a tensor using the returned generator.
    x = torch.rand(5, generator=generator1)

    # Reset the seeds by calling set_deterministic again.
    generator2 = ru.set_deterministic(seed)
    y = torch.rand(5, generator=generator2)
    # The two random tensors should be identical.
    assert torch.allclose(x, y)

    # Test that numpy produces identical values after re-seeding.
    np.random.rand()
    # Reset seeds for numpy.
    np.random.seed(seed)
    np.random.rand()
    # After set_deterministic, numpy seed should have been set.
    ru.set_deterministic(seed)
    np_val2 = np.random.rand()
    np.random.seed(seed)
    expected_np2 = np.random.rand()
    # Both values computed after seeding with the same seed should be equal.
    assert np.isclose(np_val2, expected_np2)

    # Test Python's random module.
    random.random()
    random.seed(seed)
    random.random()
    # After set_deterministic, random.seed was called.
    ru.set_deterministic(seed)
    rand_val2 = random.random()
    random.seed(seed)
    expected_rand2 = random.random()
    assert rand_val2 == expected_rand2

def test_seed_worker():
    # Set a known global torch seed.
    torch.manual_seed(123)
    # Compute the worker seed.
    worker_seed = torch.initial_seed() % 2**32

    # Call seed_worker (the worker_id is unused in the function).
    ru.seed_worker(0)

    # Get a random number from NumPy and random.
    np_val = np.random.rand()
    rand_val = random.random()

    # Now, simulate the same seeding to compute expected values.
    np.random.seed(worker_seed)
    expected_np = np.random.rand()
    random.seed(worker_seed)
    expected_rand = random.random()

    # The values produced after seed_worker should match those when re-seeded with worker_seed.
    assert np.isclose(np_val, expected_np)
    assert rand_val == expected_rand

