import pytest
import torch

import dr_gen.train.evaluate as eu


def test_accuracy_top1_top5() -> None:
    """Test accuracy() with topk=(1,5) on a batch of 4 samples.

    Sample breakdown:
      - Sample 0: highest score at index 3, target=3 → correct for both top1 and top5.
      - Sample 1: highest score at index 0, but target=5 appears in top5 (second highest) → incorrect top1, correct top5.
      - Sample 2: highest score at index 8, target=8 → correct for both.
      - Sample 3: highest score at index 0, target=9 not in top5 → incorrect.

    Expected:
      - Top1 accuracy: 2 correct / 4 samples = 50%
      - Top5 accuracy: 3 correct / 4 samples = 75%
    """
    output = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05],  # sample 0
            [0.9, 0.1, 0.2, 0.3, 0.4, 0.8, 0.5, 0.6, 0.7, 0.05],  # sample 1
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1],  # sample 2
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],  # sample 3
        ]
    )
    target = torch.tensor([3, 5, 8, 9])
    top1, top5 = eu.accuracy(output, target, topk=(1, 5))
    assert top1 == pytest.approx(50.0)
    assert top5 == pytest.approx(75.0)


def test_accuracy_top1_only() -> None:
    """Test accuracy() with only topk=(1,).

    All predictions are arranged to be correct.
    """
    output = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.1, 0.0],
            [0.2, 0.1, 0.7],
        ]
    )
    target = torch.tensor([1, 0, 2])
    [top1] = eu.accuracy(output, target, topk=(1,))
    # All predictions are correct so accuracy should be 100%
    assert top1 == pytest.approx(100.0)


def test_accuracy_with_onehot_target() -> None:
    """Test accuracy() when the target is a one-hot encoded 2D tensor.

    The function should extract the index of the maximum value along dim=1.

    Sample breakdown:
      - Sample 0: one-hot target corresponds to label 2; highest score at index 2 → correct.
      - Sample 1: one-hot target corresponds to label 0; highest score is at index 1 → incorrect.

    Expected top1 accuracy: 50%
    """
    output = torch.tensor(
        [
            [0.1, 0.3, 0.6, 0.0],  # sample 0: highest at index 2
            [0.4, 0.5, 0.0, 0.1],  # sample 1: highest at index 1
        ]
    )
    # Create one-hot encoded targets.
    target = torch.tensor(
        [
            [0, 0, 1, 0],  # corresponds to label 2
            [1, 0, 0, 0],  # corresponds to label 0
        ],
        dtype=torch.float32,
    )
    [top1] = eu.accuracy(output, target, topk=(1,))
    assert top1 == pytest.approx(50.0)
