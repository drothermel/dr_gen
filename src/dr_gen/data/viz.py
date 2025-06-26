from typing import Any

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.v2 import functional as f


def plot_first_from_dl(dl):
    feats, labels = next(iter(dl))
    plot([feats[0].squeeze()])


# Copy helper from https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
def plot(imgs, row_title=None, **imshow_kwargs: Any) -> None:  # noqa: ANN401
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            if isinstance(img, tuple):
                img, target = img  # noqa: PLW2901
            img = f.to_image(img)  # noqa: PLW2901
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()  # noqa: PLW2901
                img /= img.max()  # noqa: PLW2901

            img = f.to_dtype(img, torch.uint8, scale=True)  # noqa: PLW2901

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
