"""Two ways to manufacture out-of-catalog queries.

1. Held-out categories (no extra data): drop whole SOP categories from the
   gallery; their images become queries for products the catalog does not
   carry. This is the *hard* case -- they are still products, photographed the
   same way, so the model has every reason to find a near match.

2. Foreign images (e.g. ImageNet): things that are not catalog products at all.
   Note that ImageNet contains toasters, mugs and bicycles too, so this set is
   a mixture of clearly-unrelated and category-overlapping queries.
"""
import numpy as np


def heldout_category_split(super_labels: np.ndarray, holdout: tuple,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (gallery_idx, ood_idx) over the test split."""
    is_ood = np.isin(super_labels, np.asarray(holdout))
    return np.flatnonzero(~is_ood), np.flatnonzero(is_ood)
