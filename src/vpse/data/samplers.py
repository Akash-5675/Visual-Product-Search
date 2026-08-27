"""PK batch sampler: each batch holds P classes x K images per class.

In-batch mining (batch-hard triplet) only works if every anchor has positives
in the batch — random shuffling over 11k classes almost never provides them.
"""
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler):
    def __init__(self, labels: np.ndarray, p: int, k: int):
        self.p, self.k = p, k
        self.index_by_label = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.index_by_label[int(lab)].append(idx)
        # classes with fewer than 2 images can't form positive pairs
        self.usable = [l for l, idxs in self.index_by_label.items() if len(idxs) >= 2]
        self.batches_per_epoch = len(labels) // (p * k)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch = []
            for lab in random.sample(self.usable, self.p):
                idxs = self.index_by_label[lab]
                replace = len(idxs) < self.k
                batch.extend(np.random.choice(idxs, self.k, replace=replace).tolist())
            yield batch
