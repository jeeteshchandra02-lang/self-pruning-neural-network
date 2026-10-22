"""Only scikit-learn call in this project: load a standard dataset."""
from __future__ import annotations

import numpy as np


def load_data(split_seed: int = 2026) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    # sklearn is used ONLY to load_digits; all splitting, normalization,
    # shuffling, training, accuracy and gradient logic is handwritten.
    from sklearn.datasets import load_digits
    x, y = load_digits(return_X_y=True)
    x = np.asarray(x, dtype=np.float64) / 16.0  # known pixel scale (0..16)
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    train, valid, test = [], [], []
    for cls in np.unique(y):
        indices = np.flatnonzero(y == cls)
        indices = rng.permutation(indices)
        n = len(indices)
        n_train = int(n * 0.70)
        n_valid = int(n * 0.15)
        train.extend(indices[:n_train])
        valid.extend(indices[n_train:n_train + n_valid])
        test.extend(indices[n_train + n_valid:])
    return tuple((x[np.sort(ids)], y[np.sort(ids)]) for ids in (train, valid, test))
