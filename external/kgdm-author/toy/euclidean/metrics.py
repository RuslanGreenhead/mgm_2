from typing import Optional, Union

import numpy as np
import torch
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def c2st(
    x: ArrayLike,
    y: ArrayLike,
    n_folds: int = 5,
    hidden_sizes: tuple = (64, 64),
    max_iter: int = 500,
    seed: Optional[int] = None,
) -> float:
    """Classifier two-sample test accuracy. 0.5 ≈ indistinguishable, 1.0 = separable."""
    x = _to_numpy(x)
    y = _to_numpy(y)
    z = np.concatenate([x, y], axis=0)
    labels = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        max_iter=max_iter,
        random_state=seed,
    )
    scores = cross_val_score(clf, z, labels, cv=n_folds)
    return float(scores.mean())
