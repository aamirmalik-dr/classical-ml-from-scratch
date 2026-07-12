"""Extend the library: a nearest-centroid classifier on the shared base class.

The base classes in ``classml.base`` are the whole contract. Subclass
``Classifier``, implement ``fit`` (returning self) and ``predict``, and you
inherit ``score`` and the fitted-state check for free, plus you plug straight
into the gallery and benchmark tooling.

    python examples/custom_estimator.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from classml.base import Classifier, check_array

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_moons.csv"


class NearestCentroid(Classifier):
    """Assign each point to the class whose training-mean is closest.

    Attributes:
        classes_: Sorted unique class labels seen during fit.
        centroids_: Per-class feature means of shape (n_classes, n_features).
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> NearestCentroid:
        """Store one centroid per class."""
        if y is None:
            raise ValueError("y is required")
        X = check_array(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the label of the nearest class centroid."""
        self._check_fitted("centroids_")
        X = check_array(X)
        dists = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[np.argmin(dists, axis=1)]


def main() -> None:
    """Fit the custom estimator on the committed sample and score it."""
    table = np.loadtxt(SAMPLE, delimiter=",", skiprows=1)
    X, y = table[:, :2], table[:, 2].astype(int)
    model = NearestCentroid().fit(X, y)
    print(f"nearest-centroid training accuracy: {model.score(X, y):.4f}")


if __name__ == "__main__":
    main()
