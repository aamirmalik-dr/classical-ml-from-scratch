"""k-nearest-neighbours classification.

A lazy learner: fit only stores the training set, and predict finds, for each
query point, the k closest training points by Euclidean distance and takes a
majority vote. Ties in the vote are broken toward the class whose nearest member
is closest, which keeps predictions deterministic.

The decision boundary of kNN is a piecewise-linear surface that grows smoother as
k increases, which makes k a natural knob to sweep in a decision-boundary
animation.
"""

from __future__ import annotations

import numpy as np

from .base import Classifier, check_array


def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between every row of A and every row of B."""
    d = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


class KNeighborsClassifier(Classifier):
    """k-nearest-neighbours classifier with uniform or distance weighting.

    Args:
        n_neighbors: Number of neighbours k used for the vote.
        weights: "uniform" for a plain majority vote, or "distance" to weight
            each neighbour by the inverse of its distance.

    Attributes:
        classes_: Sorted unique class labels seen during fit.
        X_: Stored training features of shape (n_samples, n_features).
        y_: Stored training labels of shape (n_samples,).
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform") -> None:
        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.classes_: np.ndarray | None = None
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> KNeighborsClassifier:
        """Store the training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            The fitted estimator.
        """
        if y is None:
            raise ValueError("y is required for KNeighborsClassifier.fit")
        X = check_array(X)
        y = np.asarray(y).ravel()
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples")
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        return self

    def _vote(self, dists: np.ndarray, neigh_idx: np.ndarray) -> np.ndarray:
        """Turn per-query neighbour indices into predicted labels."""
        class_to_int = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([class_to_int[c] for c in self.y_])
        n_classes = len(self.classes_)
        preds = np.empty(neigh_idx.shape[0], dtype=self.classes_.dtype)
        for q in range(neigh_idx.shape[0]):
            idx = neigh_idx[q]
            labels = y_int[idx]
            if self.weights == "uniform":
                weights = np.ones(self.n_neighbors)
            else:
                weights = 1.0 / (np.sqrt(dists[q, idx]) + 1e-12)
            scores = np.bincount(labels, weights=weights, minlength=n_classes)
            preds[q] = self.classes_[int(np.argmax(scores))]
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X by majority vote of the k nearest neighbours.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        self._check_fitted("X_")
        X = check_array(X)
        dists = _pairwise_sq_dists(X, self.X_)
        neigh_idx = np.argpartition(dists, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        return self._vote(dists, neigh_idx)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class probabilities from the neighbour vote fractions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probabilities of shape (n_samples, n_classes), rows summing to one.
        """
        self._check_fitted("X_")
        X = check_array(X)
        class_to_int = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([class_to_int[c] for c in self.y_])
        n_classes = len(self.classes_)
        dists = _pairwise_sq_dists(X, self.X_)
        neigh_idx = np.argpartition(dists, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        proba = np.empty((X.shape[0], n_classes))
        for q in range(X.shape[0]):
            idx = neigh_idx[q]
            labels = y_int[idx]
            if self.weights == "uniform":
                weights = np.ones(self.n_neighbors)
            else:
                weights = 1.0 / (np.sqrt(dists[q, idx]) + 1e-12)
            scores = np.bincount(labels, weights=weights, minlength=n_classes)
            proba[q] = scores / scores.sum()
        return proba
