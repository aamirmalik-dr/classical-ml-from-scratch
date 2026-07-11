"""K-means clustering with k-means++ initialisation and restarts.

Lloyd's algorithm alternates two steps until cluster labels stop changing:

1. Labeling: attach each point to its nearest centroid (squared Euclidean).
2. Update: move each centroid to the mean of its assigned points.

The objective (inertia) is the sum of squared distances of samples to their
closest centroid. Because Lloyd's algorithm only finds a local optimum, the fit
is restarted n_init times from different k-means++ seeds and the best run wins.
"""

from __future__ import annotations

import numpy as np

from .base import Estimator, check_array


def _pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of X and rows of C."""
    d = (X**2).sum(axis=1)[:, None] + (C**2).sum(axis=1)[None, :] - 2.0 * X @ C.T
    return np.maximum(d, 0.0)


class KMeans(Estimator):
    """K-means clustering.

    Args:
        n_clusters: Number of clusters k.
        n_init: Number of independent restarts; the run with the lowest
            inertia is kept.
        max_iter: Maximum Lloyd iterations per run.
        tol: Convergence threshold on the change in inertia.
        random_state: Seed for reproducible initialisation.

    Attributes:
        cluster_centers_: Final centroids of shape (n_clusters, n_features).
        labels_: Cluster index of each training sample.
        inertia_: Sum of squared distances to the closest centroid.
        n_iter_: Iterations run by the best restart.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++ seeding: spread initial centroids proportionally to distance."""
        n_samples = X.shape[0]
        centers = np.empty((self.n_clusters, X.shape[1]))
        centers[0] = X[rng.integers(n_samples)]
        closest_sq = _pairwise_sq_dists(X, centers[:1]).ravel()
        for k in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total <= 0:
                centers[k] = X[rng.integers(n_samples)]
            else:
                probs = closest_sq / total
                centers[k] = X[rng.choice(n_samples, p=probs)]
            closest_sq = np.minimum(closest_sq, _pairwise_sq_dists(X, centers[k : k + 1]).ravel())
        return centers

    def _run_once(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """One Lloyd's-algorithm run, returns (centers, labels, inertia, n_iter)."""
        centers = self._init_centroids(X, rng)
        prev_inertia = np.inf
        labels = np.zeros(X.shape[0], dtype=int)
        for it in range(1, self.max_iter + 1):
            dists = _pairwise_sq_dists(X, centers)
            labels = np.argmin(dists, axis=1)
            inertia = float(dists[np.arange(X.shape[0]), labels].sum())
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    centers[k] = X[mask].mean(axis=0)
                else:
                    # Re-seed an empty cluster at the point farthest from its centroid.
                    centers[k] = X[np.argmax(dists[np.arange(X.shape[0]), labels])]
            if prev_inertia - inertia < self.tol:
                return centers, labels, inertia, it
            prev_inertia = inertia
        return centers, labels, prev_inertia, self.max_iter

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> KMeans:
        """Cluster the data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Ignored, present for interface consistency.

        Returns:
            The fitted estimator.
        """
        X = check_array(X)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")
        rng = np.random.default_rng(self.random_state)
        best: tuple[np.ndarray, np.ndarray, float, int] | None = None
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._run_once(X, rng)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia, n_iter)
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample in X to its nearest learned centroid.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Cluster indices of shape (n_samples,).
        """
        self._check_fitted("cluster_centers_")
        X = check_array(X)
        return np.argmin(_pairwise_sq_dists(X, self.cluster_centers_), axis=1)
