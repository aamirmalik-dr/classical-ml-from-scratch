"""Principal component analysis via singular value decomposition.

Center the data, take the SVD X_c = U S V^T, and use the rows of V^T as the
principal axes. The explained variance of component i is s_i^2 / (n - 1). SVD is
preferred over an eigendecomposition of the covariance matrix because it avoids
forming X^T X and squaring the condition number.
"""

from __future__ import annotations

import numpy as np

from .base import Estimator, check_array


class PCA(Estimator):
    """Principal component analysis.

    Args:
        n_components: Number of components to keep. None keeps all.

    Attributes:
        components_: Principal axes of shape (n_components, n_features).
        explained_variance_: Variance explained by each component.
        explained_variance_ratio_: Fraction of total variance per component.
        singular_values_: Singular values of the centered data.
        mean_: Per-feature mean of the training data.
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> PCA:
        """Learn the principal axes of X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Ignored, present for interface consistency.

        Returns:
            The fitted estimator.
        """
        X = check_array(X)
        n_samples = X.shape[0]
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # Sign convention: make the largest absolute entry of each axis positive
        # so results are deterministic across SVD implementations.
        signs = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
        Vt *= signs[:, None]

        k = self.n_components if self.n_components is not None else Vt.shape[0]
        full_variance = (S**2) / (n_samples - 1)
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.explained_variance_ = full_variance[:k]
        self.explained_variance_ratio_ = full_variance[:k] / full_variance.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the principal axes.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Projected data of shape (n_samples, n_components).
        """
        self._check_fitted("components_")
        X = check_array(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit the model and return the projection of X."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map projected data back to the original feature space.

        Args:
            Z: Projected data of shape (n_samples, n_components).

        Returns:
            Reconstruction of shape (n_samples, n_features).
        """
        self._check_fitted("components_")
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_ + self.mean_
