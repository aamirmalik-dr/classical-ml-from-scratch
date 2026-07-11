"""Minimal estimator base classes shared by all algorithms in classml.

The interface deliberately mirrors the fit/predict convention popularised by
scikit-learn, but everything here is implemented independently on top of NumPy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def check_array(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    """Validate and convert input to a 2D float64 array.

    Args:
        X: Array-like of shape (n_samples, n_features).
        name: Name used in error messages.

    Returns:
        A contiguous float64 array of shape (n_samples, n_features).

    Raises:
        ValueError: If the input is not 2D or contains non-finite values.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


class Estimator(ABC):
    """Base class for all classml estimators."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Estimator:
        """Fit the estimator to data and return self."""

    def _check_fitted(self, attribute: str) -> None:
        """Raise if the estimator has not been fitted yet.

        Args:
            attribute: Name of an attribute that fit() is expected to set.

        Raises:
            RuntimeError: If the attribute is missing.
        """
        if getattr(self, attribute, None) is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet. Call fit() before using this method."
            )


class Regressor(Estimator):
    """Base class for regressors, adds an R^2 score."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 on the given data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True targets of shape (n_samples,).

        Returns:
            The R^2 score.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot


class Classifier(Estimator):
    """Base class for classifiers, adds an accuracy score."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy on the given data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True labels of shape (n_samples,).

        Returns:
            The accuracy in [0, 1].
        """
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))
