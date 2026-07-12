"""Gaussian naive Bayes classifier.

Each class is modelled as a product of per-feature Gaussians (the naive
conditional-independence assumption). Fitting is a single pass that estimates a
per-class prior, mean, and variance. Prediction picks the class with the largest
log posterior, computed in log space to avoid underflow.

    log p(y=c | x) = log pi_c + sum_j log N(x_j | mu_cj, sigma^2_cj) + const

The resulting decision boundary between two classes is quadratic in x, which sits
between the linear boundary of logistic regression and the flexible boundary of
kNN or an RBF SVM.
"""

from __future__ import annotations

import numpy as np

from .base import Classifier, check_array


class GaussianNB(Classifier):
    """Gaussian naive Bayes classifier.

    Args:
        var_smoothing: Fraction of the largest feature variance added to every
            variance for numerical stability, matching the scikit-learn default.

    Attributes:
        classes_: Sorted unique class labels seen during fit.
        priors_: Class prior probabilities of shape (n_classes,).
        theta_: Per-class feature means of shape (n_classes, n_features).
        var_: Per-class feature variances of shape (n_classes, n_features).
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> GaussianNB:
        """Estimate per-class priors, means, and variances.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            The fitted estimator.
        """
        if y is None:
            raise ValueError("y is required for GaussianNB.fit")
        X = check_array(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.theta_ = np.empty((n_classes, n_features))
        self.var_ = np.empty((n_classes, n_features))
        self.priors_ = np.empty(n_classes)
        epsilon = self.var_smoothing * X.var(axis=0).max()
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + epsilon
            self.priors_[i] = Xc.shape[0] / X.shape[0]
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Unnormalised log posterior of every sample under every class."""
        jll = np.empty((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            log_prior = np.log(self.priors_[i])
            term = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i]))
            term = term - 0.5 * np.sum(((X - self.theta_[i]) ** 2) / self.var_[i], axis=1)
            jll[:, i] = log_prior + term
        return jll

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        self._check_fitted("theta_")
        X = check_array(X)
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via a softmax over the log posteriors.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probabilities of shape (n_samples, n_classes), rows summing to one.
        """
        self._check_fitted("theta_")
        X = check_array(X)
        jll = self._joint_log_likelihood(X)
        m = jll.max(axis=1, keepdims=True)
        exp = np.exp(jll - m)
        return exp / exp.sum(axis=1, keepdims=True)
