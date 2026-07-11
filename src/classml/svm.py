"""Kernel support vector machine trained with a simplified SMO solver.

The soft-margin SVM dual problem is

    max_a  sum_i a_i - (1/2) sum_ij a_i a_j y_i y_j K(x_i, x_j)
    s.t.   0 <= a_i <= C,  sum_i a_i y_i = 0

Sequential minimal optimisation picks pairs of multipliers (a_i, a_j) and solves
the two-variable subproblem analytically, clipping to the box constraints. This
implementation follows the simplified SMO scheme: iterate over samples whose KKT
conditions are violated, pair each with a random second sample, and stop after a
number of consecutive passes without any update.
"""

from __future__ import annotations

import numpy as np

from .base import Classifier, check_array


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """RBF (Gaussian) kernel matrix K[i, j] = exp(-gamma ||a_i - b_j||^2).

    Args:
        A: Array of shape (n, d).
        B: Array of shape (m, d).
        gamma: Kernel width parameter, must be positive.

    Returns:
        Kernel matrix of shape (n, m).
    """
    sq = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


class KernelSVM(Classifier):
    """Binary soft-margin SVM with an RBF kernel, solved by simplified SMO.

    Args:
        C: Soft-margin penalty. Larger C fits the training data harder.
        gamma: RBF kernel width. "scale" uses 1 / (n_features * Var(X)).
        tol: KKT violation tolerance.
        max_passes: Consecutive full passes without updates before stopping.
        max_iter: Hard cap on the number of full passes.
        random_state: Seed for the random pairing of multipliers.

    Attributes:
        classes_: The two original class labels, mapped internally to -1/+1.
        support_: Indices of training points with nonzero multipliers.
        alpha_: Nonzero dual coefficients at the support vectors.
        support_vectors_: Training points with nonzero multipliers.
        support_labels_: The -1/+1 labels of the support vectors.
        b_: Bias term of the decision function.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: float | str = "scale",
        tol: float = 1e-3,
        max_passes: int = 5,
        max_iter: int = 200,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.support_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.support_vectors_: np.ndarray | None = None
        self.support_labels_: np.ndarray | None = None
        self.b_: float = 0.0
        self.gamma_: float = 0.0

    def _resolve_gamma(self, X: np.ndarray) -> float:
        """Turn the gamma parameter into a concrete positive float."""
        if self.gamma == "scale":
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        return float(self.gamma)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> KernelSVM:
        """Train the SVM on a binary problem.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels with exactly two distinct values.

        Returns:
            The fitted estimator.
        """
        if y is None:
            raise ValueError("y is required for KernelSVM.fit")
        X = check_array(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("KernelSVM supports exactly two classes")
        t = np.where(y == self.classes_[1], 1.0, -1.0)

        self.gamma_ = self._resolve_gamma(X)
        K = rbf_kernel(X, X, self.gamma_)
        n = X.shape[0]
        alpha = np.zeros(n)
        b = 0.0
        rng = np.random.default_rng(self.random_state)

        def f(i: int) -> float:
            return float((alpha * t) @ K[:, i] + b)

        passes = 0
        it = 0
        while passes < self.max_passes and it < self.max_iter:
            it += 1
            num_changed = 0
            for i in range(n):
                E_i = f(i) - t[i]
                if (t[i] * E_i < -self.tol and alpha[i] < self.C) or (
                    t[i] * E_i > self.tol and alpha[i] > 0
                ):
                    j = int(rng.integers(n - 1))
                    if j >= i:
                        j += 1
                    E_j = f(j) - t[j]
                    a_i_old, a_j_old = alpha[i], alpha[j]
                    if t[i] != t[j]:
                        L = max(0.0, a_j_old - a_i_old)
                        H = min(self.C, self.C + a_j_old - a_i_old)
                    else:
                        L = max(0.0, a_i_old + a_j_old - self.C)
                        H = min(self.C, a_i_old + a_j_old)
                    if L >= H:
                        continue
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    a_j = a_j_old - t[j] * (E_i - E_j) / eta
                    a_j = min(max(a_j, L), H)
                    if abs(a_j - a_j_old) < 1e-7:
                        continue
                    a_i = a_i_old + t[i] * t[j] * (a_j_old - a_j)
                    alpha[i], alpha[j] = a_i, a_j
                    b1 = (
                        b
                        - E_i
                        - t[i] * (a_i - a_i_old) * K[i, i]
                        - t[j] * (a_j - a_j_old) * K[i, j]
                    )
                    b2 = (
                        b
                        - E_j
                        - t[i] * (a_i - a_i_old) * K[i, j]
                        - t[j] * (a_j - a_j_old) * K[j, j]
                    )
                    if 0 < a_i < self.C:
                        b = b1
                    elif 0 < a_j < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)
                    num_changed += 1
            passes = passes + 1 if num_changed == 0 else 0

        sv = alpha > 1e-8
        self.support_ = np.where(sv)[0]
        self.alpha_ = alpha[sv]
        self.support_vectors_ = X[sv]
        self.support_labels_ = t[sv]
        self.b_ = b
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance-like score of each sample to the decision boundary.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Decision values of shape (n_samples,).
        """
        self._check_fitted("support_vectors_")
        X = check_array(X)
        K = rbf_kernel(X, self.support_vectors_, self.gamma_)
        return K @ (self.alpha_ * self.support_labels_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,), in the original label space.
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])
