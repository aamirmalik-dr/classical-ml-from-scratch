"""Logistic regression trained with batch gradient descent.

Binary case: p(y=1 | x) = sigmoid(x . w + b), trained by minimising the mean
cross-entropy loss with optional L2 regularisation

    L(w, b) = -(1/n) sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)] + (l2 / 2n) ||w||^2

Multiclass inputs are handled with a one-vs-rest scheme: one binary model per
class, predictions take the class with the highest probability.
"""

from __future__ import annotations

import numpy as np

from .base import Classifier, check_array


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression(Classifier):
    """Logistic regression classifier (binary or one-vs-rest multiclass).

    Args:
        learning_rate: Gradient descent step size.
        n_iter: Maximum number of gradient descent iterations.
        l2: L2 regularisation strength (0 disables regularisation).
        tol: Stop early when the loss improvement falls below this value.

    Attributes:
        classes_: Sorted unique class labels seen during fit.
        coef_: Weights of shape (n_classes_or_1, n_features).
        intercept_: Biases of shape (n_classes_or_1,).
        loss_history_: Per-iteration losses of the last binary problem fitted.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iter: int = 2000,
        l2: float = 0.0,
        tol: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self.tol = tol
        self.classes_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> LogisticRegression:
        """Fit the classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            The fitted estimator.
        """
        if y is None:
            raise ValueError("y is required for LogisticRegression.fit")
        X = check_array(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least two classes")

        if len(self.classes_) == 2:
            targets = [(y == self.classes_[1]).astype(np.float64)]
        else:
            targets = [(y == c).astype(np.float64) for c in self.classes_]

        weights, biases = [], []
        for target in targets:
            w, b = self._fit_binary(X, target)
            weights.append(w)
            biases.append(b)
        self.coef_ = np.vstack(weights)
        self.intercept_ = np.asarray(biases)
        return self

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Fit a single binary problem with 0/1 targets, returns (w, b)."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        self.loss_history_ = []
        prev_loss = np.inf
        for _ in range(self.n_iter):
            p = _sigmoid(X @ w + b)
            eps = 1e-12
            loss = float(
                -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                + 0.5 * self.l2 * np.dot(w, w) / n_samples
            )
            self.loss_history_.append(loss)
            error = p - y
            grad_w = (X.T @ error + self.l2 * w) / n_samples
            grad_b = float(error.mean())
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
            if prev_loss - loss < self.tol:
                break
            prev_loss = loss
        return w, b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probabilities of shape (n_samples, n_classes). For binary problems
            the columns follow the order of classes_. For multiclass, one-vs-rest
            scores are normalised to sum to one.
        """
        self._check_fitted("coef_")
        X = check_array(X)
        scores = _sigmoid(X @ self.coef_.T + self.intercept_)
        if len(self.classes_) == 2:
            return np.column_stack([1.0 - scores[:, 0], scores[:, 0]])
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
