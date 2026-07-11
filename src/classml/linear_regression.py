"""Linear regression solved in closed form or by batch gradient descent.

Model: y_hat = X w + b, fitted by minimising mean squared error

    L(w, b) = (1 / 2n) * sum_i (x_i . w + b - y_i)^2

The closed-form solver uses the normal equations via a least-squares solve on the
intercept-augmented design matrix. The gradient descent solver iterates

    w <- w - lr * (1/n) X^T (X w + b - y)
    b <- b - lr * (1/n) sum(X w + b - y)
"""

from __future__ import annotations

import numpy as np

from .base import Regressor, check_array


class LinearRegression(Regressor):
    """Ordinary least squares linear regression.

    Args:
        solver: "closed_form" for the normal-equation solution or "gd" for
            batch gradient descent.
        learning_rate: Step size for gradient descent. Ignored by closed form.
        n_iter: Number of gradient descent iterations. Ignored by closed form.
        tol: Stop gradient descent early when the loss improvement between
            iterations falls below this value.

    Attributes:
        coef_: Learned weights of shape (n_features,).
        intercept_: Learned bias term.
        loss_history_: Per-iteration MSE losses (gradient descent only).
    """

    def __init__(
        self,
        solver: str = "closed_form",
        learning_rate: float = 0.01,
        n_iter: int = 5000,
        tol: float = 1e-10,
    ) -> None:
        if solver not in ("closed_form", "gd"):
            raise ValueError(f"solver must be 'closed_form' or 'gd', got {solver!r}")
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> LinearRegression:
        """Fit the model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Targets of shape (n_samples,).

        Returns:
            The fitted estimator.
        """
        if y is None:
            raise ValueError("y is required for LinearRegression.fit")
        X = check_array(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.solver == "closed_form":
            self._fit_closed_form(X, y)
        else:
            self._fit_gradient_descent(X, y)
        return self

    def _fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> None:
        """Solve the normal equations with a numerically stable lstsq."""
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Minimise MSE with full-batch gradient descent."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        self.loss_history_ = []
        prev_loss = np.inf
        for _ in range(self.n_iter):
            residual = X @ w + b - y
            loss = float(0.5 * np.mean(residual**2))
            self.loss_history_.append(loss)
            grad_w = X.T @ residual / n_samples
            grad_b = float(residual.mean())
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
            if prev_loss - loss < self.tol:
                break
            prev_loss = loss
        self.coef_ = w
        self.intercept_ = b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,).
        """
        self._check_fitted("coef_")
        X = check_array(X)
        return X @ self.coef_ + self.intercept_
