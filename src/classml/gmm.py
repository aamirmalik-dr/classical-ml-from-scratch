"""Gaussian mixture model fitted with expectation-maximisation.

The model is p(x) = sum_k pi_k N(x | mu_k, Sigma_k) with full covariances.
EM alternates:

- E step: responsibilities r_ik = pi_k N(x_i | mu_k, Sigma_k) / sum_j pi_j N(...),
  computed in log space for numerical stability.
- M step: closed-form updates of pi_k, mu_k, Sigma_k from the responsibilities.

Each EM iteration is guaranteed not to decrease the data log-likelihood, which is
used as the convergence criterion. Means are initialised with an internal k-means
run and covariances with the data covariance.
"""

from __future__ import annotations

import numpy as np

from .base import Estimator, check_array
from .kmeans import KMeans


class GaussianMixtureEM(Estimator):
    """Gaussian mixture model with full covariance matrices.

    Args:
        n_components: Number of mixture components.
        max_iter: Maximum EM iterations.
        tol: Convergence threshold on the change in mean log-likelihood.
        reg_covar: Diagonal ridge added to covariances for numerical stability.
        random_state: Seed for the k-means initialisation.

    Attributes:
        weights_: Mixing coefficients pi of shape (n_components,).
        means_: Component means of shape (n_components, n_features).
        covariances_: Component covariances of shape (n_components, d, d).
        log_likelihood_history_: Mean log-likelihood after each EM iteration.
        n_iter_: Number of EM iterations actually run.
        converged_: Whether the tolerance was reached before max_iter.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 200,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.log_likelihood_history_: list[float] = []
        self.n_iter_: int = 0
        self.converged_: bool = False

    def _log_gaussians(self, X: np.ndarray) -> np.ndarray:
        """Log density of every sample under every component, shape (n, k)."""
        n, d = X.shape
        out = np.empty((n, self.n_components))
        for k in range(self.n_components):
            cov = self.covariances_[k]
            chol = np.linalg.cholesky(cov)
            diff = X - self.means_[k]
            z = np.linalg.solve(chol, diff.T)
            maha = (z**2).sum(axis=0)
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            out[:, k] = -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)
        return out

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute responsibilities and the mean log-likelihood."""
        weighted = self._log_gaussians(X) + np.log(self.weights_)
        m = weighted.max(axis=1, keepdims=True)
        log_norm = m + np.log(np.exp(weighted - m).sum(axis=1, keepdims=True))
        resp = np.exp(weighted - log_norm)
        return resp, float(log_norm.mean())

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        """Update weights, means, and covariances from responsibilities."""
        n, d = X.shape
        nk = resp.sum(axis=0) + 10.0 * np.finfo(float).eps
        self.weights_ = nk / n
        self.means_ = (resp.T @ X) / nk[:, None]
        covariances = np.empty((self.n_components, d, d))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            covariances[k] = (resp[:, k][:, None] * diff).T @ diff / nk[k]
            covariances[k].flat[:: d + 1] += self.reg_covar
        self.covariances_ = covariances

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> GaussianMixtureEM:
        """Fit the mixture with EM.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Ignored, present for interface consistency.

        Returns:
            The fitted estimator.
        """
        X = check_array(X)
        n, d = X.shape
        km = KMeans(n_clusters=self.n_components, n_init=5, random_state=self.random_state).fit(X)
        self.means_ = km.cluster_centers_.copy()
        self.weights_ = np.bincount(km.labels_, minlength=self.n_components).astype(float) / n
        self.weights_ = np.maximum(self.weights_, 1e-6)
        self.weights_ /= self.weights_.sum()
        base_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(d)
        self.covariances_ = np.tile(base_cov, (self.n_components, 1, 1))

        self.log_likelihood_history_ = []
        prev_ll = -np.inf
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            resp, ll = self._e_step(X)
            self._m_step(X, resp)
            self.log_likelihood_history_.append(ll)
            self.n_iter_ = it
            if ll - prev_ll < self.tol and it > 1:
                self.converged_ = True
                break
            prev_ll = ll
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior responsibility of each component for each sample.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Responsibilities of shape (n_samples, n_components).
        """
        self._check_fitted("means_")
        X = check_array(X)
        resp, _ = self._e_step(X)
        return resp

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard-assign each sample to its most responsible component.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Component indices of shape (n_samples,).
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray) -> float:
        """Mean log-likelihood of X under the fitted mixture.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            The average per-sample log-likelihood.
        """
        self._check_fitted("means_")
        X = check_array(X)
        _, ll = self._e_step(X)
        return ll
