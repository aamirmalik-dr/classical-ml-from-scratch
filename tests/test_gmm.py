"""Tests for classml.GaussianMixtureEM against the sklearn reference."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SkGaussianMixture

from classml import GaussianMixtureEM


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    return make_blobs(n_samples=600, centers=3, cluster_std=1.2, random_state=3)


def test_cluster_agreement_with_sklearn() -> None:
    X, _ = _blobs()
    ours = GaussianMixtureEM(n_components=3, random_state=0).fit(X)
    ref = SkGaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)
    assert adjusted_rand_score(ours.predict(X), ref.predict(X)) > 0.95


def test_log_likelihood_close_to_sklearn() -> None:
    X, _ = _blobs()
    ours = GaussianMixtureEM(n_components=3, random_state=0).fit(X)
    ref = SkGaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)
    assert abs(ours.score(X) - ref.score(X)) < 0.05


def test_log_likelihood_is_monotone_nondecreasing() -> None:
    X, _ = _blobs()
    model = GaussianMixtureEM(n_components=3, random_state=0).fit(X)
    ll = np.asarray(model.log_likelihood_history_)
    assert np.all(np.diff(ll) >= -1e-9)
    assert model.converged_


def test_responsibilities_sum_to_one() -> None:
    X, _ = _blobs()
    model = GaussianMixtureEM(n_components=3, random_state=0).fit(X)
    resp = model.predict_proba(X)
    np.testing.assert_allclose(resp.sum(axis=1), np.ones(len(X)), atol=1e-9)


def test_weights_sum_to_one() -> None:
    X, _ = _blobs()
    model = GaussianMixtureEM(n_components=3, random_state=0).fit(X)
    assert abs(model.weights_.sum() - 1.0) < 1e-9
