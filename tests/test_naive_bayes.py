"""Tests for classml.GaussianNB against the sklearn reference."""

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.naive_bayes import GaussianNB as SkGaussianNB

from classml import GaussianNB


def test_iris_matches_sklearn_predictions() -> None:
    X, y = load_iris(return_X_y=True)
    ours = GaussianNB().fit(X, y)
    ref = SkGaussianNB().fit(X, y)
    # Same closed-form estimator, so predictions should be identical.
    assert np.array_equal(ours.predict(X), ref.predict(X))


def test_wine_accuracy_close_to_sklearn() -> None:
    X, y = load_wine(return_X_y=True)
    ours = GaussianNB().fit(X, y)
    ref = SkGaussianNB().fit(X, y)
    assert abs(ours.score(X, y) - ref.score(X, y)) <= 0.01


def test_parameters_match_sklearn() -> None:
    X, y = load_iris(return_X_y=True)
    ours = GaussianNB().fit(X, y)
    ref = SkGaussianNB().fit(X, y)
    np.testing.assert_allclose(ours.theta_, ref.theta_, atol=1e-9)
    np.testing.assert_allclose(ours.var_, ref.var_, rtol=1e-6)


def test_predict_proba_matches_sklearn() -> None:
    X, y = load_iris(return_X_y=True)
    ours = GaussianNB().fit(X, y).predict_proba(X)
    ref = SkGaussianNB().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(ours, ref, atol=1e-6)
    np.testing.assert_allclose(ours.sum(axis=1), np.ones(len(X)), atol=1e-12)
