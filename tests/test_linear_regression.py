"""Tests for classml.LinearRegression against the sklearn reference."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.preprocessing import StandardScaler

from classml import LinearRegression


def _data() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_diabetes(return_X_y=True)
    return StandardScaler().fit_transform(X), y


def test_closed_form_matches_sklearn_coefficients() -> None:
    X, y = _data()
    ours = LinearRegression(solver="closed_form").fit(X, y)
    ref = SkLinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-6, atol=1e-6)
    assert abs(ours.intercept_ - ref.intercept_) < 1e-6


def test_gradient_descent_matches_closed_form() -> None:
    X, y = _data()
    gd = LinearRegression(solver="gd", learning_rate=0.1, n_iter=20000, tol=1e-14).fit(X, y)
    cf = LinearRegression(solver="closed_form").fit(X, y)
    np.testing.assert_allclose(gd.coef_, cf.coef_, atol=0.5)
    assert abs(gd.intercept_ - cf.intercept_) < 0.5


def test_gd_loss_decreases_monotonically() -> None:
    X, y = _data()
    model = LinearRegression(solver="gd", learning_rate=0.1, n_iter=500).fit(X, y)
    losses = np.asarray(model.loss_history_)
    assert np.all(np.diff(losses) <= 1e-12)


def test_r2_score_matches_sklearn() -> None:
    X, y = _data()
    ours = LinearRegression().fit(X, y)
    ref = SkLinearRegression().fit(X, y)
    assert abs(ours.score(X, y) - ref.score(X, y)) < 1e-8


def test_predict_before_fit_raises() -> None:
    model = LinearRegression()
    try:
        model.predict(np.zeros((3, 2)))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for unfitted model")
