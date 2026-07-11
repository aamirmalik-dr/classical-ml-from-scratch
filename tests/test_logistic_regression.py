"""Tests for classml.LogisticRegression against the sklearn reference."""

import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.preprocessing import StandardScaler

from classml import LogisticRegression


def test_binary_accuracy_close_to_sklearn() -> None:
    X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
    X = StandardScaler().fit_transform(X)
    ours = LogisticRegression(learning_rate=0.5, n_iter=5000).fit(X, y)
    ref = SkLogisticRegression().fit(X, y)
    assert abs(ours.score(X, y) - ref.score(X, y)) <= 0.03


def test_multiclass_iris_accuracy() -> None:
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    ours = LogisticRegression(learning_rate=0.5, n_iter=5000).fit(X, y)
    ref = SkLogisticRegression(max_iter=5000).fit(X, y)
    assert ours.score(X, y) >= ref.score(X, y) - 0.05
    assert ours.score(X, y) >= 0.90


def test_predict_proba_rows_sum_to_one() -> None:
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(n_iter=500).fit(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-9)
    assert proba.shape == (len(X), 3)


def test_loss_decreases() -> None:
    X, y = make_moons(n_samples=200, noise=0.2, random_state=1)
    model = LogisticRegression(learning_rate=0.3, n_iter=300).fit(X, y)
    losses = np.asarray(model.loss_history_)
    assert losses[-1] < losses[0]
