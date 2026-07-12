"""Tests for classml.KNeighborsClassifier against the sklearn reference."""

import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.neighbors import KNeighborsClassifier as SkKNN

from classml import KNeighborsClassifier


def test_moons_matches_sklearn_labels() -> None:
    X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
    ours = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    ref = SkKNN(n_neighbors=5).fit(X, y)
    # Identical algorithm, so predictions should agree on almost every point.
    agree = np.mean(ours.predict(X) == ref.predict(X))
    assert agree >= 0.98


def test_iris_multiclass_accuracy() -> None:
    X, y = load_iris(return_X_y=True)
    ours = KNeighborsClassifier(n_neighbors=7).fit(X, y)
    ref = SkKNN(n_neighbors=7).fit(X, y)
    assert abs(ours.score(X, y) - ref.score(X, y)) <= 0.02


def test_k1_reproduces_training_labels() -> None:
    X, y = make_moons(n_samples=120, noise=0.1, random_state=1)
    ours = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    assert ours.score(X, y) == 1.0


def test_predict_proba_rows_sum_to_one() -> None:
    X, y = load_iris(return_X_y=True)
    proba = KNeighborsClassifier(n_neighbors=5).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-12)
    assert proba.shape == (len(X), 3)


def test_distance_weighting_runs() -> None:
    X, y = make_moons(n_samples=200, noise=0.25, random_state=2)
    ours = KNeighborsClassifier(n_neighbors=9, weights="distance").fit(X, y)
    ref = SkKNN(n_neighbors=9, weights="distance").fit(X, y)
    assert abs(ours.score(X, y) - ref.score(X, y)) <= 0.03
