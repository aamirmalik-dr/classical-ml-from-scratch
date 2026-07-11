"""Tests for classml.KernelSVM against the sklearn reference."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classml import KernelSVM
from classml.svm import rbf_kernel


def _moons() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=300, noise=0.2, random_state=7)
    return StandardScaler().fit_transform(X), y


def test_accuracy_close_to_sklearn_svc() -> None:
    X, y = _moons()
    ours = KernelSVM(C=1.0, gamma=1.0, random_state=0).fit(X, y)
    ref = SVC(C=1.0, gamma=1.0, kernel="rbf").fit(X, y)
    assert ours.score(X, y) >= ref.score(X, y) - 0.03
    assert ours.score(X, y) >= 0.90


def test_rbf_kernel_matches_manual_computation() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3))
    B = rng.normal(size=(4, 3))
    gamma = 0.7
    K = rbf_kernel(A, B, gamma)
    expected = np.array([[np.exp(-gamma * np.sum((a - b) ** 2)) for b in B] for a in A])
    np.testing.assert_allclose(K, expected, rtol=1e-12)


def test_predictions_agree_with_sklearn_on_most_points() -> None:
    X, y = _moons()
    ours = KernelSVM(C=1.0, gamma=1.0, random_state=0).fit(X, y)
    ref = SVC(C=1.0, gamma=1.0, kernel="rbf").fit(X, y)
    agreement = float(np.mean(ours.predict(X) == ref.predict(X)))
    assert agreement >= 0.95


def test_preserves_original_labels() -> None:
    X, y01 = _moons()
    y = np.where(y01 == 1, 5, -3)
    model = KernelSVM(C=1.0, gamma=1.0, random_state=0).fit(X, y)
    assert set(np.unique(model.predict(X))).issubset({-3, 5})


def test_rejects_more_than_two_classes() -> None:
    X = np.random.default_rng(0).normal(size=(30, 2))
    y = np.arange(30) % 3
    try:
        KernelSVM().fit(X, y)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 3-class input")
