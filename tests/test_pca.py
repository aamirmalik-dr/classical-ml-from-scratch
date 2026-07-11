"""Tests for classml.PCA against the sklearn reference."""

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA as SkPCA

from classml import PCA


def test_explained_variance_ratio_matches_sklearn() -> None:
    X, _ = load_iris(return_X_y=True)
    ours = PCA(n_components=4).fit(X)
    ref = SkPCA(n_components=4).fit(X)
    np.testing.assert_allclose(
        ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-8
    )


def test_components_match_up_to_sign() -> None:
    X, _ = load_iris(return_X_y=True)
    ours = PCA(n_components=2).fit(X)
    ref = SkPCA(n_components=2).fit(X)
    for k in range(2):
        dot = abs(float(np.dot(ours.components_[k], ref.components_[k])))
        assert dot > 1.0 - 1e-8


def test_transform_preserves_pairwise_structure() -> None:
    X, _ = load_digits(return_X_y=True)
    ours = PCA(n_components=10).fit(X)
    ref = SkPCA(n_components=10).fit(X)
    Z_ours = ours.transform(X)
    Z_ref = ref.transform(X)
    np.testing.assert_allclose(np.abs(Z_ours), np.abs(Z_ref), rtol=1e-4, atol=1e-6)


def test_inverse_transform_reconstruction() -> None:
    X, _ = load_iris(return_X_y=True)
    model = PCA(n_components=4).fit(X)
    X_rec = model.inverse_transform(model.transform(X))
    np.testing.assert_allclose(X_rec, X, atol=1e-10)


def test_partial_reconstruction_error_matches_sklearn() -> None:
    X, _ = load_digits(return_X_y=True)
    k = 16
    ours = PCA(n_components=k).fit(X)
    ref = SkPCA(n_components=k).fit(X)
    err_ours = float(np.mean((X - ours.inverse_transform(ours.transform(X))) ** 2))
    err_ref = float(np.mean((X - ref.inverse_transform(ref.transform(X))) ** 2))
    assert abs(err_ours - err_ref) < 1e-8
