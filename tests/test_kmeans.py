"""Tests for classml.KMeans against the sklearn reference."""

import numpy as np
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from classml import KMeans


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    return make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)


def test_cluster_agreement_with_sklearn() -> None:
    X, _ = _blobs()
    ours = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    ref = SkKMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    assert adjusted_rand_score(ours.labels_, ref.labels_) > 0.95


def test_inertia_close_to_sklearn() -> None:
    X, _ = _blobs()
    ours = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    ref = SkKMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    assert ours.inertia_ <= ref.inertia_ * 1.02


def test_recovers_true_blob_labels() -> None:
    X, y_true = _blobs()
    ours = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    assert adjusted_rand_score(ours.labels_, y_true) > 0.90


def test_predict_matches_training_labels() -> None:
    X, _ = _blobs()
    model = KMeans(n_clusters=4, n_init=5, random_state=0).fit(X)
    np.testing.assert_array_equal(model.predict(X), model.labels_)


def test_centers_shape() -> None:
    X, _ = _blobs()
    model = KMeans(n_clusters=4, random_state=0).fit(X)
    assert model.cluster_centers_.shape == (4, X.shape[1])
