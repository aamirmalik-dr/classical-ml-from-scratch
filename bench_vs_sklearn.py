"""Benchmark every classml estimator against its scikit-learn reference.

For each algorithm this runs the from-scratch estimator and the matching
scikit-learn estimator on the same toy dataset, reports the headline metric for
both, and records the gap and the fit time. Results are written to
``results/metrics.json`` and printed as a table.

The point is validation, not a speed contest: these clean-room NumPy
implementations are correct on the toy problems but are not tuned for scale, so
the timing column is only a rough sanity check.

Usage:
    python bench_vs_sklearn.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import load_diabetes, load_iris, load_wine, make_blobs, make_moons
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.neighbors import KNeighborsClassifier as SkKNN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as SkTree

from classml import (
    DecisionTreeClassifier,
    GaussianNB,
    KernelSVM,
    KMeans,
    KNeighborsClassifier,
    LinearRegression,
    LogisticRegression,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _timed(fn) -> tuple[Any, float]:
    """Run fn(), returning (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def bench_linear_regression() -> dict[str, Any]:
    """R^2 on the diabetes dataset, ours vs sklearn."""
    X, y = load_diabetes(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    ours, t = _timed(lambda: LinearRegression(solver="closed_form").fit(X, y))
    ref = SkLinearRegression().fit(X, y)
    return {
        "estimator": "LinearRegression",
        "dataset": "diabetes",
        "metric": "R2",
        "ours": round(ours.score(X, y), 4),
        "sklearn": round(ref.score(X, y), 4),
        "fit_seconds": round(t, 4),
    }


def bench_logistic_regression() -> dict[str, Any]:
    """Accuracy on two moons, ours vs sklearn."""
    X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
    X = StandardScaler().fit_transform(X)
    ours, t = _timed(lambda: LogisticRegression(learning_rate=0.5, n_iter=5000).fit(X, y))
    ref = SkLogisticRegression().fit(X, y)
    return {
        "estimator": "LogisticRegression",
        "dataset": "two_moons",
        "metric": "accuracy",
        "ours": round(ours.score(X, y), 4),
        "sklearn": round(ref.score(X, y), 4),
        "fit_seconds": round(t, 4),
    }


def bench_knn() -> dict[str, Any]:
    """Accuracy on iris, ours vs sklearn."""
    X, y = load_iris(return_X_y=True)
    ours, t = _timed(lambda: KNeighborsClassifier(n_neighbors=7).fit(X, y))
    ref = SkKNN(n_neighbors=7).fit(X, y)
    return {
        "estimator": "KNeighborsClassifier",
        "dataset": "iris",
        "metric": "accuracy",
        "ours": round(ours.score(X, y), 4),
        "sklearn": round(ref.score(X, y), 4),
        "fit_seconds": round(t, 4),
    }


def bench_decision_tree() -> dict[str, Any]:
    """Accuracy on iris (depth 4), ours vs sklearn."""
    X, y = load_iris(return_X_y=True)
    ours, t = _timed(lambda: DecisionTreeClassifier(max_depth=4).fit(X, y))
    ref = SkTree(max_depth=4, random_state=0).fit(X, y)
    return {
        "estimator": "DecisionTreeClassifier",
        "dataset": "iris",
        "metric": "accuracy",
        "ours": round(ours.score(X, y), 4),
        "sklearn": round(ref.score(X, y), 4),
        "fit_seconds": round(t, 4),
    }


def bench_naive_bayes() -> dict[str, Any]:
    """Accuracy on wine, ours vs sklearn."""
    X, y = load_wine(return_X_y=True)
    ours, t = _timed(lambda: GaussianNB().fit(X, y))
    ref = SkGaussianNB().fit(X, y)
    return {
        "estimator": "GaussianNB",
        "dataset": "wine",
        "metric": "accuracy",
        "ours": round(ours.score(X, y), 4),
        "sklearn": round(ref.score(X, y), 4),
        "fit_seconds": round(t, 4),
    }


def bench_svm() -> dict[str, Any]:
    """Accuracy on two moons, ours vs sklearn SVC."""
    X, y = make_moons(n_samples=300, noise=0.2, random_state=7)
    X = StandardScaler().fit_transform(X)
    ours, t = _timed(lambda: KernelSVM(C=1.0, gamma=1.0, random_state=0).fit(X, y))
    ref = SVC(C=1.0, gamma=1.0, kernel="rbf").fit(X, y)
    return {
        "estimator": "KernelSVM",
        "dataset": "two_moons",
        "metric": "accuracy",
        "ours": round(ours.score(X, y), 4),
        "sklearn": round(ref.score(X, y), 4),
        "fit_seconds": round(t, 4),
    }


def bench_kmeans() -> dict[str, Any]:
    """Clustering agreement (ARI) on blobs, ours vs sklearn labels."""
    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
    ours, t = _timed(lambda: KMeans(n_clusters=4, n_init=10, random_state=0).fit(X))
    ref = SkKMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    return {
        "estimator": "KMeans",
        "dataset": "blobs",
        "metric": "ARI_vs_true",
        "ours": round(adjusted_rand_score(ours.labels_, y_true), 4),
        "sklearn": round(adjusted_rand_score(ref.labels_, y_true), 4),
        "fit_seconds": round(t, 4),
    }


def main() -> None:
    """Run the full benchmark and write results/metrics.json."""
    RESULTS.mkdir(exist_ok=True)
    rows = [
        bench_linear_regression(),
        bench_logistic_regression(),
        bench_knn(),
        bench_decision_tree(),
        bench_naive_bayes(),
        bench_svm(),
        bench_kmeans(),
    ]

    header = f"{'estimator':<24}{'dataset':<12}{'metric':<14}{'ours':>8}{'sklearn':>10}{'fit s':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['estimator']:<24}{r['dataset']:<12}{r['metric']:<14}"
            f"{r['ours']:>8}{r['sklearn']:>10}{r['fit_seconds']:>9}"
        )

    payload = {
        "note": "Controlled validation on toy datasets. classml is educational, "
        "not tuned for speed or scale. sklearn is a reference only.",
        "numpy_version": np.__version__,
        "results": rows,
    }
    out = RESULTS / "metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
