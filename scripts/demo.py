"""End-to-end demo of every classml algorithm on sklearn toy data.

Runs each estimator, prints headline metrics next to the sklearn reference where
that makes sense, and writes one figure per algorithm to results/.

Usage:
    python scripts/demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import load_diabetes, load_digits, make_blobs, make_moons
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SkGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classml import PCA, GaussianMixtureEM, KernelSVM, KMeans, LinearRegression, LogisticRegression

RESULTS = Path(__file__).resolve().parent.parent / "results"


def demo_linear_regression() -> None:
    """Fit diabetes data with both solvers and plot predicted vs true targets."""
    X, y = load_diabetes(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    cf = LinearRegression(solver="closed_form").fit(X, y)
    gd = LinearRegression(solver="gd", learning_rate=0.1, n_iter=20000, tol=1e-14).fit(X, y)
    ref = SkLinearRegression().fit(X, y)
    print("[LinearRegression] diabetes dataset")
    print(f"  R2 closed form : {cf.score(X, y):.4f}")
    print(f"  R2 grad descent: {gd.score(X, y):.4f}")
    print(f"  R2 sklearn ref : {ref.score(X, y):.4f}")
    print(f"  max |coef diff| closed form vs sklearn: {np.max(np.abs(cf.coef_ - ref.coef_)):.2e}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, cf.predict(X), s=12, alpha=0.6)
    lims = [y.min(), y.max()]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("true target")
    ax.set_ylabel("predicted target")
    ax.set_title(f"Linear regression on diabetes (R2 = {cf.score(X, y):.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "linear_regression_fit.png", dpi=120)
    plt.close(fig)


def _plot_decision_boundary(ax: plt.Axes, model: object, X: np.ndarray, y: np.ndarray) -> None:
    """Shade the 2D decision regions of a fitted classifier."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.predict(grid).reshape(xx.shape).astype(float)
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=14, edgecolors="k", linewidths=0.3)


def demo_logistic_regression() -> None:
    """Classify two-moons data and draw the learned decision boundary."""
    X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
    X = StandardScaler().fit_transform(X)
    model = LogisticRegression(learning_rate=0.5, n_iter=5000).fit(X, y)
    acc = model.score(X, y)
    print("[LogisticRegression] two moons (linear boundary, so accuracy is capped)")
    print(f"  training accuracy: {acc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_decision_boundary(ax, model, X, y)
    ax.set_title(f"Logistic regression on two moons (acc = {acc:.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "logistic_regression_boundary.png", dpi=120)
    plt.close(fig)


def demo_kernel_svm() -> None:
    """Fit the RBF SVM on two moons and compare accuracy with sklearn SVC."""
    X, y = make_moons(n_samples=300, noise=0.2, random_state=7)
    X = StandardScaler().fit_transform(X)
    model = KernelSVM(C=1.0, gamma=1.0, random_state=0).fit(X, y)
    ref = SVC(C=1.0, gamma=1.0, kernel="rbf").fit(X, y)
    print("[KernelSVM] two moons, RBF kernel, simplified SMO")
    print(f"  training accuracy (ours)   : {model.score(X, y):.4f}")
    print(f"  training accuracy (sklearn): {ref.score(X, y):.4f}")
    print(f"  support vectors: {len(model.support_)} of {len(X)} samples")

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_decision_boundary(ax, model, X, y)
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=60,
        facecolors="none",
        edgecolors="k",
        linewidths=0.8,
        label="support vectors",
    )
    ax.legend(loc="lower left")
    ax.set_title(f"RBF SVM on two moons (acc = {model.score(X, y):.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "svm_boundary.png", dpi=120)
    plt.close(fig)


def demo_kmeans() -> None:
    """Cluster synthetic blobs and compare inertia with sklearn KMeans."""
    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
    model = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    ref = SkKMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    ari_ref = adjusted_rand_score(model.labels_, ref.labels_)
    ari_true = adjusted_rand_score(model.labels_, y_true)
    print("[KMeans] 4 gaussian blobs, k-means++ init, 10 restarts")
    print(f"  inertia (ours)   : {model.inertia_:.2f}")
    print(f"  inertia (sklearn): {ref.inertia_:.2f}")
    print(f"  ARI vs sklearn labels: {ari_ref:.4f}")
    print(f"  ARI vs true labels   : {ari_true:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", s=14)
    ax.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=200,
        c="red",
        edgecolors="k",
        label="centroids",
    )
    ax.legend()
    ax.set_title(f"K-means on blobs (inertia = {model.inertia_:.1f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "kmeans_clusters.png", dpi=120)
    plt.close(fig)


def demo_pca() -> None:
    """Project the digits dataset to 2D and report explained variance."""
    X, y = load_digits(return_X_y=True)
    model = PCA(n_components=2)
    Z = model.fit_transform(X)
    evr = model.explained_variance_ratio_
    print("[PCA] digits dataset (64 features)")
    print(f"  explained variance ratio PC1: {evr[0]:.4f}, PC2: {evr[1]:.4f}")
    print(f"  total variance captured by 2 components: {evr.sum():.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=10, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="digit")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of digits ({evr.sum():.1%} variance in 2 components)")
    fig.tight_layout()
    fig.savefig(RESULTS / "pca_digits.png", dpi=120)
    plt.close(fig)


def _cov_ellipse(ax: plt.Axes, mean: np.ndarray, cov: np.ndarray, color: str) -> None:
    """Draw 1- and 2-sigma ellipses of a 2D gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    angle = float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])))
    for n_sigma in (1.0, 2.0):
        width, height = 2.0 * n_sigma * np.sqrt(vals[-1]), 2.0 * n_sigma * np.sqrt(vals[0])
        ellipse = Ellipse(mean, width, height, angle=angle, fill=False, color=color, linewidth=1.5)
        ax.add_patch(ellipse)


def demo_gmm() -> None:
    """Fit a 3-component GMM on blobs and draw covariance ellipses."""
    X, _ = make_blobs(n_samples=600, centers=3, cluster_std=1.2, random_state=3)
    model = GaussianMixtureEM(n_components=3, random_state=0).fit(X)
    ref = SkGaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)
    print("[GaussianMixtureEM] 3 gaussian blobs, full covariance")
    print(f"  converged in {model.n_iter_} EM iterations (tol {model.tol})")
    print(f"  mean log-likelihood (ours)   : {model.score(X):.4f}")
    print(f"  mean log-likelihood (sklearn): {ref.score(X):.4f}")
    ari = adjusted_rand_score(model.predict(X), ref.predict(X))
    print(f"  ARI vs sklearn cluster labels: {ari:.4f}")

    labels = model.predict(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=12, alpha=0.7)
    colors = ["tab:red", "tab:orange", "tab:purple"]
    for k in range(model.n_components):
        _cov_ellipse(ax, model.means_[k], model.covariances_[k], colors[k % len(colors)])
    ax.set_title(f"GMM with EM (mean log-lik = {model.score(X):.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "gmm_ellipses.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run every demo in sequence."""
    RESULTS.mkdir(exist_ok=True)
    demo_linear_regression()
    print()
    demo_logistic_regression()
    print()
    demo_kernel_svm()
    print()
    demo_kmeans()
    print()
    demo_pca()
    print()
    demo_gmm()
    print(f"\nFigures written to {RESULTS}")


if __name__ == "__main__":
    main()
