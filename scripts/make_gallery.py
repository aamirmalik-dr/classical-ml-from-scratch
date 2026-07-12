"""Render the gallery contact sheet and the hero decision-boundary animation.

Two artifacts are produced from the committed synthetic sample:

* ``assets/gallery.png`` is a contact sheet of six decision regions, one per
  classifier in the library, all fitted on the same two-moons sample.
* ``assets/decision_boundary.gif`` is the hero animation. Four estimators sweep
  their complexity knob in lockstep (kNN neighbours, tree depth, RBF gamma, and
  logistic-regression training iterations) so you can watch each boundary go from
  underfit to flexible across the frames.

Both run offline on ``data/sample_moons.csv``. Regenerate with:

    python scripts/make_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classml import (
    DecisionTreeClassifier,
    GaussianNB,
    KernelSVM,
    KMeans,
    KNeighborsClassifier,
    LogisticRegression,
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SAMPLE = ROOT / "data" / "sample_moons.csv"

GRID = 220  # gallery mesh resolution
GRID_ANIM = 130  # animation mesh resolution (smaller for a lighter GIF)


def load_sample() -> tuple[np.ndarray, np.ndarray]:
    """Load the committed synthetic two-moons sample."""
    table = np.loadtxt(SAMPLE, delimiter=",", skiprows=1)
    return table[:, :2], table[:, 2].astype(int)


def _mesh(X: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a prediction grid spanning the data with a small margin."""
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid


def _draw_boundary(
    ax: plt.Axes,
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    grid: np.ndarray,
    title: str,
) -> None:
    """Shade the decision regions of a fitted model and overlay the sample."""
    zz = model.predict(grid).reshape(xx.shape).astype(float)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], alpha=0.30, cmap="coolwarm")
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.0)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12, edgecolors="k", linewidths=0.25)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def build_gallery(X: np.ndarray, y: np.ndarray) -> Path:
    """Render the six-panel contact sheet and return its path."""
    xx, yy, grid = _mesh(X, GRID)
    panels = [
        ("Logistic regression", LogisticRegression(learning_rate=0.5, n_iter=3000).fit(X, y)),
        ("Gaussian naive Bayes", GaussianNB().fit(X, y)),
        ("k-NN (k = 15)", KNeighborsClassifier(n_neighbors=15).fit(X, y)),
        ("Decision tree (depth 5)", DecisionTreeClassifier(max_depth=5).fit(X, y)),
        ("RBF kernel SVM", KernelSVM(C=1.0, gamma=1.0, random_state=0).fit(X, y)),
        ("k-means Voronoi (k = 2)", KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (title, model) in zip(axes.ravel(), panels, strict=True):
        _draw_boundary(ax, model, X, y, xx, yy, grid, title)
    fig.suptitle(
        "classml decision regions on one synthetic two-moons sample", fontsize=13, y=0.98
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ASSETS / "gallery.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _anim_frame(
    X: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    grid: np.ndarray,
    k: int,
    depth: int,
    gamma: float,
    n_iter: int,
) -> np.ndarray:
    """Render one 2x2 animation frame and return it as an RGB array."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    specs = [
        (axes[0, 0], KNeighborsClassifier(n_neighbors=k).fit(X, y), f"k-NN, k = {k}"),
        (axes[0, 1], DecisionTreeClassifier(max_depth=depth).fit(X, y), f"tree, depth = {depth}"),
        (
            axes[1, 0],
            KernelSVM(C=1.0, gamma=gamma, random_state=0).fit(X, y),
            f"RBF SVM, gamma = {gamma:.2f}",
        ),
        (
            axes[1, 1],
            LogisticRegression(learning_rate=0.5, n_iter=n_iter).fit(X, y),
            f"logistic, iters = {n_iter}",
        ),
    ]
    for ax, model, title in specs:
        _draw_boundary(ax, model, X, y, xx, yy, grid, title)
    fig.suptitle("Decision boundary vs model complexity", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def build_animation(X: np.ndarray, y: np.ndarray, n_frames: int = 16) -> Path:
    """Render the hero GIF sweeping each estimator's complexity knob."""
    xx, yy, grid = _mesh(X, GRID_ANIM)
    # Complexity rises across frames: fewer neighbours, deeper trees, larger
    # gamma, more logistic-regression iterations.
    ks = np.round(np.linspace(60, 1, n_frames)).astype(int)
    depths = np.round(np.linspace(1, 9, n_frames)).astype(int)
    gammas = np.linspace(0.1, 6.0, n_frames)
    iters = np.round(np.linspace(2, 400, n_frames)).astype(int)

    frames = []
    for i in range(n_frames):
        frames.append(
            _anim_frame(X, y, xx, yy, grid, int(ks[i]), int(depths[i]), float(gammas[i]), int(iters[i]))
        )
        print(f"  frame {i + 1}/{n_frames}")
    # Hold the final (most flexible) frame a little longer.
    frames.extend([frames[-1]] * 3)
    out = ASSETS / "decision_boundary.gif"
    imageio.mimsave(out, frames, duration=0.45, loop=0)
    return out


def main() -> None:
    """Regenerate both the gallery and the hero animation."""
    ASSETS.mkdir(exist_ok=True)
    X, y = load_sample()
    gallery = build_gallery(X, y)
    print(f"Wrote {gallery} ({gallery.stat().st_size / 1024:.0f} KB)")
    gif = build_animation(X, y)
    print(f"Wrote {gif} ({gif.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
