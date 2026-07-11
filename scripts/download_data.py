"""Dataset availability check.

This project intentionally uses only datasets that ship with scikit-learn or are
generated in memory, so there is nothing to download. Running this script simply
verifies that every dataset used by the demos and tests loads correctly.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations


def main() -> None:
    """Load every dataset used in the project and print its shape."""
    from sklearn.datasets import load_diabetes, load_digits, load_iris, make_blobs, make_moons

    loaders = {
        "load_diabetes": lambda: load_diabetes(return_X_y=True),
        "load_iris": lambda: load_iris(return_X_y=True),
        "load_digits": lambda: load_digits(return_X_y=True),
        "make_blobs": lambda: make_blobs(n_samples=500, centers=4, random_state=42),
        "make_moons": lambda: make_moons(n_samples=300, noise=0.2, random_state=7),
    }
    print("All data comes from sklearn loaders or generators. Nothing to download.")
    for name, loader in loaders.items():
        X, y = loader()
        print(f"  {name:15s} OK  X={X.shape} y={y.shape}")


if __name__ == "__main__":
    main()
