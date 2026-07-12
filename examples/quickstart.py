"""Thirty-second quickstart: fit and predict on the committed sample, no network.

Run from the repo root:

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from classml import KNeighborsClassifier

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_moons.csv"


def main() -> None:
    """Load the synthetic sample, fit kNN, and report training accuracy."""
    table = np.loadtxt(SAMPLE, delimiter=",", skiprows=1)
    X, y = table[:, :2], table[:, 2].astype(int)

    model = KNeighborsClassifier(n_neighbors=15).fit(X, y)
    preds = model.predict(X[:5])
    print("first five predictions:", preds.tolist())
    print("first five true labels:", y[:5].tolist())
    print(f"training accuracy: {model.score(X, y):.4f}")


if __name__ == "__main__":
    main()
