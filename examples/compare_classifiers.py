"""Fit every classifier through the shared API and print a small accuracy table.

Because all estimators expose the same fit / predict / score interface, swapping
one for another is a single line. This example loops over a list of constructed
models on the committed sample.

    python examples/compare_classifiers.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from classml import (
    DecisionTreeClassifier,
    GaussianNB,
    KernelSVM,
    KNeighborsClassifier,
    LogisticRegression,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_moons.csv"


def main() -> None:
    """Fit each classifier on the sample and print its training accuracy."""
    table = np.loadtxt(SAMPLE, delimiter=",", skiprows=1)
    X, y = table[:, :2], table[:, 2].astype(int)

    models = {
        "logistic regression": LogisticRegression(learning_rate=0.5, n_iter=3000),
        "gaussian naive bayes": GaussianNB(),
        "k-nearest neighbours": KNeighborsClassifier(n_neighbors=15),
        "decision tree (d=5)": DecisionTreeClassifier(max_depth=5),
        "rbf kernel svm": KernelSVM(C=1.0, gamma=1.0, random_state=0),
    }

    print(f"{'model':<24}{'train accuracy':>16}")
    print("-" * 40)
    for name, model in models.items():
        model.fit(X, y)
        print(f"{name:<24}{model.score(X, y):>16.4f}")


if __name__ == "__main__":
    main()
