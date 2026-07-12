"""Generate the committed synthetic 2D sample dataset.

The quickstart, the gallery, and the hero animation all run on a small synthetic
two-moons dataset so the demo needs no network and no external download. This
script regenerates ``data/sample_moons.csv`` deterministically. The data is
entirely synthetic (scikit-learn ``make_moons``), so it is license clean and safe
to commit.

Usage:
    python scripts/generate_sample.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data"
SAMPLE = DATA / "sample_moons.csv"


def generate(n_samples: int = 300, noise: float = 0.25, seed: int = 0) -> np.ndarray:
    """Create a standardised two-moons sample as an (n, 3) array [x1, x2, label]."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = StandardScaler().fit_transform(X)
    return np.column_stack([X, y.astype(np.float64)])


def main() -> None:
    """Write the sample CSV to data/."""
    DATA.mkdir(exist_ok=True)
    table = generate()
    header = "x1,x2,label"
    np.savetxt(SAMPLE, table, delimiter=",", header=header, comments="", fmt=["%.6f", "%.6f", "%d"])
    print(f"Wrote {len(table)} rows to {SAMPLE}")


if __name__ == "__main__":
    main()
