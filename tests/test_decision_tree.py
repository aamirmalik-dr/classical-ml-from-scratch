"""Tests for classml.DecisionTreeClassifier against the sklearn reference."""

from sklearn.datasets import load_iris, make_moons
from sklearn.tree import DecisionTreeClassifier as SkTree

from classml import DecisionTreeClassifier


def test_iris_unpruned_fits_training_data() -> None:
    X, y = load_iris(return_X_y=True)
    ours = DecisionTreeClassifier().fit(X, y)
    # A fully grown Gini tree separates the training set almost perfectly.
    assert ours.score(X, y) >= 0.99


def test_depth_limited_close_to_sklearn() -> None:
    X, y = load_iris(return_X_y=True)
    ours = DecisionTreeClassifier(max_depth=3).fit(X, y)
    ref = SkTree(max_depth=3, criterion="gini", random_state=0).fit(X, y)
    assert abs(ours.score(X, y) - ref.score(X, y)) <= 0.05


def test_moons_generalisation_close_to_sklearn() -> None:
    X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
    Xtr, ytr = X[:300], y[:300]
    Xte, yte = X[300:], y[300:]
    ours = DecisionTreeClassifier(max_depth=5).fit(Xtr, ytr)
    ref = SkTree(max_depth=5, random_state=0).fit(Xtr, ytr)
    assert abs(ours.score(Xte, yte) - ref.score(Xte, yte)) <= 0.08


def test_depth_grows_with_max_depth() -> None:
    X, y = make_moons(n_samples=200, noise=0.2, random_state=1)
    shallow = DecisionTreeClassifier(max_depth=2).fit(X, y)
    deep = DecisionTreeClassifier(max_depth=8).fit(X, y)
    assert shallow.depth_ <= 2
    assert deep.depth_ > shallow.depth_


def test_stump_makes_a_single_split() -> None:
    X, y = make_moons(n_samples=200, noise=0.2, random_state=3)
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert stump.depth_ == 1
    assert stump.root_.feature is not None
