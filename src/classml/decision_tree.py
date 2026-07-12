"""Classification decision tree (CART) built by greedy Gini splitting.

The tree is grown top-down. At each node the split (feature, threshold) that most
reduces the Gini impurity of the two children is chosen by scanning every feature
and every midpoint between consecutive sorted values. Growth stops at max_depth,
when a node is pure, or when a node has too few samples to split.

Shallow trees give blocky axis-aligned decision regions that refine as max_depth
grows, which makes depth a natural knob for a decision-boundary animation.
"""

from __future__ import annotations

import numpy as np

from .base import Classifier, check_array


class _Node:
    """A single node of the tree, internal or leaf."""

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.value: int | None = None  # class index for leaves


def _gini(counts: np.ndarray) -> float:
    """Gini impurity of a class-count vector."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p**2))


class DecisionTreeClassifier(Classifier):
    """CART classification tree using Gini impurity.

    Args:
        max_depth: Maximum depth of the tree. None grows until purity or the
            minimum-split constraint stops growth.
        min_samples_split: Minimum samples required to attempt a split.
        min_samples_leaf: Minimum samples required in each child of a split.

    Attributes:
        classes_: Sorted unique class labels seen during fit.
        n_classes_: Number of distinct classes.
        root_: Root node of the fitted tree.
        depth_: Depth actually reached by the fitted tree.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.root_: _Node | None = None
        self.depth_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> DecisionTreeClassifier:
        """Grow the tree on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            The fitted estimator.
        """
        if y is None:
            raise ValueError("y is required for DecisionTreeClassifier.fit")
        X = check_array(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        class_to_int = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([class_to_int[c] for c in y])
        self.depth_ = 0
        self.root_ = self._grow(X, y_int, depth=0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """Recursively build a node from the samples reaching it."""
        node = _Node()
        counts = np.bincount(y, minlength=self.n_classes_)
        majority = int(np.argmax(counts))
        self.depth_ = max(self.depth_, depth)

        pure = np.count_nonzero(counts) == 1
        too_small = X.shape[0] < self.min_samples_split
        at_max = self.max_depth is not None and depth >= self.max_depth
        if pure or too_small or at_max:
            node.value = majority
            return node

        feature, threshold = self._best_split(X, y, counts)
        if feature is None:
            node.value = majority
            return node

        left_mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, parent_counts: np.ndarray
    ) -> tuple[int | None, float | None]:
        """Find the (feature, threshold) with the largest Gini gain."""
        n_samples = X.shape[0]
        parent_impurity = _gini(parent_counts)
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None

        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind="mergesort")
            x_sorted = X[order, feature]
            y_sorted = y[order]
            left_counts = np.zeros(self.n_classes_)
            right_counts = parent_counts.astype(float).copy()
            for i in range(1, n_samples):
                c = y_sorted[i - 1]
                left_counts[c] += 1
                right_counts[c] -= 1
                if x_sorted[i] == x_sorted[i - 1]:
                    continue
                n_left = i
                n_right = n_samples - i
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                impurity = (
                    n_left / n_samples * _gini(left_counts)
                    + n_right / n_samples * _gini(right_counts)
                )
                gain = parent_impurity - impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = 0.5 * (x_sorted[i] + x_sorted[i - 1])
        return best_feature, best_threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X by routing each sample to a leaf.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        self._check_fitted("root_")
        X = check_array(X)
        out = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i, row in enumerate(X):
            node = self.root_
            while node.value is None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = self.classes_[node.value]
        return out
