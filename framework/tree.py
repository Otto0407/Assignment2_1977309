"""
framework/tree.py
-----------------
Streaming-compatible decision tree classifier built with numpy only.
Supports both batch fit and incremental partial_fit for online learning.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal node dataclass
# ---------------------------------------------------------------------------

class _Node:
    """
    A node in the decision tree.

    Attributes
    ----------
    feature : int or None
        Index of the feature used for splitting (None for leaf).
    threshold : float or None
        Threshold value for the split (None for leaf).
    left : _Node or None
        Left child (samples where feature <= threshold).
    right : _Node or None
        Right child (samples where feature > threshold).
    value : int or None
        Predicted class label at this leaf node.
    proba : np.ndarray or None
        Class probability distribution at this leaf node, shape (n_classes,).
    """

    def __init__(
        self,
        feature: int = None,
        threshold: float = None,
        left: '_Node' = None,
        right: '_Node' = None,
        value: int = None,
        proba: np.ndarray = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


# ---------------------------------------------------------------------------
# DecisionTreeClassifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """
    A decision tree classifier that supports both batch fitting and
    incremental (streaming) partial_fit via chunk-wise tree reconstruction.

    Parameters
    ----------
    max_depth : int, default 5
        Maximum depth of the tree.
    min_samples_split : int, default 2
        Minimum number of samples required to split an internal node.
    criterion : str, one of {'gini', 'entropy'}, default 'gini'
        Impurity criterion used to evaluate splits.
    max_features : str, int, or None, default None
        Number of features to consider when looking for the best split.
        'sqrt' -> int(sqrt(d)), 'log2' -> int(log2(d)), int -> exact,
        None -> all features.
    random_state : int or None, default None
        Seed for the random number generator (feature subsampling).
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = 'gini',
        max_features: 'str | int | None' = None,
        random_state: int = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.tree_: _Node = None
        self.classes_: np.ndarray = None
        # Accumulated data for streaming
        self._X_acc: np.ndarray = None
        self._y_acc: np.ndarray = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Build the tree from scratch on the full dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build(X, y, depth=0, rng=rng)
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray = None,
    ) -> 'DecisionTreeClassifier':
        """
        Incrementally grow the tree by accumulating data and rebuilding.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        classes : np.ndarray, shape (n_classes,) or None
            All possible class labels. If None, inferred from y.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self._X_acc is None:
            self._X_acc = X
            self._y_acc = y
        else:
            self._X_acc = np.vstack([self._X_acc, X])
            self._y_acc = np.concatenate([self._y_acc, y])
        if classes is not None:
            self.classes_ = classes
        return self.fit(self._X_acc, self._y_acc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted class labels.
        """
        if self.tree_ is None:
            raise RuntimeError("Call fit or partial_fit before predict.")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_sample(x, self.tree_) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, n_classes)
            Class probability estimates.
        """
        if self.tree_ is None:
            raise RuntimeError("Call fit or partial_fit before predict_proba.")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_proba_sample(x, self.tree_) for x in X])

    # ------------------------------------------------------------------
    # Private tree-building helpers
    # ------------------------------------------------------------------

    def _resolve_max_features(self, d: int) -> int:
        """Resolve max_features to a concrete integer count."""
        if self.max_features is None:
            return d
        if isinstance(self.max_features, int):
            return min(self.max_features, d)
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(d)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(d)))
        raise ValueError(f"Unknown max_features value: {self.max_features!r}")

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int, rng) -> _Node:
        """Recursively build the tree."""
        n, d = X.shape
        n_classes = len(self.classes_)

        # Leaf conditions
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._make_leaf(y)

        # Choose feature subset
        max_feat = self._resolve_max_features(d)
        feature_indices = rng.choice(d, size=max_feat, replace=False)

        best_feat, best_thr, best_gain = self._best_split(X, y, feature_indices)
        if best_feat is None or best_gain <= 0:
            return self._make_leaf(y)

        left_mask = X[:, best_feat] <= best_thr
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return self._make_leaf(y)

        left = self._build(X[left_mask], y[left_mask], depth + 1, rng)
        right = self._build(X[right_mask], y[right_mask], depth + 1, rng)
        return _Node(feature=best_feat, threshold=best_thr, left=left, right=right)

    def _make_leaf(self, y: np.ndarray) -> _Node:
        """Create a leaf node from a set of labels."""
        n = len(y)
        n_classes = len(self.classes_)
        proba = np.zeros(n_classes, dtype=float)
        for k, cls in enumerate(self.classes_):
            proba[k] = np.sum(y == cls) / n
        value = self.classes_[np.argmax(proba)]
        return _Node(value=value, proba=proba)

    def _gini(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity for a set of labels.

        Parameters
        ----------
        y : np.ndarray, shape (n,)

        Returns
        -------
        float
            Gini impurity in [0, 0.5].
        """
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return 1.0 - np.sum(probs ** 2)

    def _entropy(self, y: np.ndarray) -> float:
        """
        Compute Shannon entropy for a set of labels.

        Parameters
        ----------
        y : np.ndarray, shape (n,)

        Returns
        -------
        float
            Entropy in nats.
        """
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _impurity(self, y: np.ndarray) -> float:
        """Dispatch to the selected criterion."""
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        raise ValueError(f"Unknown criterion: {self.criterion!r}")

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray = None,
    ) -> tuple:
        """
        Find the best (feature, threshold) split by information gain.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        feature_indices : np.ndarray or None
            Subset of feature indices to search. If None, use all.

        Returns
        -------
        tuple : (best_feature_idx, best_threshold, best_gain)
            If no valid split found, returns (None, None, -inf).
        """
        n, d = X.shape
        if feature_indices is None:
            feature_indices = np.arange(d)

        base_impurity = self._impurity(y)
        best_feat = None
        best_thr = None
        best_gain = -np.inf

        for feat in feature_indices:
            thresholds = np.unique(X[:, feat])
            if len(thresholds) < 2:
                continue
            # Use midpoints between consecutive unique values
            mid_thresholds = (thresholds[:-1] + thresholds[1:]) / 2.0
            for thr in mid_thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                if n_left == 0 or n_right == 0:
                    continue
                gain = base_impurity - (
                    n_left / n * self._impurity(y[left_mask])
                    + n_right / n * self._impurity(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = thr

        return best_feat, best_thr, best_gain

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_sample(self, x: np.ndarray, node: _Node):
        """Traverse the tree for a single sample and return the class label."""
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def _predict_proba_sample(self, x: np.ndarray, node: _Node) -> np.ndarray:
        """Traverse the tree and return class probabilities for one sample."""
        if node.is_leaf:
            return node.proba
        if x[node.feature] <= node.threshold:
            return self._predict_proba_sample(x, node.left)
        return self._predict_proba_sample(x, node.right)
