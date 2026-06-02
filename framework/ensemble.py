"""
framework/ensemble.py
---------------------
Ensemble classifiers (Bagging, Random Forest) built on top of
framework.tree.DecisionTreeClassifier. Uses only numpy.
"""

from __future__ import annotations

import numpy as np
from framework.tree import DecisionTreeClassifier


class EnsembleClassifier:
    """
    Generic ensemble classifier supporting bagging and random-forest strategies.

    Parameters
    ----------
    n_estimators : int, default 10
        Number of base estimators.
    method : str, one of {'bagging', 'random_forest'}, default 'bagging'
        Ensemble strategy. 'bagging' uses bootstrap + all features;
        'random_forest' uses bootstrap + feature subsampling ('sqrt').
    max_depth : int, default 5
        Maximum tree depth for each estimator.
    max_features : str or int, default 'sqrt'
        Number of features per split passed to each tree.
        Ignored (set to None) when method='bagging'.
    random_state : int or None, default None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        method: str = 'bagging',
        max_depth: int = 5,
        max_features: str = 'sqrt',
        random_state: int = None,
    ) -> None:
        if method not in ('bagging', 'random_forest'):
            raise ValueError(f"method must be 'bagging' or 'random_forest', got {method!r}")
        self.n_estimators = n_estimators
        self.method = method
        self.max_depth = max_depth
        # bagging uses all features; random_forest subsamples
        self.max_features = None if method == 'bagging' else max_features
        self.random_state = random_state
        self.estimators_: list = []
        self.classes_: np.ndarray = None
        # Master RNG – used only to seed per-estimator RNGs deterministically
        self._rng = np.random.default_rng(random_state)
        # Store per-estimator seeds so bootstrap is reproducible across calls
        self._estimator_seeds: np.ndarray = None

    # ------------------------------------------------------------------
    def _make_estimator(self, seed: int) -> DecisionTreeClassifier:
        """Instantiate a single base estimator with a fixed seed."""
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=seed,
        )

    def _bootstrap(self, n: int, seed: int) -> np.ndarray:
        """Return bootstrap indices of length n using a dedicated RNG."""
        return np.random.default_rng(seed).choice(n, size=n, replace=True)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleClassifier':
        """
        Fit all estimators from scratch on bootstrap samples of (X, y).

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
        n = X.shape[0]

        # Generate one seed per estimator and store for use in partial_fit
        self._estimator_seeds = self._rng.integers(0, 2**31, size=self.n_estimators)
        self.estimators_ = []

        for seed in self._estimator_seeds:
            idx = self._bootstrap(n, int(seed))
            est = self._make_estimator(int(seed))
            est.fit(X[idx], y[idx])
            self.estimators_.append(est)
        return self

    # ------------------------------------------------------------------
    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray = None,
    ) -> 'EnsembleClassifier':
        """
        Incrementally update each estimator with a bootstrap sample of the chunk.

        On the first call the estimators are created. Subsequent calls call
        each estimator's partial_fit so it accumulates data over chunks.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        classes : np.ndarray or None
            All possible class labels. Pass on every call to ensure correct
            probability output shape when early chunks miss some classes.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if classes is not None:
            self.classes_ = np.asarray(classes)
        elif self.classes_ is None:
            self.classes_ = np.unique(y)

        n = X.shape[0]

        # Initialise estimators and their seeds on first call
        if len(self.estimators_) == 0:
            self._estimator_seeds = self._rng.integers(
                0, 2**31, size=self.n_estimators
            )
            for seed in self._estimator_seeds:
                self.estimators_.append(self._make_estimator(int(seed)))

        # Derive chunk-specific bootstrap seed by XOR-ing estimator seed with
        # the number of samples seen so far (monotonically increasing → unique)
        n_seen = sum(
            est._X_acc.shape[0] if est._X_acc is not None else 0
            for est in self.estimators_[:1]
        )
        for i, est in enumerate(self.estimators_):
            chunk_seed = int(self._estimator_seeds[i]) ^ int(n_seen + 1)
            idx = self._bootstrap(n, chunk_seed)
            est.partial_fit(X[idx], y[idx], classes=self.classes_)

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict averaged class probabilities across all estimators.

        Handles the case where individual trees were trained on a bootstrap
        that only contained a subset of classes: missing columns are filled
        with 0 before averaging, then the result is renormalised to sum to 1.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, n_classes)
            Mean class probabilities, rows sum to 1.
        """
        if not self.estimators_:
            raise RuntimeError("Call fit or partial_fit before predict_proba.")
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba_sum = np.zeros((n_samples, n_classes), dtype=float)

        for est in self.estimators_:
            est_proba = est.predict_proba(X)          # (n, len(est.classes_))
            for k, cls in enumerate(est.classes_):
                global_idx = np.searchsorted(self.classes_, cls)
                if global_idx < n_classes and self.classes_[global_idx] == cls:
                    proba_sum[:, global_idx] += est_proba[:, k]

        # Average and renormalise (guards against missing-class bootstrap edge case)
        proba_avg = proba_sum / len(self.estimators_)
        row_sums = proba_avg.sum(axis=1, keepdims=True)
        # Where all estimators predicted 0 for all classes (shouldn't happen
        # after at least one chunk), fall back to uniform distribution
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return proba_avg / row_sums

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels by majority vote (argmax of averaged probas).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return mean accuracy on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)

        Returns
        -------
        float
        """
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# RandomForestClassifier – thin subclass that enforces 'sqrt' max_features
# ---------------------------------------------------------------------------

class RandomForestClassifier(EnsembleClassifier):
    """
    Random forest classifier.

    Equivalent to EnsembleClassifier(method='random_forest', max_features='sqrt').

    Parameters
    ----------
    n_estimators : int, default 10
    max_depth : int, default 5
    max_features : str or int, default 'sqrt'
    random_state : int or None, default None
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 5,
        max_features: str = 'sqrt',
        random_state: int = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            method='random_forest',
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
        )
