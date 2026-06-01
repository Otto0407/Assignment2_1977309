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
        self.n_estimators = n_estimators
        self.method = method
        self.max_depth = max_depth
        self.max_features = max_features if method != 'bagging' else None
        self.random_state = random_state
        self.estimators_: list = []
        self.classes_: np.ndarray = None
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    def _make_estimator(self, seed: int) -> DecisionTreeClassifier:
        """Instantiate a single base estimator."""
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=seed,
        )

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleClassifier':
        """
        Fit all estimators on bootstrap samples of (X, y).

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
        self.estimators_ = []
        seeds = self._rng.integers(0, 2**31, size=self.n_estimators)
        for seed in seeds:
            rng_i = np.random.default_rng(int(seed))
            idx = rng_i.choice(n, size=n, replace=True)
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
        Incrementally update each estimator using a bootstrap sample of the chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            New data chunk.
        y : np.ndarray, shape (n,)
        classes : np.ndarray or None
            All possible class labels. Required on first call if not inferable.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if classes is not None:
            self.classes_ = classes
        elif self.classes_ is None:
            self.classes_ = np.unique(y)

        n = X.shape[0]
        seeds = self._rng.integers(0, 2**31, size=self.n_estimators)

        if len(self.estimators_) == 0:
            for seed in seeds:
                self.estimators_.append(self._make_estimator(int(seed)))

        for i, est in enumerate(self.estimators_):
            rng_i = np.random.default_rng(int(seeds[i]))
            idx = rng_i.choice(n, size=n, replace=True)
            est.partial_fit(X[idx], y[idx], classes=self.classes_)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict averaged class probabilities across all estimators.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, n_classes)
            Mean class probabilities.
        """
        if not self.estimators_:
            raise RuntimeError("Call fit or partial_fit before predict_proba.")
        X = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        proba_sum = np.zeros((X.shape[0], n_classes), dtype=float)
        for est in self.estimators_:
            # Align classes in case a tree was trained on a subset
            est_proba = est.predict_proba(X)
            for k, cls in enumerate(est.classes_):
                idx = np.where(self.classes_ == cls)[0]
                if len(idx) > 0:
                    proba_sum[:, idx[0]] += est_proba[:, k]
        return proba_sum / len(self.estimators_)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels by majority vote across all estimators.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted class labels.
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
            Fraction of correctly classified samples.
        """
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# RandomForestClassifier – thin subclass that enforces 'sqrt' max_features
# ---------------------------------------------------------------------------

class RandomForestClassifier(EnsembleClassifier):
    """
    Random forest classifier.

    Equivalent to EnsembleClassifier with method='random_forest' and
    max_features='sqrt'.

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
