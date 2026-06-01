"""
framework/pipeline.py
---------------------
Streaming Pipeline that chains transformers and a final estimator.
Uses only numpy (no sklearn dependency).
"""

from __future__ import annotations

import numpy as np
from framework.metrics import accuracy_score


class Pipeline:
    """
    Sequential pipeline of (name, step) pairs where every step except the
    last must expose a transform() method, and the last step may be an
    estimator with fit/predict/predict_proba/score.

    All steps optionally support partial_fit for streaming use.

    Parameters
    ----------
    steps : list of (str, object) tuples
        Ordered list of (name, transformer_or_estimator) pairs.
        Names must be unique.
    """

    def __init__(self, steps: list) -> None:
        if not steps:
            raise ValueError("steps must be a non-empty list.")
        names = [name for name, _ in steps]
        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique.")
        self.steps = steps

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    @property
    def _transformers(self):
        return [(name, step) for name, step in self.steps[:-1]]

    def __getitem__(self, name: str):
        """
        Return the step with the given name.

        Parameters
        ----------
        name : str

        Returns
        -------
        object – the step object
        """
        for n, step in self.steps:
            if n == name:
                return step
        raise KeyError(f"No step named '{name}'.")

    # ------------------------------------------------------------------
    def _transform_through(self, X: np.ndarray) -> np.ndarray:
        """Pass X through all transformer steps (all but last)."""
        for _name, step in self._transformers:
            X = step.transform(X)
        return X

    # ------------------------------------------------------------------
    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        **fit_params,
    ) -> 'Pipeline':
        """
        Incrementally fit each step on the data flowing through the pipeline.

        For transformer steps: call partial_fit(X) then transform(X) to pass
        to the next step. For the final estimator: call partial_fit(X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,) or None

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        for _name, step in self._transformers:
            if hasattr(step, 'partial_fit'):
                step.partial_fit(X)
            X = step.transform(X)
        est = self._final_estimator
        if hasattr(est, 'partial_fit'):
            est.partial_fit(X, y, **fit_params)
        elif hasattr(est, 'fit'):
            est.fit(X, y)
        return self

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'Pipeline':
        """
        Fit each step in sequence.

        Transformers are fit+transformed; the final estimator is fit on the
        transformed data.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,) or None

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        for _name, step in self._transformers:
            if hasattr(step, 'fit'):
                step.fit(X)
            elif hasattr(step, 'partial_fit'):
                step.partial_fit(X)
            X = step.transform(X)
        est = self._final_estimator
        if hasattr(est, 'fit'):
            est.fit(X, y)
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply all transformer steps (excludes the final estimator).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, d_transformed)
        """
        X = np.asarray(X, dtype=float)
        return self._transform_through(X)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X and predict with the final estimator.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        X = np.asarray(X, dtype=float)
        X_t = self._transform_through(X)
        return self._final_estimator.predict(X_t)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X and return class probabilities from the final estimator.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, n_classes)
        """
        X = np.asarray(X, dtype=float)
        X_t = self._transform_through(X)
        return self._final_estimator.predict_proba(X_t)

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy of predictions on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)

        Returns
        -------
        float : accuracy in [0, 1]
        """
        y = np.asarray(y)
        return accuracy_score(y, self.predict(X))
