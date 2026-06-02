"""
framework/preprocessing.py
---------------------------
Streaming-compatible scalers and encoders. All transformers support
partial_fit for incremental / chunk-wise learning. Only numpy is used.

Vectorisation notes
-------------------
StandardScaler.partial_fit
    Uses Chan's parallel algorithm to combine (count, mean, M2) from the
    incoming chunk with the running state in O(1) scalar ops — no Python
    loop over rows.

Imputer.partial_fit (mean strategy)
    Per-feature Welford is applied in a single vectorised pass using masked
    arrays; no Python loop over rows or columns.

Imputer.transform
    NaN positions are filled via np.where broadcast — no column loop.

OneHotEncoder.transform
    The inner "loop over categories" is replaced by broadcasting
    X[:, j] (n,1) == cats (1, C) to produce the full block at once.

OneHotEncoder.partial_fit / get_feature_names
    Retain a Python loop over features because each feature has a
    variable-length category list; numpy cannot unify these into a
    single rectangular operation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chan_combine(
    n_a: int, mean_a: np.ndarray, M2_a: np.ndarray,
    n_b: int, mean_b: np.ndarray, M2_b: np.ndarray,
) -> tuple:
    """
    Chan's parallel algorithm: combine two sets of sufficient statistics.

    Parameters
    ----------
    n_a, mean_a, M2_a : count / mean / sum-of-sq-dev for set A
    n_b, mean_b, M2_b : count / mean / sum-of-sq-dev for set B

    Returns
    -------
    (n_ab, mean_ab, M2_ab) : combined statistics
    """
    n_ab = n_a + n_b
    if n_ab == 0:
        return 0, mean_a.copy(), M2_a.copy()
    delta = mean_b - mean_a
    mean_ab = (n_a * mean_a + n_b * mean_b) / n_ab
    M2_ab = M2_a + M2_b + delta ** 2 * n_a * n_b / n_ab
    return n_ab, mean_ab, M2_ab


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class StandardScaler:
    """
    Standardise features to zero mean and unit variance.
    Supports online (partial_fit) and batch (fit_transform) usage.

    Running mean and variance are updated with Chan's parallel algorithm
    — one vectorised operation per chunk, no Python loop over rows.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: np.ndarray = None
        self._M2: np.ndarray = None

    # --- internal state helpers ---

    def _var(self) -> np.ndarray:
        """Population variance from current state."""
        return self._M2 / self._count if self._count > 0 else np.ones_like(self._mean)

    def _std(self) -> np.ndarray:
        std = np.sqrt(self._var())
        std[std == 0] = 1.0
        return std

    # --- public API ---

    def partial_fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Incrementally update running mean and variance.

        Uses Chan's parallel algorithm: compute chunk statistics once,
        then merge with the running state — O(d) numpy ops, zero row loops.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_b, d = X.shape

        # Chunk statistics (vectorised)
        mean_b = np.mean(X, axis=0)                    # (d,)
        M2_b = np.sum((X - mean_b) ** 2, axis=0)       # (d,)

        if self._mean is None:
            self._mean = np.zeros(d, dtype=float)
            self._M2 = np.zeros(d, dtype=float)

        self._count, self._mean, self._M2 = _chan_combine(
            self._count, self._mean, self._M2,
            n_b, mean_b, M2_b,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardise X using the running mean and variance.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, d)
            Zero-mean, unit-variance transformed array.
        """
        if self._mean is None:
            raise RuntimeError("Call partial_fit before transform.")
        return (np.asarray(X, dtype=float) - self._mean) / self._std()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X then transform X. X : (n, d) -> (n, d)."""
        return self.partial_fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the standardisation.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, d)
        """
        if self._mean is None:
            raise RuntimeError("Call partial_fit before inverse_transform.")
        return np.asarray(X, dtype=float) * self._std() + self._mean


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """
    Scale features to a given range [feature_range[0], feature_range[1]].
    Supports partial_fit for streaming use.

    Parameters
    ----------
    feature_range : tuple (min, max), default (0, 1)
    """

    def __init__(self, feature_range: tuple = (0, 1)) -> None:
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range must satisfy lo < hi, got {feature_range}")
        self.feature_range = feature_range
        self._data_min: np.ndarray = None
        self._data_max: np.ndarray = None

    def partial_fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Update running min/max from a new chunk (fully vectorised).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        if self._data_min is None:
            self._data_min = chunk_min.copy()
            self._data_max = chunk_max.copy()
        else:
            np.minimum(self._data_min, chunk_min, out=self._data_min)
            np.maximum(self._data_max, chunk_max, out=self._data_max)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale X to feature_range.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, d)
        """
        if self._data_min is None:
            raise RuntimeError("Call partial_fit before transform.")
        X = np.asarray(X, dtype=float)
        scale = self._data_max - self._data_min
        scale = np.where(scale == 0, 1.0, scale)
        lo, hi = self.feature_range
        return (X - self._data_min) / scale * (hi - lo) + lo

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the min-max scaling.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, d)
        """
        if self._data_min is None:
            raise RuntimeError("Call partial_fit before inverse_transform.")
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        scale = self._data_max - self._data_min
        scale = np.where(scale == 0, 1.0, scale)
        return (X - lo) / (hi - lo) * scale + self._data_min


# ---------------------------------------------------------------------------
# Imputer
# ---------------------------------------------------------------------------

class Imputer:
    """
    Impute missing values (NaN) in feature arrays.

    Parameters
    ----------
    strategy : str, one of {'mean', 'median', 'constant'}, default 'mean'
    fill_value : float
        Value used when strategy='constant'.
    """

    def __init__(self, strategy: str = 'mean', fill_value: float = 0.0) -> None:
        if strategy not in ('mean', 'median', 'constant'):
            raise ValueError(f"Unknown strategy '{strategy}'.")
        self.strategy = strategy
        self.fill_value = fill_value
        self._statistics: np.ndarray = None
        # Mean strategy: per-feature Welford state (all vectorised)
        self._count: np.ndarray = None   # shape (d,) int
        self._mean: np.ndarray = None    # shape (d,)
        self._M2: np.ndarray = None      # shape (d,)
        # Median strategy: buffer of all chunks
        self._buffer: list = []

    def partial_fit(self, X: np.ndarray) -> 'Imputer':
        """
        Update imputation statistics from a new chunk.

        Mean strategy: per-feature Chan combination — zero Python row/column
        loops; NaN positions are masked out per feature before combining.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, d = X.shape

        if self.strategy == 'mean':
            if self._count is None:
                self._count = np.zeros(d, dtype=float)
                self._mean  = np.zeros(d, dtype=float)
                self._M2    = np.zeros(d, dtype=float)

            # Per-feature: count valid (non-NaN) samples in this chunk
            valid_mask = ~np.isnan(X)                          # (n, d) bool
            n_b = valid_mask.sum(axis=0).astype(float)         # (d,)

            # Chunk mean: nanmean per feature (returns 0 where n_b==0)
            X_safe = np.where(valid_mask, X, 0.0)
            mean_b = np.where(n_b > 0, X_safe.sum(axis=0) / np.where(n_b > 0, n_b, 1), 0.0)

            # Chunk M2: sum of squared deviations from chunk mean
            diff = np.where(valid_mask, X - mean_b, 0.0)      # (n, d)
            M2_b = (diff ** 2).sum(axis=0)                     # (d,)

            # Chan combine per feature (vectorised)
            n_ab = self._count + n_b
            safe_n_ab = np.where(n_ab > 0, n_ab, 1.0)
            delta = mean_b - self._mean
            self._mean = np.where(
                n_ab > 0,
                (self._count * self._mean + n_b * mean_b) / safe_n_ab,
                self._mean,
            )
            self._M2 = self._M2 + M2_b + delta ** 2 * self._count * n_b / safe_n_ab
            self._count = n_ab

            self._statistics = np.where(self._count > 0, self._mean, np.nan)

        elif self.strategy == 'median':
            self._buffer.append(X)
            combined = np.concatenate(self._buffer, axis=0)
            self._statistics = np.nanmedian(combined, axis=0)

        else:  # constant
            self._statistics = np.full(d, self.fill_value, dtype=float)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace NaN values with the fitted statistic (fully vectorised).

        Uses np.where broadcast — no column loop.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, d)
        """
        if self._statistics is None:
            raise RuntimeError("Call partial_fit before transform.")
        X = np.asarray(X, dtype=float)
        nan_mask = np.isnan(X)                              # (n, d)
        fill = np.broadcast_to(self._statistics, X.shape)  # (n, d) view
        return np.where(nan_mask, fill, X)


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    Supports partial_fit to incrementally discover new categories.

    Notes
    -----
    A Python loop over features is unavoidable in partial_fit and transform
    because each feature has a different number of categories, making the
    output widths non-uniform. The inner "loop over categories" inside
    transform has been replaced by a single broadcasting comparison.
    """

    def __init__(self) -> None:
        self.categories_: list = None   # list[np.ndarray[str]], one per feature

    def partial_fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """
        Update known categories from a new chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Categorical feature array (converted to str internally).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=str)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        _, d = X.shape
        if self.categories_ is None:
            self.categories_ = [np.array([], dtype=str) for _ in range(d)]
        for j in range(d):                                  # unavoidable: ragged output
            self.categories_[j] = np.union1d(self.categories_[j], X[:, j])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        One-hot encode X using the known categories.

        The inner category loop is replaced by broadcasting:
            X[:, j].reshape(-1, 1) == cats.reshape(1, -1)  →  (n, C_j) bool block

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, sum_of_category_counts)
        """
        if self.categories_ is None:
            raise RuntimeError("Call partial_fit before transform.")
        X = np.asarray(X, dtype=str)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape
        blocks = []
        for j in range(d):                                  # unavoidable: ragged widths
            cats = self.categories_[j]                      # (C_j,)
            # Vectorised: compare each sample against all categories at once
            block = (X[:, j].reshape(-1, 1) == cats.reshape(1, -1)).astype(float)
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    def get_feature_names(self) -> list:
        """
        Return feature names for the one-hot encoded columns.

        Returns
        -------
        list of str – names in the form 'x{feature_idx}_{category}'
        """
        if self.categories_ is None:
            return []
        return [
            f"x{j}_{cat}"
            for j, cats in enumerate(self.categories_)
            for cat in cats
        ]
