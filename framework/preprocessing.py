"""
framework/preprocessing.py
---------------------------
Streaming-compatible scalers and encoders. All transformers support
partial_fit for incremental / chunk-wise learning. Only numpy is used.
"""

import numpy as np
from framework.stats import welford_update


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class StandardScaler:
    """
    Standardise features to zero mean and unit variance.
    Supports online (partial_fit) and batch (fit_transform) usage.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: np.ndarray = None
        self._M2: np.ndarray = None

    def partial_fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Incrementally update running mean and variance using Welford's
        algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input chunk.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, d = X.shape
        if self._mean is None:
            self._mean = np.zeros(d, dtype=float)
            self._M2 = np.zeros(d, dtype=float)
        for i in range(n):
            self._count += 1
            delta = X[i] - self._mean
            self._mean += delta / self._count
            delta2 = X[i] - self._mean
            self._M2 += delta * delta2
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
        X = np.asarray(X, dtype=float)
        var = self._M2 / self._count if self._count > 0 else np.ones_like(self._mean)
        std = np.sqrt(var)
        std[std == 0] = 1.0  # Avoid division by zero for constant features
        return (X - self._mean) / std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit on X then transform X.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, d)
        """
        return self.partial_fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the standardisation.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Standardised data.

        Returns
        -------
        np.ndarray, shape (n, d)
            Data in the original scale.
        """
        if self._mean is None:
            raise RuntimeError("Call partial_fit before inverse_transform.")
        X = np.asarray(X, dtype=float)
        var = self._M2 / self._count if self._count > 0 else np.ones_like(self._mean)
        std = np.sqrt(var)
        std[std == 0] = 1.0
        return X * std + self._mean


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
        self.feature_range = feature_range
        self._data_min: np.ndarray = None
        self._data_max: np.ndarray = None

    def partial_fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Update running min/max from a new chunk.

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
            self._data_min = chunk_min
            self._data_max = chunk_max
        else:
            self._data_min = np.minimum(self._data_min, chunk_min)
            self._data_max = np.maximum(self._data_max, chunk_max)
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
        scale[scale == 0] = 1.0
        lo, hi = self.feature_range
        X_std = (X - self._data_min) / scale
        return X_std * (hi - lo) + lo

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
        scale[scale == 0] = 1.0
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
        Imputation strategy. For 'constant', fill_value must be supplied
        at transform time or set via the fill_value attribute.
    fill_value : float, optional
        Value used when strategy='constant'.
    """

    def __init__(self, strategy: str = 'mean', fill_value: float = 0.0) -> None:
        if strategy not in ('mean', 'median', 'constant'):
            raise ValueError(f"Unknown strategy '{strategy}'.")
        self.strategy = strategy
        self.fill_value = fill_value
        self._statistics: np.ndarray = None
        # For online mean imputation
        self._count: int = 0
        self._mean: np.ndarray = None
        self._M2: np.ndarray = None

    def partial_fit(self, X: np.ndarray) -> 'Imputer':
        """
        Update imputation statistics from a new chunk.

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
        if self.strategy == 'mean':
            for i in range(X.shape[0]):
                if self._mean is None:
                    self._mean = np.zeros(X.shape[1], dtype=float)
                    self._M2 = np.zeros(X.shape[1], dtype=float)
                self._count += 1
                valid = ~np.isnan(X[i])
                # Only update non-NaN entries
                for j in np.where(valid)[0]:
                    _, self._mean[j], self._M2[j] = welford_update(
                        self._count - 1, self._mean[j], self._M2[j], X[i, j]
                    )
            self._statistics = self._mean.copy() if self._mean is not None else None
        elif self.strategy == 'median':
            # Accumulate all data for median (approximate for large streams)
            if not hasattr(self, '_buffer'):
                self._buffer = []
            self._buffer.append(X)
            combined = np.concatenate(self._buffer, axis=0)
            self._statistics = np.nanmedian(combined, axis=0)
        else:  # constant
            if X.ndim == 2:
                self._statistics = np.full(X.shape[1], self.fill_value)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace NaN values with the fitted statistic.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, d)
            Imputed array.
        """
        if self._statistics is None:
            raise RuntimeError("Call partial_fit before transform.")
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self._statistics[j]
        return X


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    Supports partial_fit to incrementally discover new categories.
    """

    def __init__(self) -> None:
        self.categories_: list = None  # list of sorted arrays, one per feature

    def partial_fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """
        Update known categories from a new chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Categorical feature array. Values will be converted to str.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=str)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape
        if self.categories_ is None:
            self.categories_ = [np.array([], dtype=str) for _ in range(d)]
        for j in range(d):
            new_cats = np.unique(X[:, j])
            self.categories_[j] = np.union1d(self.categories_[j], new_cats)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        One-hot encode X using the known categories.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n, sum_of_category_counts)
            Binary one-hot encoded matrix.
        """
        if self.categories_ is None:
            raise RuntimeError("Call partial_fit before transform.")
        X = np.asarray(X, dtype=str)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape
        blocks = []
        for j in range(d):
            cats = self.categories_[j]
            block = np.zeros((n, len(cats)), dtype=float)
            for k, cat in enumerate(cats):
                block[:, k] = (X[:, j] == cat).astype(float)
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    def get_feature_names(self) -> list:
        """
        Return feature names for the one-hot encoded columns.

        Returns
        -------
        list of str
            Names in the form 'x{feature_idx}_{category}'.
        """
        if self.categories_ is None:
            return []
        names = []
        for j, cats in enumerate(self.categories_):
            for cat in cats:
                names.append(f"x{j}_{cat}")
        return names
