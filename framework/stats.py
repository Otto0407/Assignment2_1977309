"""
framework/stats.py
------------------
Streaming statistical functions. All functions work chunk-wise using only numpy.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Welford online algorithm helper
# ---------------------------------------------------------------------------

def welford_update(count: int, mean: float, M2: float, new_value: float) -> tuple:
    """
    Single-sample Welford online update for running mean and variance.

    Parameters
    ----------
    count : int
        Number of samples seen so far (before this update).
    mean : float
        Current running mean.
    M2 : float
        Current sum of squared deviations from the mean.
    new_value : float
        The new scalar observation.

    Returns
    -------
    tuple
        (count+1, new_mean, new_M2) where
        new_mean : float  – updated mean
        new_M2   : float  – updated sum of squared deviations
    """
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return count, mean, M2


# ---------------------------------------------------------------------------
# Standalone chunk functions
# ---------------------------------------------------------------------------

def chunk_mean(X: np.ndarray) -> np.ndarray:
    """
    Compute per-feature mean over a chunk.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input chunk with n samples and d features.

    Returns
    -------
    np.ndarray, shape (d,)
        Per-feature mean values.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.mean(X, axis=0)


def chunk_variance(X: np.ndarray, ddof: int = 0) -> np.ndarray:
    """
    Compute per-feature variance over a chunk.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input chunk.
    ddof : int, optional (default=0)
        Delta degrees of freedom. Use ddof=1 for sample variance.

    Returns
    -------
    np.ndarray, shape (d,)
        Per-feature variance values.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.var(X, axis=0, ddof=ddof)


def chunk_quantile(X: np.ndarray, q: float) -> np.ndarray:
    """
    Compute per-feature quantile over a chunk.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input chunk.
    q : float
        Quantile in [0, 1].

    Returns
    -------
    np.ndarray, shape (d,)
        Per-feature quantile values.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q}")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.quantile(X, q, axis=0)


def chunk_histogram(
    X: np.ndarray,
    bins: int = 10,
    range_: tuple = None,
) -> tuple:
    """
    Compute per-feature histogram over a chunk.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input chunk.
    bins : int, optional (default=10)
        Number of equal-width bins.
    range_ : tuple (min, max) or None
        Range for the bins. If None, uses [X.min(), X.max()] per feature.

    Returns
    -------
    counts : np.ndarray, shape (d, bins)
        Bin counts for each feature.
    edges : np.ndarray, shape (d, bins+1)
        Bin edges for each feature.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    counts_list = []
    edges_list = []
    for j in range(d):
        col = X[:, j]
        hist_range = range_ if range_ is not None else (col.min(), col.max())
        c, e = np.histogram(col, bins=bins, range=hist_range)
        counts_list.append(c)
        edges_list.append(e)
    return np.array(counts_list), np.array(edges_list)


# ---------------------------------------------------------------------------
# StreamStats class
# ---------------------------------------------------------------------------

class StreamStats:
    """
    Maintains running statistics over streaming chunks using Welford's
    online algorithm.

    Parameters
    ----------
    n_features : int
        Number of features (columns) expected in each chunk.
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.reset()

    def reset(self) -> None:
        """Reset all running statistics to their initial state."""
        self._count = 0
        self._mean = np.zeros(self.n_features, dtype=float)
        self._M2 = np.zeros(self.n_features, dtype=float)

    def update(self, X_chunk: np.ndarray) -> None:
        """
        Update running statistics with a new chunk of data.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_samples, n_features)
            New data chunk. Each row is a sample.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(1, -1)
        if X_chunk.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X_chunk.shape[1]}"
            )
        for i in range(X_chunk.shape[0]):
            for j in range(self.n_features):
                self._count_j = getattr(self, '_counts', None)
                # Per-feature update using vectorised Welford
                pass
            # Vectorised Welford step over all features at once
            self._count += 1
            delta = X_chunk[i] - self._mean
            self._mean += delta / self._count
            delta2 = X_chunk[i] - self._mean
            self._M2 += delta * delta2

    def mean(self) -> np.ndarray:
        """
        Return the current running mean per feature.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Per-feature running mean.
        """
        return self._mean.copy()

    def variance(self) -> np.ndarray:
        """
        Return the current running population variance per feature.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Per-feature population variance (ddof=0).
        """
        if self._count < 2:
            return np.zeros(self.n_features, dtype=float)
        return self._M2 / self._count

    def std(self) -> np.ndarray:
        """
        Return the current running population standard deviation per feature.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Per-feature standard deviation.
        """
        return np.sqrt(self.variance())
