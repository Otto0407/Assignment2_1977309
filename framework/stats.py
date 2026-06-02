"""
framework/stats.py
------------------
Streaming statistical functions. All functions work chunk-wise using only numpy.

Numerical stability notes
-------------------------
- Welford's online algorithm is used for running mean/variance to avoid
  catastrophic cancellation with large-magnitude or high-precision data.
- NaN values in input chunks are ignored per-feature (nanmean / nanvar
  semantics): a feature column becomes NaN only if ALL its values are NaN.
- chunk_histogram with a zero-range column (all values equal) places every
  sample in the centre bin to avoid undefined behaviour.
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
    tuple : (count+1, new_mean, new_M2)
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
    Compute per-feature mean over a chunk, ignoring NaN values.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input chunk with n samples and d features.

    Returns
    -------
    np.ndarray, shape (d,)
        Per-feature mean values. NaN if all values in a column are NaN.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.nanmean(X, axis=0)


def chunk_variance(X: np.ndarray, ddof: int = 0) -> np.ndarray:
    """
    Compute per-feature variance over a chunk, ignoring NaN values.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input chunk.
    ddof : int, optional (default=0)
        Delta degrees of freedom. Use ddof=1 for sample variance.

    Returns
    -------
    np.ndarray, shape (d,)
        Per-feature variance values. NaN if a column has fewer valid
        values than required by ddof.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.nanvar(X, axis=0, ddof=ddof)


def chunk_quantile(X: np.ndarray, q: float) -> np.ndarray:
    """
    Compute per-feature quantile over a chunk, ignoring NaN values.

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
    return np.nanquantile(X, q, axis=0)


def chunk_histogram(
    X: np.ndarray,
    bins: int = 10,
    range_: tuple = None,
) -> tuple:
    """
    Compute per-feature histogram over a chunk (NaN values are excluded).

    When a feature column has zero range (all values identical) and no
    explicit range_ is given, all samples are placed in a single bin
    centred on that value.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input chunk.
    bins : int, optional (default=10)
        Number of equal-width bins.
    range_ : tuple (min, max) or None
        Shared range for all features. If None, each feature uses its own
        [nanmin, nanmax]; zero-range columns are handled safely.

    Returns
    -------
    counts : np.ndarray, shape (d, bins)
        Bin counts for each feature (NaN-excluded).
    edges : np.ndarray, shape (d, bins+1)
        Bin edges for each feature.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    _, d = X.shape
    counts_list = []
    edges_list = []
    for j in range(d):
        col = X[:, j]
        col_valid = col[~np.isnan(col)]
        if range_ is not None:
            hist_range = range_
        else:
            lo, hi = (np.nanmin(col), np.nanmax(col)) if len(col_valid) > 0 else (0.0, 1.0)
            if lo == hi:
                # Zero-range: create a symmetric bin around the constant value
                lo, hi = lo - 0.5, hi + 0.5
            hist_range = (lo, hi)
        c, e = np.histogram(col_valid, bins=bins, range=hist_range)
        counts_list.append(c)
        edges_list.append(e)
    return np.array(counts_list), np.array(edges_list)


# ---------------------------------------------------------------------------
# StreamStats class
# ---------------------------------------------------------------------------

class StreamStats:
    """
    Maintains running statistics over streaming chunks using Welford's
    online algorithm (vectorised over features).

    NaN values in a chunk are skipped per-feature: each feature maintains
    its own sample count so that a NaN in one feature does not affect others.

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
        self._count = np.zeros(self.n_features, dtype=np.int64)  # per-feature count
        self._mean = np.zeros(self.n_features, dtype=float)
        self._M2 = np.zeros(self.n_features, dtype=float)

    def update(self, X_chunk: np.ndarray) -> None:
        """
        Update running statistics with a new chunk of data.

        NaN values are ignored per-feature.

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
        # Vectorised Welford update row-by-row; skip NaN per feature
        for row in X_chunk:
            valid = ~np.isnan(row)
            if not np.any(valid):
                continue
            self._count[valid] += 1
            delta = np.where(valid, row - self._mean, 0.0)
            self._mean[valid] += delta[valid] / self._count[valid]
            delta2 = np.where(valid, row - self._mean, 0.0)
            self._M2[valid] += delta[valid] * delta2[valid]

    def mean(self) -> np.ndarray:
        """
        Return the current running mean per feature.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Per-feature running mean. NaN for features with no valid data.
        """
        result = self._mean.copy()
        result[self._count == 0] = np.nan
        return result

    def variance(self) -> np.ndarray:
        """
        Return the current running population variance per feature (ddof=0).

        Returns
        -------
        np.ndarray, shape (n_features,)
            Per-feature population variance. 0.0 for features with < 2 samples.
        """
        var = np.zeros(self.n_features, dtype=float)
        enough = self._count >= 2
        var[enough] = self._M2[enough] / self._count[enough]
        return var

    def std(self) -> np.ndarray:
        """
        Return the current running population standard deviation per feature.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        return np.sqrt(self.variance())

    @property
    def n_samples_seen(self) -> np.ndarray:
        """
        Per-feature sample count (excludes NaN observations).

        Returns
        -------
        np.ndarray, shape (n_features,), dtype int64
        """
        return self._count.copy()
