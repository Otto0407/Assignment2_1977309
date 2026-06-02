"""
framework/metrics.py
--------------------
Streaming and batch classification metrics. Uses only numpy.

Vectorisation notes
-------------------
confusion_matrix / ConfusionMatrix.update
    Uses np.bincount on (y_true * n_classes + y_pred) — one call, zero loops.

roc_auc_score
    TPR/FPR curves built with np.cumsum after argsort — no Python iteration.

precision_score / recall_score
    Per-class TP/FP/FN computed with vectorised boolean masks; the only
    remaining Python loop is over classes (unavoidable: variable output width).

Accuracy (windowed)
    Uses collections.deque(maxlen=window_size) — extend() replaces the
    per-element append/popleft loop.
"""

from __future__ import annotations

from collections import deque
import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class StreamingMetric:
    """
    Abstract base class for streaming metrics.

    Subclasses must implement update(), result(), and reset().
    """

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        raise NotImplementedError

    def result(self) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

class Accuracy(StreamingMetric):
    """
    Streaming accuracy metric.

    Parameters
    ----------
    window_size : int or None
        If given, computes accuracy over the last window_size predictions.
        If None, computes cumulative accuracy.
    """

    def __init__(self, window_size: int = None) -> None:
        self.window_size = window_size
        self._correct = 0
        self._total = 0
        self._window: deque = (
            deque(maxlen=window_size) if window_size is not None else None
        )

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        hits = (y_true == y_pred).astype(int)
        if self._window is not None:
            self._window.extend(hits.tolist())
        else:
            self._correct += int(hits.sum())
            self._total += len(hits)

    def result(self) -> float:
        if self._window is not None:
            if len(self._window) == 0:
                return 0.0
            return float(np.mean(list(self._window)))
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
        if self._window is not None:
            self._window.clear()


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

class Precision(StreamingMetric):
    """
    Streaming precision metric.

    Parameters
    ----------
    average : str, one of {'macro', 'micro', 'weighted'}, default 'macro'
    window_size : int or None
    """

    def __init__(self, average: str = 'macro', window_size: int = None) -> None:
        self.average = average
        self.window_size = window_size
        self._y_true_buf: list = []
        self._y_pred_buf: list = []
        self._window_size = window_size

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._y_true_buf.extend(list(np.asarray(y_true)))
        self._y_pred_buf.extend(list(np.asarray(y_pred)))
        if self._window_size is not None:
            self._y_true_buf = self._y_true_buf[-self._window_size:]
            self._y_pred_buf = self._y_pred_buf[-self._window_size:]

    def result(self) -> float:
        if not self._y_true_buf:
            return 0.0
        return precision_score(
            np.array(self._y_true_buf),
            np.array(self._y_pred_buf),
            average=self.average,
        )

    def reset(self) -> None:
        self._y_true_buf = []
        self._y_pred_buf = []


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

class Recall(StreamingMetric):
    """
    Streaming recall metric.

    Parameters
    ----------
    average : str, default 'macro'
    window_size : int or None
    """

    def __init__(self, average: str = 'macro', window_size: int = None) -> None:
        self.average = average
        self._window_size = window_size
        self._y_true_buf: list = []
        self._y_pred_buf: list = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._y_true_buf.extend(list(np.asarray(y_true)))
        self._y_pred_buf.extend(list(np.asarray(y_pred)))
        if self._window_size is not None:
            self._y_true_buf = self._y_true_buf[-self._window_size:]
            self._y_pred_buf = self._y_pred_buf[-self._window_size:]

    def result(self) -> float:
        if not self._y_true_buf:
            return 0.0
        return recall_score(
            np.array(self._y_true_buf),
            np.array(self._y_pred_buf),
            average=self.average,
        )

    def reset(self) -> None:
        self._y_true_buf = []
        self._y_pred_buf = []


# ---------------------------------------------------------------------------
# F1Score
# ---------------------------------------------------------------------------

class F1Score(StreamingMetric):
    """
    Streaming F1 score metric.

    Parameters
    ----------
    average : str, default 'macro'
    window_size : int or None
    """

    def __init__(self, average: str = 'macro', window_size: int = None) -> None:
        self.average = average
        self._window_size = window_size
        self._y_true_buf: list = []
        self._y_pred_buf: list = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._y_true_buf.extend(list(np.asarray(y_true)))
        self._y_pred_buf.extend(list(np.asarray(y_pred)))
        if self._window_size is not None:
            self._y_true_buf = self._y_true_buf[-self._window_size:]
            self._y_pred_buf = self._y_pred_buf[-self._window_size:]

    def result(self) -> float:
        if not self._y_true_buf:
            return 0.0
        return f1_score(
            np.array(self._y_true_buf),
            np.array(self._y_pred_buf),
            average=self.average,
        )

    def reset(self) -> None:
        self._y_true_buf = []
        self._y_pred_buf = []


# ---------------------------------------------------------------------------
# ConfusionMatrix
# ---------------------------------------------------------------------------

class ConfusionMatrix(StreamingMetric):
    """
    Streaming confusion matrix.

    Parameters
    ----------
    n_classes : int
        Number of target classes.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self._matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        # Mask out-of-range labels
        valid = (y_true >= 0) & (y_true < self.n_classes) & \
                (y_pred >= 0) & (y_pred < self.n_classes)
        y_true, y_pred = y_true[valid], y_pred[valid]
        # Vectorised: linearise (row, col) index, count with bincount
        linear_idx = y_true * self.n_classes + y_pred
        counts = np.bincount(linear_idx, minlength=self.n_classes ** 2)
        self._matrix += counts.reshape(self.n_classes, self.n_classes)

    def result(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray, shape (n_classes, n_classes)
            Entry [i, j] = count of true-class-i predicted as class-j.
        """
        return self._matrix.copy()

    def reset(self) -> None:
        self._matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

class AUC(StreamingMetric):
    """
    Streaming ROC-AUC metric (binary classification).

    Parameters
    ----------
    window_size : int or None
    """

    def __init__(self, window_size: int = None) -> None:
        self._window_size = window_size
        self._y_true_buf: list = []
        self._y_score_buf: list = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._y_true_buf.extend(list(np.asarray(y_true)))
        self._y_score_buf.extend(list(np.asarray(y_pred)))
        if self._window_size is not None:
            self._y_true_buf = self._y_true_buf[-self._window_size:]
            self._y_score_buf = self._y_score_buf[-self._window_size:]

    def result(self) -> float:
        if not self._y_true_buf:
            return 0.0
        return roc_auc_score(
            np.array(self._y_true_buf),
            np.array(self._y_score_buf),
        )

    def reset(self) -> None:
        self._y_true_buf = []
        self._y_score_buf = []


# ---------------------------------------------------------------------------
# Standalone batch functions
# ---------------------------------------------------------------------------

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute fraction of correct predictions.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
    y_pred : np.ndarray, shape (n,)

    Returns
    -------
    float : accuracy in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro',
) -> float:
    """
    Compute precision.

    Parameters
    ----------
    y_true  : np.ndarray, shape (n,)
    y_pred  : np.ndarray, shape (n,)
    average : str, one of {'macro', 'micro', 'weighted'}, default 'macro'

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    # micro: global TP / (TP + FP) = accuracy
    if average == 'micro':
        tp = int(np.sum(y_true == y_pred))
        total = len(y_true)
        return float(tp / total) if total > 0 else 0.0
    # Only iterate over classes that appear in y_true (per sklearn convention)
    classes = np.unique(y_true)
    scores = []
    weights = []
    for cls in classes:
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        denom = tp + fp
        scores.append(float(tp / denom) if denom > 0 else 0.0)
        weights.append(int(np.sum(y_true == cls)))
    if average == 'weighted':
        total = sum(weights)
        return float(np.dot(scores, weights) / total) if total > 0 else 0.0
    return float(np.mean(scores)) if scores else 0.0


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro',
) -> float:
    """
    Compute recall.

    Parameters
    ----------
    y_true  : np.ndarray, shape (n,)
    y_pred  : np.ndarray, shape (n,)
    average : str, one of {'macro', 'micro', 'weighted'}, default 'macro'

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    if average == 'micro':
        tp = int(np.sum(y_true == y_pred))
        total = len(y_true)
        return float(tp / total) if total > 0 else 0.0
    classes = np.unique(y_true)
    scores = []
    weights = []
    for cls in classes:
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        denom = tp + fn
        scores.append(float(tp / denom) if denom > 0 else 0.0)
        weights.append(int(np.sum(y_true == cls)))
    if average == 'weighted':
        total = sum(weights)
        return float(np.dot(scores, weights) / total) if total > 0 else 0.0
    return float(np.mean(scores)) if scores else 0.0


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro',
) -> float:
    """
    Compute F1 score as harmonic mean of precision and recall.

    Parameters
    ----------
    y_true  : np.ndarray, shape (n,)
    y_pred  : np.ndarray, shape (n,)
    average : str, default 'macro'

    Returns
    -------
    float
    """
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
    y_pred : np.ndarray, shape (n,)

    Returns
    -------
    np.ndarray, shape (n_classes, n_classes)
        Entry [i, j] = count of true-class-i predicted as class-j.
        Classes are sorted (ascending).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    # Map labels to contiguous 0..n-1 indices
    lookup = np.searchsorted(classes, y_true)
    pred_idx = np.searchsorted(classes, y_pred)
    linear_idx = lookup * n + pred_idx
    cm = np.bincount(linear_idx, minlength=n * n).reshape(n, n)
    return cm.astype(int)


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC-AUC for binary classification using the trapezoidal rule.

    Parameters
    ----------
    y_true  : np.ndarray, shape (n,) – binary labels (0/1)
    y_score : np.ndarray, shape (n,) – predicted scores/probabilities

    Returns
    -------
    float : AUC in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # Sort by descending score; stable sort for tie-breaking
    order = np.argsort(-y_score, kind='stable')
    y_sorted = y_true[order]
    # Vectorised cumulative TP and FP
    tp_cumsum = np.cumsum(y_sorted == 1)
    fp_cumsum = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tp_cumsum / n_pos])
    fpr = np.concatenate([[0.0], fp_cumsum / n_neg])
    trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(abs(trap(tpr, fpr)))
