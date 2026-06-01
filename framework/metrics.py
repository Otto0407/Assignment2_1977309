"""
framework/metrics.py
--------------------
Streaming and batch classification metrics. Uses only numpy.
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
        """
        Incorporate a new batch of predictions.

        Parameters
        ----------
        y_true : np.ndarray, shape (n,)
        y_pred : np.ndarray, shape (n,)
        """
        raise NotImplementedError("TODO: implement update")

    def result(self) -> float:
        """
        Compute and return the current metric value.

        Returns
        -------
        float
        """
        raise NotImplementedError("TODO: implement result")

    def reset(self) -> None:
        """Reset accumulated state."""
        raise NotImplementedError("TODO: implement reset")


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
        self._window: deque = deque() if window_size is not None else None

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Parameters
        ----------
        y_true : np.ndarray, shape (n,)
        y_pred : np.ndarray, shape (n,)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        hits = (y_true == y_pred).astype(int)
        if self._window is not None:
            for h in hits:
                self._window.append(h)
                if len(self._window) > self.window_size:
                    self._window.popleft()
        else:
            self._correct += hits.sum()
            self._total += len(hits)

    def result(self) -> float:
        """
        Returns
        -------
        float : accuracy in [0, 1]
        """
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
        """
        Parameters
        ----------
        y_true : np.ndarray, shape (n,)
        y_pred : np.ndarray, shape (n,)
        """
        self._y_true_buf.extend(list(np.asarray(y_true)))
        self._y_pred_buf.extend(list(np.asarray(y_pred)))
        if self._window_size is not None:
            self._y_true_buf = self._y_true_buf[-self._window_size:]
            self._y_pred_buf = self._y_pred_buf[-self._window_size:]

    def result(self) -> float:
        """
        Returns
        -------
        float : precision score
        """
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
        """
        Parameters
        ----------
        y_true : np.ndarray, shape (n,)
        y_pred : np.ndarray, shape (n,)
        """
        self._y_true_buf.extend(list(np.asarray(y_true)))
        self._y_pred_buf.extend(list(np.asarray(y_pred)))
        if self._window_size is not None:
            self._y_true_buf = self._y_true_buf[-self._window_size:]
            self._y_pred_buf = self._y_pred_buf[-self._window_size:]

    def result(self) -> float:
        """
        Returns
        -------
        float : recall score
        """
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
        """
        Parameters
        ----------
        y_true : np.ndarray, shape (n,)
        y_pred : np.ndarray, shape (n,)
        """
        self._y_true_buf.extend(list(np.asarray(y_true)))
        self._y_pred_buf.extend(list(np.asarray(y_pred)))
        if self._window_size is not None:
            self._y_true_buf = self._y_true_buf[-self._window_size:]
            self._y_pred_buf = self._y_pred_buf[-self._window_size:]

    def result(self) -> float:
        """
        Returns
        -------
        float : F1 score
        """
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
        self._matrix = np.zeros((n_classes, n_classes), dtype=int)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Parameters
        ----------
        y_true : np.ndarray, shape (n,)
        y_pred : np.ndarray, shape (n,)
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.n_classes and 0 <= p < self.n_classes:
                self._matrix[t, p] += 1

    def result(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray, shape (n_classes, n_classes)
            Accumulated confusion matrix where entry [i, j] is the count
            of samples with true label i predicted as label j.
        """
        return self._matrix.copy()

    def reset(self) -> None:
        self._matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)


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
        """
        Parameters
        ----------
        y_true  : np.ndarray, shape (n,) – binary labels
        y_pred  : np.ndarray, shape (n,) – predicted scores / probabilities
        """
        self._y_true_buf.extend(list(np.asarray(y_true)))
        self._y_score_buf.extend(list(np.asarray(y_pred)))
        if self._window_size is not None:
            self._y_true_buf = self._y_true_buf[-self._window_size:]
            self._y_score_buf = self._y_score_buf[-self._window_size:]

    def result(self) -> float:
        """
        Returns
        -------
        float : ROC-AUC score
        """
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
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == 'micro':
        tp = np.sum(y_true == y_pred)
        fp = np.sum(y_pred != y_true)
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    scores = []
    weights = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        denom = tp + fp
        scores.append(float(tp / denom) if denom > 0 else 0.0)
        weights.append(int(np.sum(y_true == cls)))
    if average == 'weighted':
        total = sum(weights)
        if total == 0:
            return 0.0
        return float(np.dot(scores, weights) / total)
    return float(np.mean(scores))


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
    classes = np.unique(y_true)
    if average == 'micro':
        tp = np.sum(y_true == y_pred)
        fn = np.sum(y_true != y_pred)
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    scores = []
    weights = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        denom = tp + fn
        scores.append(float(tp / denom) if denom > 0 else 0.0)
        weights.append(int(np.sum(y_true == cls)))
    if average == 'weighted':
        total = sum(weights)
        if total == 0:
            return 0.0
        return float(np.dot(scores, weights) / total)
    return float(np.mean(scores))


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
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


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
    # Sort by descending score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    # Trapezoidal integration
    trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc = float(trap(tpr_list, fpr_list))
    return abs(auc)
