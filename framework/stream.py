"""
framework/stream.py
-------------------
StreamTrainer orchestrates a Pipeline over streaming data chunks,
logging per-chunk metrics and (optionally) memory usage. Uses only numpy.
"""

from __future__ import annotations

import os
import numpy as np
from framework.pipeline import Pipeline
from framework.metrics import StreamingMetric, Accuracy


class StreamTrainer:
    """
    Orchestrates streaming training using a Pipeline and a set of
    StreamingMetric instances.

    Parameters
    ----------
    pipeline : Pipeline
        A fully-constructed Pipeline instance (transformers + estimator).
    metrics : list of StreamingMetric, optional
        Metrics to evaluate after each chunk. Defaults to [Accuracy()].
    log_memory : bool, default True
        If True, record the process RSS memory (in MB) after each chunk.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        metrics: list = None,
        log_memory: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.metrics: list = metrics if metrics is not None else [Accuracy()]
        self.log_memory = log_memory
        self._log: list = []
        self._chunk_idx: int = 0

    # ------------------------------------------------------------------
    def fit_chunk(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Incrementally train the pipeline on one chunk then evaluate.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Feature chunk.
        y : np.ndarray, shape (n,)
            Label chunk.

        Returns
        -------
        dict
            Keys: 'chunk' (int), one key per metric name, optionally
            'memory_mb' (float).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.pipeline.partial_fit(X, y)
        return self._evaluate(X, y, fit=True)

    # ------------------------------------------------------------------
    def score_chunk(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Score the pipeline on a chunk WITHOUT updating the model.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)

        Returns
        -------
        dict
            Same structure as fit_chunk output.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        return self._evaluate(X, y, fit=False)

    # ------------------------------------------------------------------
    def _evaluate(self, X: np.ndarray, y: np.ndarray, fit: bool) -> dict:
        """Internal helper: compute metrics and build log entry."""
        y_pred = self.pipeline.predict(X)

        record: dict = {'chunk': self._chunk_idx}

        for metric in self.metrics:
            metric.update(y, y_pred)
            name = type(metric).__name__.lower()
            record[name] = metric.result()

        if self.log_memory:
            record['memory_mb'] = self._get_memory_mb()

        if fit:
            self._chunk_idx += 1
            self._log.append(record)

        return record

    # ------------------------------------------------------------------
    @staticmethod
    def _get_memory_mb() -> float:
        """
        Estimate current process RSS memory in MB.
        Falls back to 0.0 if /proc/self/status is unavailable.
        """
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        kb = int(line.split()[1])
                        return kb / 1024.0
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    def get_log(self) -> list:
        """
        Return all per-chunk metric records accumulated so far.

        Returns
        -------
        list of dict
            Each dict has keys: 'chunk', metric names, and optionally
            'memory_mb'.
        """
        return list(self._log)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset chunk counter, log, and all metric states."""
        self._log = []
        self._chunk_idx = 0
        for metric in self.metrics:
            metric.reset()
