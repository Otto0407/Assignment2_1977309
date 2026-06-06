"""
framework/__init__.py
---------------------
Public API for the streaming ML framework.

All major classes and functions are importable directly from `framework`.

Example
-------
>>> from framework import Pipeline, RandomForestClassifier, StandardScaler
>>> from framework import StreamTrainer, Accuracy
"""

# Stats
from framework.stats import (
    StreamStats,
    chunk_mean,
    chunk_variance,
    chunk_quantile,
    chunk_histogram,
    welford_update,
)

# Preprocessing
from framework.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder,
)

# Tree
from framework.tree import (
    DecisionTreeClassifier,
    _Node,
)

# Ensemble
from framework.ensemble import (
    EnsembleClassifier,
    RandomForestClassifier,
)

# Metrics – classes
from framework.metrics import (
    StreamingMetric,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    AUC,
    # Standalone functions
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# Pipeline
from framework.pipeline import Pipeline

# StreamTrainer
from framework.stream import StreamTrainer

# Visualisation
from framework.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)

# I/O
from framework.io import (
    load_csv,
    save_csv,
    stream_csv,
    split_into_chunks,
)

__all__ = [
    # stats
    "StreamStats",
    "chunk_mean",
    "chunk_variance",
    "chunk_quantile",
    "chunk_histogram",
    "welford_update",
    # preprocessing
    "StandardScaler",
    "MinMaxScaler",
    "Imputer",
    "OneHotEncoder",
    # tree
    "DecisionTreeClassifier",
    "_Node",
    # ensemble
    "EnsembleClassifier",
    "RandomForestClassifier",
    # metrics classes
    "StreamingMetric",
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "ConfusionMatrix",
    "AUC",
    # metrics functions
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "roc_auc_score",
    # pipeline
    "Pipeline",
    # stream
    "StreamTrainer",
    # visualise
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
    # io
    "load_csv",
    "save_csv",
    "stream_csv",
    "split_into_chunks",
]
