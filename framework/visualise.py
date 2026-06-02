"""
framework/visualise.py
----------------------
Plotting utilities using only matplotlib (no sklearn, no scipy).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(
    metric_values: list,
    title: str = '',
    ylabel: str = '',
    save_path: str = None,
) -> None:
    """
    Plot a single metric as a line chart over streaming chunks.

    Parameters
    ----------
    metric_values : list of float
        One value per chunk (e.g., accuracy per chunk).
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    save_path : str or None
        If given, save the figure to this path instead of calling plt.show().
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    chunks = list(range(len(metric_values)))
    ax.plot(chunks, metric_values, marker='o', linewidth=1.5, markersize=4)
    ax.set_xlabel('Chunk')
    ax.set_ylabel(ylabel if ylabel else 'Metric value')
    ax.set_title(title if title else 'Metric over time')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()


def compare_models(
    metric1: list,
    metric2: list,
    labels: list,
    title: str = '',
    ylabel: str = '',
    save_path: str = None,
) -> None:
    """
    Overlay two model metric traces on the same axes.

    Parameters
    ----------
    metric1 : list of float
        Per-chunk metric for model 1.
    metric2 : list of float
        Per-chunk metric for model 2. Must have the same length as metric1.
    labels : list of str
        [label_for_model1, label_for_model2]
    title : str, optional
    ylabel : str, optional
    save_path : str or None
        Save figure if given, else plt.show().
    """
    if len(metric1) != len(metric2):
        raise ValueError("metric1 and metric2 must have the same length.")
    if len(labels) < 2:
        raise ValueError("labels must contain at least two entries.")

    chunks = list(range(len(metric1)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chunks, metric1, marker='o', linewidth=1.5, markersize=4,
            label=labels[0])
    ax.plot(chunks, metric2, marker='s', linewidth=1.5, markersize=4,
            linestyle='--', label=labels[1])
    ax.set_xlabel('Chunk')
    ax.set_ylabel(ylabel if ylabel else 'Metric value')
    ax.set_title(title if title else 'Model comparison')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()


def plot_predictions_vs_ground_truth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = '',
    save_path: str = None,
) -> None:
    """
    Scatter plot comparing ground-truth vs predicted labels for one chunk.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        True labels.
    y_pred : np.ndarray, shape (n,)
        Predicted labels.
    title : str, optional
    save_path : str or None
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}."
        )
    n = len(y_true)
    indices = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(indices, y_true, label='Ground truth', alpha=0.7, s=20,
               marker='o')
    ax.scatter(indices, y_pred, label='Predictions', alpha=0.7, s=20,
               marker='x')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Class label')
    ax.set_title(title if title else 'Predictions vs Ground Truth')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    save_path: str = None,
) -> None:
    """
    Display a confusion matrix as a colour-coded heatmap.

    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
        Confusion matrix where entry [i, j] is the count for true class i
        predicted as class j.
    class_names : list of str or None
        Tick labels for each class. If None, uses '0', '1', ...
    save_path : str or None
    """
    cm = np.asarray(cm)
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    if len(class_names) != n:
        raise ValueError(
            f"class_names has {len(class_names)} entries but confusion matrix "
            f"is {n}×{n}."
        )

    fig, ax = plt.subplots(figsize=(max(4, n), max(3, n)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()
