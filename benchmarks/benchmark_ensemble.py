"""
benchmarks/benchmark_ensemble.py
---------------------------------
Benchmark DecisionTreeClassifier vs RandomForestClassifier on streaming
classification data. Records per-chunk wall-clock time and accuracy, then
prints a summary table.

Usage
-----
    python benchmarks/benchmark_ensemble.py
"""

import time
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from framework.tree import DecisionTreeClassifier
from framework.ensemble import RandomForestClassifier
from framework.metrics import accuracy_score
from framework.io import split_into_chunks


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def make_classification(
    n_samples: int = 2000,
    n_features: int = 10,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple:
    """
    Generate a synthetic classification dataset.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) – class labels in {0, ..., n_classes-1}
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    # Use first two features to define class boundaries
    weights = rng.normal(size=n_features)
    scores = X @ weights
    boundaries = np.linspace(scores.min(), scores.max(), n_classes + 1)
    y = np.digitize(scores, boundaries[1:-1])  # labels 0..n_classes-1
    return X, y


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(
    clf,
    X_chunks: list,
    y_chunks: list,
    classes: np.ndarray,
    name: str,
) -> list:
    """
    Stream chunks through `clf` and record (chunk_id, fit_time_ms, accuracy).

    Returns
    -------
    list of dict with keys: 'chunk', 'fit_ms', 'accuracy'
    """
    records = []
    for i, (Xc, yc) in enumerate(zip(X_chunks, y_chunks)):
        t0 = time.perf_counter()
        clf.partial_fit(Xc, yc, classes=classes)
        t1 = time.perf_counter()
        fit_ms = (t1 - t0) * 1000.0

        y_pred = clf.predict(Xc)
        acc = accuracy_score(yc, y_pred)
        records.append({'chunk': i, 'fit_ms': fit_ms, 'accuracy': acc})
    return records


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------

def print_summary(records: list, model_name: str) -> None:
    header = f"{'Chunk':>6} {'Fit (ms)':>10} {'Accuracy':>10}"
    sep = '-' * len(header)
    print(f"\n=== {model_name} ===")
    print(header)
    print(sep)
    for r in records:
        print(f"{r['chunk']:>6} {r['fit_ms']:>10.2f} {r['accuracy']:>10.4f}")
    avg_ms = np.mean([r['fit_ms'] for r in records])
    avg_acc = np.mean([r['accuracy'] for r in records])
    print(sep)
    print(f"{'AVG':>6} {avg_ms:>10.2f} {avg_acc:>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N_SAMPLES = 2000
    N_FEATURES = 10
    N_CLASSES = 2
    N_CHUNKS = 10
    CHUNK_SIZE = N_SAMPLES // N_CHUNKS
    MAX_DEPTH = 5
    N_ESTIMATORS = 10
    SEED = 42

    print(f"Generating {N_SAMPLES} samples, {N_FEATURES} features, "
          f"{N_CLASSES} classes ...")
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        seed=SEED,
    )
    classes = np.unique(y)
    chunks = split_into_chunks(X, y, n_chunks=N_CHUNKS)
    X_chunks = [c[0] for c in chunks]
    y_chunks = [c[1] for c in chunks]

    print(f"Split into {N_CHUNKS} chunks of ~{CHUNK_SIZE} samples each.\n")

    # --- DecisionTreeClassifier ---
    dt = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=SEED)
    dt_records = benchmark(dt, X_chunks, y_chunks, classes, 'DecisionTree')
    print_summary(dt_records, 'DecisionTreeClassifier')

    # --- RandomForestClassifier ---
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=SEED,
    )
    rf_records = benchmark(rf, X_chunks, y_chunks, classes, 'RandomForest')
    print_summary(rf_records, f'RandomForestClassifier (n={N_ESTIMATORS})')

    # --- Head-to-head comparison ---
    dt_avg_acc = np.mean([r['accuracy'] for r in dt_records])
    rf_avg_acc = np.mean([r['accuracy'] for r in rf_records])
    dt_avg_ms = np.mean([r['fit_ms'] for r in dt_records])
    rf_avg_ms = np.mean([r['fit_ms'] for r in rf_records])

    print("\n=== HEAD-TO-HEAD SUMMARY ===")
    print(f"{'Model':<35} {'Avg Accuracy':>14} {'Avg Fit (ms)':>14}")
    print('-' * 65)
    print(f"{'DecisionTreeClassifier':<35} {dt_avg_acc:>14.4f} {dt_avg_ms:>14.2f}")
    print(f"{'RandomForestClassifier':<35} {rf_avg_acc:>14.4f} {rf_avg_ms:>14.2f}")
    winner = 'RandomForest' if rf_avg_acc > dt_avg_acc else 'DecisionTree'
    print(f"\nHigher accuracy: {winner}")


if __name__ == '__main__':
    main()
