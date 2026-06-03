"""
benchmarks/benchmark_ensemble.py
---------------------------------
Benchmark DecisionTreeClassifier (base) vs EnsembleClassifier/Bagging vs
RandomForestClassifier under a streaming scenario.

Records per-chunk wall-clock fit time, accuracy, and RSS memory usage,
then prints a summary table and a head-to-head comparison.

Usage
-----
    python benchmarks/benchmark_ensemble.py
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from framework.tree import DecisionTreeClassifier
from framework.ensemble import EnsembleClassifier, RandomForestClassifier
from framework.metrics import accuracy_score
from framework.io import split_into_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_classification(
    n_samples: int = 2000,
    n_features: int = 10,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple:
    """Generate a synthetic classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    scores = X @ weights
    boundaries = np.linspace(scores.min(), scores.max(), n_classes + 1)
    y = np.digitize(scores, boundaries[1:-1])
    return X, y


def _get_rss_mb() -> float:
    """Return current process RSS memory in MB (Linux /proc; fallback 0.0)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def benchmark(clf, X_chunks, y_chunks, classes, name) -> list:
    """
    Stream chunks through clf and record per-chunk metrics.

    Returns
    -------
    list of dict with keys: 'chunk', 'fit_ms', 'accuracy', 'memory_mb'
    """
    records = []
    for i, (Xc, yc) in enumerate(zip(X_chunks, y_chunks)):
        t0 = time.perf_counter()
        clf.partial_fit(Xc, yc, classes=classes)
        fit_ms = (time.perf_counter() - t0) * 1000.0

        acc = accuracy_score(yc, clf.predict(Xc))
        mem = _get_rss_mb()
        records.append({'chunk': i, 'fit_ms': fit_ms, 'accuracy': acc, 'memory_mb': mem})
    return records


def print_table(records, model_name) -> None:
    header = f"{'Chunk':>6} {'Fit (ms)':>10} {'Accuracy':>10} {'RSS (MB)':>10}"
    sep = '-' * len(header)
    print(f"\n=== {model_name} ===")
    print(header)
    print(sep)
    for r in records:
        print(f"{r['chunk']:>6} {r['fit_ms']:>10.2f} {r['accuracy']:>10.4f} {r['memory_mb']:>10.1f}")
    avg_ms  = np.mean([r['fit_ms']     for r in records])
    avg_acc = np.mean([r['accuracy']   for r in records])
    avg_mem = np.mean([r['memory_mb']  for r in records])
    print(sep)
    print(f"{'AVG':>6} {avg_ms:>10.2f} {avg_acc:>10.4f} {avg_mem:>10.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N_SAMPLES    = 2000
    N_FEATURES   = 10
    N_CLASSES    = 2
    N_CHUNKS     = 10
    MAX_DEPTH    = 5
    N_ESTIMATORS = 10
    SEED         = 42

    print(f"Dataset : {N_SAMPLES} samples × {N_FEATURES} features, "
          f"{N_CLASSES} classes")
    print(f"Chunks  : {N_CHUNKS}  |  Estimators (ensemble) : {N_ESTIMATORS}\n")

    X, y = make_classification(N_SAMPLES, N_FEATURES, N_CLASSES, SEED)
    classes = np.unique(y)
    chunks  = split_into_chunks(X, y, n_chunks=N_CHUNKS)
    X_chunks = [c[0] for c in chunks]
    y_chunks = [c[1] for c in chunks]

    # --- Base model: DecisionTree ---
    dt = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=SEED)
    dt_rec = benchmark(dt, X_chunks, y_chunks, classes, 'DecisionTree')
    print_table(dt_rec, 'DecisionTreeClassifier  [base]')

    # --- Ensemble: Bagging ---
    bag = EnsembleClassifier(
        n_estimators=N_ESTIMATORS, method='bagging',
        max_depth=MAX_DEPTH, random_state=SEED,
    )
    bag_rec = benchmark(bag, X_chunks, y_chunks, classes, 'Bagging')
    print_table(bag_rec, f'EnsembleClassifier — Bagging  (n={N_ESTIMATORS})')

    # --- Ensemble: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH, random_state=SEED,
    )
    rf_rec = benchmark(rf, X_chunks, y_chunks, classes, 'RandomForest')
    print_table(rf_rec, f'RandomForestClassifier  (n={N_ESTIMATORS})')

    # --- Head-to-head summary ---
    def _avg(rec, key):
        return np.mean([r[key] for r in rec])

    models = [
        ('DecisionTreeClassifier  [base]',           dt_rec),
        (f'EnsembleClassifier — Bagging (n={N_ESTIMATORS})', bag_rec),
        (f'RandomForestClassifier        (n={N_ESTIMATORS})', rf_rec),
    ]

    print("\n" + "=" * 72)
    print("HEAD-TO-HEAD SUMMARY")
    print("=" * 72)
    print(f"{'Model':<46} {'Avg Acc':>8} {'Avg ms':>8} {'Avg MB':>8}")
    print('-' * 72)
    for name, rec in models:
        print(f"{name:<46} {_avg(rec,'accuracy'):>8.4f} "
              f"{_avg(rec,'fit_ms'):>8.1f} {_avg(rec,'memory_mb'):>8.1f}")

    print()
    best_acc   = max(models, key=lambda m: _avg(m[1], 'accuracy'))
    fastest    = min(models, key=lambda m: _avg(m[1], 'fit_ms'))
    lowest_mem = min(models, key=lambda m: _avg(m[1], 'memory_mb'))
    print(f"  Highest accuracy : {best_acc[0].strip()}")
    print(f"  Fastest fit      : {fastest[0].strip()}")
    print(f"  Lowest memory    : {lowest_mem[0].strip()}")

    # Accuracy gain of best ensemble over base
    dt_acc  = _avg(dt_rec,  'accuracy')
    rf_acc  = _avg(rf_rec,  'accuracy')
    bag_acc = _avg(bag_rec, 'accuracy')
    best_ens_acc = max(rf_acc, bag_acc)
    print(f"\n  Ensemble accuracy gain over base : "
          f"+{(best_ens_acc - dt_acc)*100:.2f} pp")


if __name__ == '__main__':
    main()
