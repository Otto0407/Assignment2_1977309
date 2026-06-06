# Assignment2_1977309 – Streaming ML Framework

A pure-NumPy streaming machine-learning framework implementing online
statistical functions, preprocessing transformers, decision trees, ensembles,
evaluation metrics, and I/O utilities.

## Dependencies

- Python ≥ 3.9
- `numpy`
- `matplotlib`

No sklearn, scipy, or other third-party ML libraries are used.

## Installation

```bash
git clone <repo-url>
cd Assignment2_1977309
pip install numpy matplotlib
```

## Directory layout

```
Assignment2_1977309/
├── framework/
│   ├── __init__.py        # public re-exports
│   ├── stats.py           # StreamStats, chunk_mean/variance/quantile/histogram
│   ├── preprocessing.py   # StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
│   ├── tree.py            # DecisionTreeClassifier
│   ├── ensemble.py        # EnsembleClassifier, RandomForestClassifier
│   ├── metrics.py         # streaming metric classes + batch functions
│   ├── pipeline.py        # Pipeline (chains transformers + estimator)
│   ├── stream.py          # StreamTrainer (orchestrates pipeline + logging)
│   ├── visualise.py       # matplotlib plotting helpers
│   └── io.py              # load_csv, save_csv, stream_csv, split_into_chunks
├── tests/                 # per-module unittest suites (307 tests)
│   ├── test_stats.py
│   ├── test_preprocessing.py
│   ├── test_tree.py
│   ├── test_ensemble.py
│   ├── test_metrics.py
│   ├── test_pipeline.py
│   ├── test_stream.py
│   ├── test_integration_pipeline_metrics.py
│   └── test_visualise.py
├── tests.py               # curated pytest suite (41 tests)
├── conftest.py            # pytest configuration
├── pytest.ini
├── demo/
│   └── stream_demo.ipynb  # end-to-end streaming demo
├── benchmarks/
│   └── benchmark_ensemble.py
└── README.md
```

## Quick start

```python
from framework import (
    Pipeline, StandardScaler, RandomForestClassifier,
    StreamTrainer, Accuracy,
)
from framework.io import load_csv, split_into_chunks
import numpy as np

# Load data from CSV
X_all, headers = load_csv('data.csv', has_header=True)
X = X_all[:, :-1]
y = X_all[:, -1].astype(int)

# Split into chunks to simulate streaming
chunks = split_into_chunks(X, y, n_chunks=10)

# Build pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=10, max_depth=5)),
])

# Incremental training with StreamTrainer
trainer = StreamTrainer(pipe, metrics=[Accuracy()])
for Xc, yc in chunks:
    record = trainer.fit_chunk(Xc, yc)
    print(record)
```

## Module overview

| Module | Key classes / functions |
|---|---|
| `io.py` | `load_csv`, `save_csv`, `stream_csv`, `split_into_chunks` |
| `stats.py` | `StreamStats`, `chunk_mean`, `chunk_variance`, `chunk_quantile`, `chunk_histogram` |
| `preprocessing.py` | `StandardScaler`, `MinMaxScaler`, `Imputer(strategy)`, `OneHotEncoder` |
| `tree.py` | `DecisionTreeClassifier(max_depth, criterion, max_features)` |
| `ensemble.py` | `EnsembleClassifier(method='bagging'/'random_forest')`, `RandomForestClassifier` |
| `metrics.py` | Streaming: `Accuracy`, `F1Score`, `ConfusionMatrix`, `AUC` · Batch: `accuracy_score`, `f1_score`, `confusion_matrix`, `roc_auc_score` |
| `pipeline.py` | `Pipeline(steps)` — supports `fit`, `partial_fit`, `predict`, `predict_proba`, `score` |
| `stream.py` | `StreamTrainer(pipeline, metrics)` — `fit_chunk`, `score_chunk`, `get_log`, `reset` |
| `visualise.py` | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth` |

All preprocessing transformers and estimators expose `partial_fit` for
incremental learning and accept `NaN` values where noted.

## Running the tests

**Curated pytest suite (recommended):**

```bash
pytest tests.py -v
```

**Full per-module unittest suite:**

```bash
python -m unittest discover tests/
```

## Running the demo

```bash
jupyter notebook demo/stream_demo.ipynb
```

The notebook walks through all four core requirements: loading a CSV,
splitting into chunks, incremental `partial_fit` training, and visualising
metrics with `visualise.py`.

## Running the benchmark

```bash
python benchmarks/benchmark_ensemble.py
```

Sample output (2000 samples, 10 features, 10 estimators):

```
Model                                  Avg Acc   Avg ms   Avg MB
---------------------------------------------------------------
DecisionTreeClassifier  [base]          0.8885   2257ms     65.0
EnsembleClassifier — Bagging (n=10)     0.9115  14726ms     66.9
RandomForestClassifier  (n=10)          0.9200   4522ms     69.2

Highest accuracy : RandomForestClassifier
Fastest fit      : DecisionTreeClassifier
Ensemble accuracy gain over base: +3.15 pp
```

RandomForest achieves the highest accuracy with a reasonable fit time,
while the base DecisionTree is fastest and most memory-efficient.

## Design notes

**`partial_fit` is accumulative, not purely online.**
Each call appends the new chunk to an internal buffer and rebuilds the
model from scratch on all accumulated data. This guarantees the model
always reflects the best possible fit over everything seen so far, at
the cost of fit time growing linearly with the number of chunks. This
is the intended trade-off for this framework.

**Streaming metrics track cumulative state.**
`Accuracy`, `F1Score`, and other streaming metric classes aggregate
predictions across *all* chunks seen since the last `reset()`. Their
`result()` reflects overall performance, not the instantaneous score
on the most recent chunk. Use the batch functions (`accuracy_score`,
`f1_score`, etc.) if you need a per-chunk snapshot instead.

## Module inter-dependencies

```
io            <- (no internal imports)
stats         <- (no internal imports)
preprocessing <- stats
tree          <- (no internal imports)
ensemble      <- tree
metrics       <- (no internal imports)
pipeline      <- (no internal imports)
stream        <- pipeline, metrics
visualise     <- (no internal imports)
__init__      <- all of the above
```
