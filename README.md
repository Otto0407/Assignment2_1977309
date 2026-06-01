# Assignment2_1977309 – Streaming ML Framework

A pure-numpy streaming machine-learning framework implementing online
statistical functions, preprocessing transformers, decision trees, ensembles,
evaluation metrics, and I/O utilities.

## Dependencies

- Python ≥ 3.9
- `numpy`
- `matplotlib`

No sklearn, scipy, or other third-party ML libraries are used.

## Directory layout

```
Assignment2_1977309/
├── framework/
│   ├── __init__.py        # public re-exports
│   ├── stats.py           # StreamStats, chunk_mean/variance/quantile/histogram, welford_update
│   ├── preprocessing.py   # StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
│   ├── tree.py            # DecisionTreeClassifier, _Node
│   ├── ensemble.py        # EnsembleClassifier, RandomForestClassifier
│   ├── metrics.py         # Streaming metric classes + standalone functions
│   ├── pipeline.py        # Pipeline (chains transformers + estimator)
│   ├── stream.py          # StreamTrainer (orchestrates pipeline + logging)
│   ├── visualise.py       # Matplotlib plotting helpers
│   └── io.py              # load_csv, save_csv, stream_csv, train_test_split, split_into_chunks
├── tests/
│   ├── __init__.py
│   ├── test_stats.py
│   ├── test_preprocessing.py
│   ├── test_tree.py
│   ├── test_ensemble.py
│   ├── test_metrics.py
│   ├── test_pipeline.py
│   └── test_stream.py
├── demo/
│   └── stream_demo.ipynb  # End-to-end streaming demo
├── benchmarks/
│   └── benchmark_ensemble.py
└── README.md
```

## Quick start

```python
from framework import (
    Pipeline, StandardScaler, RandomForestClassifier,
    StreamTrainer, Accuracy, split_into_chunks,
)
import numpy as np

# Synthetic data
rng = np.random.default_rng(0)
X = rng.normal(size=(500, 4))
y = (X[:, 0] > 0).astype(int)

# Build pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=10, max_depth=5)),
])

# Stream training
trainer = StreamTrainer(pipe, metrics=[Accuracy()])
for Xc, yc in split_into_chunks(X, y, n_chunks=5):
    record = trainer.fit_chunk(Xc, yc)
    print(record)
```

## Running the tests

```bash
python -m unittest discover tests/
```

## Running the benchmark

```bash
python benchmarks/benchmark_ensemble.py
```

## Module inter-dependencies

```
io          <- (no internal imports)
stats       <- (no internal imports)
preprocessing <- stats (welford_update)
tree        <- (no internal imports)
ensemble    <- tree
metrics     <- (no internal imports)
pipeline    <- metrics
stream      <- pipeline, metrics
visualise   <- (no internal imports)
__init__    <- all of the above
```
