"""
tests/test_pipeline.py
----------------------
Unit tests for framework.pipeline – Pipeline chaining transformers + estimator.

Covers:
  - Basic fit / predict / predict_proba / score
  - partial_fit: single chunk, multiple chunks
  - transform() applies only transformer steps
  - __getitem__ access, KeyError on missing name
  - Error: empty steps, duplicate names, predict/transform before fit
  - Multiple transformers in sequence
  - Imputer inside pipeline (NaN handling end-to-end)
  - Method chaining (fit/partial_fit return self)
  - Streaming: accuracy improves over chunks
  - Transformer without partial_fit in partial_fit pipeline
  - Single-step pipeline (estimator only)
  - predict_proba probabilities sum to 1
"""

import unittest
import numpy as np

from framework.pipeline import Pipeline
from framework.preprocessing import StandardScaler, MinMaxScaler, Imputer
from framework.tree import DecisionTreeClassifier
from framework.ensemble import RandomForestClassifier


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=10.0, scale=3.0, size=(n, 4))
    y = (X[:, 0] + X[:, 2] > 20).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestPipelineConstruction(unittest.TestCase):

    def test_empty_steps_raises(self):
        with self.assertRaises(ValueError):
            Pipeline([])

    def test_duplicate_names_raises(self):
        with self.assertRaises(ValueError):
            Pipeline([
                ('scaler', StandardScaler()),
                ('scaler', StandardScaler()),
            ])

    def test_getitem_returns_step(self):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        self.assertIsInstance(pipe['sc'], StandardScaler)
        self.assertIsInstance(pipe['clf'], DecisionTreeClassifier)

    def test_getitem_missing_raises(self):
        pipe = Pipeline([('sc', StandardScaler()), ('clf', DecisionTreeClassifier())])
        with self.assertRaises(KeyError):
            _ = pipe['nonexistent']


# ---------------------------------------------------------------------------
# fit / predict / score
# ---------------------------------------------------------------------------

class TestPipelineFit(unittest.TestCase):

    def _pipe(self, seed=0):
        return Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=4, random_state=seed)),
        ])

    def test_fit_returns_self(self):
        X, y = _make_data()
        pipe = self._pipe()
        self.assertIs(pipe.fit(X, y), pipe)

    def test_fit_predict_shape(self):
        X, y = _make_data()
        pipe = self._pipe()
        pipe.fit(X, y)
        self.assertEqual(pipe.predict(X).shape, (len(X),))

    def test_fit_score_reasonable(self):
        X, y = _make_data(n=300)
        pipe = self._pipe()
        pipe.fit(X, y)
        self.assertGreater(pipe.score(X, y), 0.75)
        self.assertLessEqual(pipe.score(X, y), 1.0)

    def test_score_consistent_with_accuracy(self):
        from framework.metrics import accuracy_score
        X, y = _make_data()
        pipe = self._pipe()
        pipe.fit(X, y)
        self.assertAlmostEqual(pipe.score(X, y), accuracy_score(y, pipe.predict(X)))

    def test_predict_before_fit_raises(self):
        pipe = self._pipe()
        X, _ = _make_data()
        with self.assertRaises(Exception):
            pipe.predict(X)

    def test_transform_before_fit_raises(self):
        pipe = self._pipe()
        X, _ = _make_data()
        with self.assertRaises(Exception):
            pipe.transform(X)

    def test_fit_transformer_fitted_after_fit(self):
        """After fit(), scaler must have computed statistics."""
        X, y = _make_data()
        pipe = self._pipe()
        pipe.fit(X, y)
        self.assertIsNotNone(pipe['sc']._mean)


# ---------------------------------------------------------------------------
# transform() — transformer-only pass
# ---------------------------------------------------------------------------

class TestPipelineTransform(unittest.TestCase):

    def test_transform_shape_unchanged_by_scaler(self):
        X, y = _make_data()
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        pipe.fit(X, y)
        Xt = pipe.transform(X)
        self.assertEqual(Xt.shape, X.shape)

    def test_transform_does_not_call_estimator(self):
        """transform() must stop before the final estimator."""
        X, y = _make_data()
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        pipe.fit(X, y)
        Xt = pipe.transform(X)
        # StandardScaler produces zero-mean output
        np.testing.assert_allclose(Xt.mean(axis=0), np.zeros(4), atol=1e-10)

    def test_multiple_transformers_chain(self):
        X, y = _make_data()
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('mm', MinMaxScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        pipe.fit(X, y)
        Xt = pipe.transform(X)
        # After StandardScaler + MinMaxScaler, values should be in [0, 1]
        self.assertGreaterEqual(Xt.min(), -0.01)
        self.assertLessEqual(Xt.max(), 1.01)


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

class TestPipelinePredictProba(unittest.TestCase):

    def test_proba_shape(self):
        X, y = _make_data()
        pipe = Pipeline([
            ('mm', MinMaxScaler()),
            ('rf', RandomForestClassifier(n_estimators=3, random_state=0)),
        ])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        self.assertEqual(proba.shape[0], len(X))
        self.assertEqual(proba.shape[1], 2)

    def test_proba_sums_to_one(self):
        X, y = _make_data()
        pipe = Pipeline([
            ('mm', MinMaxScaler()),
            ('rf', RandomForestClassifier(n_estimators=5, random_state=0)),
        ])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)


# ---------------------------------------------------------------------------
# partial_fit (streaming)
# ---------------------------------------------------------------------------

class TestPipelinePartialFit(unittest.TestCase):

    def test_partial_fit_returns_self(self):
        X, y = _make_data(n=100)
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        self.assertIs(pipe.partial_fit(X, y), pipe)

    def test_partial_fit_single_chunk(self):
        X, y = _make_data()
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        pipe.partial_fit(X, y)
        preds = pipe.predict(X)
        self.assertEqual(preds.shape, (len(X),))

    def test_partial_fit_multiple_chunks_accuracy(self):
        X, y = _make_data(n=400)
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=3, random_state=1)),
        ])
        classes = np.array([0, 1])
        for Xc, yc in zip(np.array_split(X, 4), np.array_split(y, 4)):
            pipe.partial_fit(Xc, yc, classes=classes)
        self.assertGreater(pipe.score(X, y), 0.70)

    def test_partial_fit_scaler_updated_each_chunk(self):
        """Scaler running mean must change after each partial_fit call."""
        X, y = _make_data(n=200)
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        pipe.partial_fit(X[:100], y[:100])
        mean_after_1 = pipe['sc']._mean.copy()
        pipe.partial_fit(X[100:], y[100:])
        mean_after_2 = pipe['sc']._mean.copy()
        self.assertFalse(np.allclose(mean_after_1, mean_after_2))

    def test_partial_fit_streaming_with_rf(self):
        X, y = _make_data(n=400)
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=5, random_state=2)),
        ])
        classes = np.array([0, 1])
        for Xc, yc in zip(np.array_split(X, 4), np.array_split(y, 4)):
            pipe.partial_fit(Xc, yc, classes=classes)
        self.assertGreater(pipe.score(X, y), 0.65)

    def test_partial_fit_classes_param_forwarded(self):
        """classes kwarg must reach the estimator's partial_fit."""
        X, y = _make_data(n=100)
        # Use only class-0 samples; classes=[0,1] must still be stored
        mask = y == 0
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        pipe.partial_fit(X[mask], y[mask], classes=np.array([0, 1]))
        np.testing.assert_array_equal(pipe['clf'].classes_, [0, 1])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases(unittest.TestCase):

    def test_single_step_estimator_only(self):
        """A pipeline with only one step (the estimator) must work."""
        X, y = _make_data()
        pipe = Pipeline([('clf', DecisionTreeClassifier(random_state=0))])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        self.assertEqual(preds.shape, (len(X),))

    def test_imputer_in_pipeline_fills_nan(self):
        """Imputer as transformer must fill NaN before classifier sees data."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(100, 4))
        y = (X[:, 0] > 0).astype(int)
        # Introduce NaN into X
        X_nan = X.copy()
        X_nan[rng.random(size=X.shape) < 0.15] = np.nan

        pipe = Pipeline([
            ('imp', Imputer(strategy='mean')),
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        pipe.fit(X_nan, y)
        preds = pipe.predict(X_nan)
        # Should not crash; predictions shape correct
        self.assertEqual(preds.shape, (len(X_nan),))
        self.assertFalse(np.any(np.isnan(preds.astype(float))))

    def test_transformer_without_partial_fit_in_partial_fit_pipe(self):
        """If a transformer has no partial_fit, partial_fit() must still
        call its transform() so data flows through correctly."""

        class DoubleTransformer:
            """Simple transformer with no partial_fit."""
            def transform(self, X):
                return X * 2.0

        pipe = Pipeline([
            ('dbl', DoubleTransformer()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        X, y = _make_data(n=60)
        pipe.partial_fit(X, y)
        preds = pipe.predict(X)
        self.assertEqual(preds.shape, (len(X),))

    def test_predict_values_in_classes(self):
        X, y = _make_data()
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        self.assertTrue(set(preds.tolist()).issubset({0, 1}))


if __name__ == '__main__':
    unittest.main()
