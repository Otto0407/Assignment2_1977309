"""
tests/test_pipeline.py
----------------------
Unit tests for framework.pipeline – Pipeline chaining transformers + estimator.
"""

import unittest
import numpy as np

from framework.pipeline import Pipeline
from framework.preprocessing import StandardScaler, MinMaxScaler
from framework.tree import DecisionTreeClassifier
from framework.ensemble import RandomForestClassifier


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=10.0, scale=3.0, size=(n, 4))
    y = (X[:, 0] + X[:, 2] > 20).astype(int)
    return X, y


class TestPipelineBasics(unittest.TestCase):

    def _make_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=4, random_state=0)),
        ])

    def test_fit_then_predict(self):
        X, y = _make_data()
        pipe = self._make_pipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        self.assertEqual(preds.shape, (len(X),))

    def test_partial_fit_then_predict(self):
        X, y = _make_data()
        pipe = self._make_pipeline()
        for chunk in np.array_split(X, 4):
            idx = np.where(np.isin(y, [0, 1]))[0]
            pass  # simple partial_fit over all data
        pipe.partial_fit(X, y)
        preds = pipe.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_score(self):
        X, y = _make_data(n=300)
        pipe = self._make_pipeline()
        pipe.fit(X, y)
        score = pipe.score(X, y)
        self.assertGreater(score, 0.75)
        self.assertLessEqual(score, 1.0)

    def test_getitem_step(self):
        pipe = self._make_pipeline()
        scaler = pipe['scaler']
        self.assertIsInstance(scaler, StandardScaler)

    def test_getitem_missing_key_raises(self):
        pipe = self._make_pipeline()
        with self.assertRaises(KeyError):
            _ = pipe['nonexistent']

    def test_transform_applies_only_transformers(self):
        X, y = _make_data()
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=0)),
        ])
        pipe.fit(X, y)
        Xt = pipe.transform(X)
        # Should have same shape (StandardScaler doesn't change column count)
        self.assertEqual(Xt.shape, X.shape)

    def test_predict_proba(self):
        X, y = _make_data()
        pipe = Pipeline([
            ('mm', MinMaxScaler()),
            ('rf', RandomForestClassifier(n_estimators=3, random_state=0)),
        ])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        self.assertEqual(proba.shape[0], len(X))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)

    def test_empty_steps_raises(self):
        with self.assertRaises(ValueError):
            Pipeline([])

    def test_duplicate_names_raises(self):
        with self.assertRaises(ValueError):
            Pipeline([
                ('scaler', StandardScaler()),
                ('scaler', StandardScaler()),
            ])

    def test_incremental_pipeline_multiple_chunks(self):
        X, y = _make_data(n=400)
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=3, random_state=1)),
        ])
        classes = np.array([0, 1])
        for Xc, yc in zip(np.array_split(X, 4), np.array_split(y, 4)):
            pipe.partial_fit(Xc, yc, classes=classes)
        acc = pipe.score(X, y)
        self.assertGreater(acc, 0.70)


if __name__ == '__main__':
    unittest.main()
