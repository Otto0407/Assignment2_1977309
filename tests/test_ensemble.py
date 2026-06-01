"""
tests/test_ensemble.py
----------------------
Unit tests for framework.ensemble – EnsembleClassifier, RandomForestClassifier.
"""

import unittest
import numpy as np

from framework.ensemble import EnsembleClassifier, RandomForestClassifier


def _make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


class TestEnsembleClassifier(unittest.TestCase):

    def test_fit_returns_self(self):
        X, y = _make_data()
        clf = EnsembleClassifier(n_estimators=3, random_state=0)
        self.assertIs(clf.fit(X, y), clf)

    def test_predict_shape(self):
        X, y = _make_data()
        clf = EnsembleClassifier(n_estimators=3, random_state=1).fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X),))

    def test_predict_proba_shape_and_sum(self):
        X, y = _make_data()
        clf = EnsembleClassifier(n_estimators=5, random_state=2).fit(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)

    def test_score_reasonable(self):
        X, y = _make_data(n=500)
        clf = EnsembleClassifier(n_estimators=5, max_depth=4, random_state=3)
        clf.fit(X, y)
        self.assertGreater(clf.score(X, y), 0.85)

    def test_partial_fit(self):
        X, y = _make_data(n=400)
        clf = EnsembleClassifier(n_estimators=3, random_state=4)
        for chunk_X, chunk_y in zip(np.array_split(X, 4), np.array_split(y, 4)):
            clf.partial_fit(chunk_X, chunk_y, classes=np.array([0, 1]))
        acc = clf.score(X, y)
        self.assertGreater(acc, 0.70)

    def test_estimator_count(self):
        X, y = _make_data()
        clf = EnsembleClassifier(n_estimators=7, random_state=5).fit(X, y)
        self.assertEqual(len(clf.estimators_), 7)


class TestRandomForestClassifier(unittest.TestCase):

    def test_is_ensemble_subclass(self):
        self.assertTrue(issubclass(RandomForestClassifier, EnsembleClassifier))

    def test_max_features_sqrt(self):
        X, y = _make_data()
        clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        for est in clf.estimators_:
            self.assertEqual(est.max_features, 'sqrt')

    def test_predict_and_score(self):
        X, y = _make_data(n=400)
        clf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=1)
        clf.fit(X, y)
        self.assertGreater(clf.score(X, y), 0.80)


if __name__ == '__main__':
    unittest.main()
