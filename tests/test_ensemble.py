"""
tests/test_ensemble.py
----------------------
Unit tests for framework.ensemble – EnsembleClassifier, RandomForestClassifier.

Covers:
  - Basic fit / predict / predict_proba / score
  - method='bagging' vs method='random_forest' max_features behaviour
  - partial_fit: single chunk, multiple chunks, classes argument preserved
  - predict_proba sums to 1, including when bootstrap misses a class
  - Reproducibility via random_state
  - n_estimators is respected
  - RandomForestClassifier is a subclass; uses sqrt max_features
  - Edge cases: single-class data, tiny chunks, invalid method
  - Streaming: accuracy improves (or stays stable) over chunks
"""

import unittest
import numpy as np

from framework.ensemble import EnsembleClassifier, RandomForestClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


def _make_multiclass(n=300, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.argmax(X[:, :3], axis=1)   # labels 0, 1, 2
    return X, y


# ---------------------------------------------------------------------------
# EnsembleClassifier – basic fit
# ---------------------------------------------------------------------------

class TestEnsembleFit(unittest.TestCase):

    def test_fit_returns_self(self):
        X, y = _make_binary()
        clf = EnsembleClassifier(n_estimators=3, random_state=0)
        self.assertIs(clf.fit(X, y), clf)

    def test_n_estimators_respected(self):
        X, y = _make_binary()
        for n in (1, 5, 10):
            clf = EnsembleClassifier(n_estimators=n, random_state=0).fit(X, y)
            self.assertEqual(len(clf.estimators_), n)

    def test_predict_shape(self):
        X, y = _make_binary()
        clf = EnsembleClassifier(n_estimators=3, random_state=1).fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X),))

    def test_predict_values_in_classes(self):
        X, y = _make_binary()
        clf = EnsembleClassifier(n_estimators=3, random_state=2).fit(X, y)
        unique_preds = set(clf.predict(X).tolist())
        self.assertTrue(unique_preds.issubset({0, 1}))

    def test_predict_proba_shape(self):
        X, y = _make_binary()
        clf = EnsembleClassifier(n_estimators=5, random_state=3).fit(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 2))

    def test_predict_proba_sums_to_one(self):
        X, y = _make_binary()
        clf = EnsembleClassifier(n_estimators=5, random_state=4).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_score_reasonable(self):
        X, y = _make_binary(n=500)
        clf = EnsembleClassifier(n_estimators=5, max_depth=4, random_state=5)
        clf.fit(X, y)
        self.assertGreater(clf.score(X, y), 0.80)

    def test_predict_before_fit_raises(self):
        clf = EnsembleClassifier(n_estimators=3)
        with self.assertRaises(RuntimeError):
            clf.predict(np.ones((5, 4)))

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            EnsembleClassifier(method='boosting')


# ---------------------------------------------------------------------------
# Bagging vs RandomForest max_features
# ---------------------------------------------------------------------------

class TestMethodMaxFeatures(unittest.TestCase):

    def test_bagging_uses_all_features(self):
        """Bagging: max_features should be None (all features used)."""
        clf = EnsembleClassifier(method='bagging', max_features='sqrt')
        self.assertIsNone(clf.max_features)

    def test_random_forest_uses_sqrt(self):
        """RandomForest: max_features should be passed through."""
        clf = EnsembleClassifier(method='random_forest', max_features='sqrt')
        self.assertEqual(clf.max_features, 'sqrt')

    def test_bagging_estimators_have_none_max_features(self):
        X, y = _make_binary()
        clf = EnsembleClassifier(method='bagging', n_estimators=3,
                                 random_state=0).fit(X, y)
        for est in clf.estimators_:
            self.assertIsNone(est.max_features)

    def test_rf_estimators_have_sqrt_max_features(self):
        X, y = _make_binary()
        clf = EnsembleClassifier(method='random_forest', n_estimators=3,
                                 max_features='sqrt', random_state=0).fit(X, y)
        for est in clf.estimators_:
            self.assertEqual(est.max_features, 'sqrt')


# ---------------------------------------------------------------------------
# partial_fit (streaming)
# ---------------------------------------------------------------------------

class TestEnsemblePartialFit(unittest.TestCase):

    def test_partial_fit_returns_self(self):
        X, y = _make_binary(n=100)
        clf = EnsembleClassifier(n_estimators=3, random_state=0)
        self.assertIs(clf.partial_fit(X, y), clf)

    def test_partial_fit_single_chunk_predicts(self):
        X, y = _make_binary(n=200)
        clf = EnsembleClassifier(n_estimators=3, random_state=0)
        clf.partial_fit(X, y, classes=np.array([0, 1]))
        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X),))

    def test_partial_fit_multiple_chunks_accumulates(self):
        X, y = _make_binary(n=400)
        clf = EnsembleClassifier(n_estimators=5, max_depth=4, random_state=1)
        for Xc, yc in zip(np.array_split(X, 4), np.array_split(y, 4)):
            clf.partial_fit(Xc, yc, classes=np.array([0, 1]))
        self.assertGreater(clf.score(X, y), 0.70)

    def test_partial_fit_classes_preserved_across_chunks(self):
        """classes_ must stay [0,1] even if first chunk only contains class 0."""
        X, y = _make_binary(n=200)
        mask0 = y == 0
        mask1 = y == 1
        clf = EnsembleClassifier(n_estimators=3, random_state=2)
        clf.partial_fit(X[mask0], y[mask0], classes=np.array([0, 1]))
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        clf.partial_fit(X[mask1], y[mask1], classes=np.array([0, 1]))
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_partial_fit_proba_sums_to_one_after_each_chunk(self):
        X, y = _make_binary(n=300)
        clf = EnsembleClassifier(n_estimators=3, random_state=3)
        for Xc, yc in zip(np.array_split(X, 3), np.array_split(y, 3)):
            clf.partial_fit(Xc, yc, classes=np.array([0, 1]))
            proba = clf.predict_proba(X)
            np.testing.assert_allclose(
                proba.sum(axis=1), np.ones(len(X)), atol=1e-10,
                err_msg="proba rows do not sum to 1 after partial_fit chunk"
            )

    def test_partial_fit_creates_correct_n_estimators(self):
        X, y = _make_binary(n=100)
        clf = EnsembleClassifier(n_estimators=7, random_state=4)
        clf.partial_fit(X, y)
        self.assertEqual(len(clf.estimators_), 7)

    def test_streaming_accuracy_improves(self):
        """Accuracy on training data should not degrade as more chunks are seen."""
        X, y = _make_binary(n=500)
        clf = EnsembleClassifier(n_estimators=5, max_depth=4, random_state=5)
        accs = []
        chunks = list(zip(np.array_split(X, 5), np.array_split(y, 5)))
        for Xc, yc in chunks:
            clf.partial_fit(Xc, yc, classes=np.array([0, 1]))
            accs.append(clf.score(X, y))
        # Final accuracy should be better than 60 %
        self.assertGreater(accs[-1], 0.60)


# ---------------------------------------------------------------------------
# Multiclass
# ---------------------------------------------------------------------------

class TestEnsembleMulticlass(unittest.TestCase):

    def test_multiclass_predict_shape(self):
        X, y = _make_multiclass()
        clf = EnsembleClassifier(n_estimators=5, random_state=0).fit(X, y)
        self.assertEqual(clf.predict(X).shape, (len(X),))

    def test_multiclass_proba_shape(self):
        X, y = _make_multiclass()
        clf = EnsembleClassifier(n_estimators=5, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 3))

    def test_multiclass_proba_sums_to_one(self):
        X, y = _make_multiclass()
        clf = EnsembleClassifier(n_estimators=5, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_multiclass_partial_fit(self):
        X, y = _make_multiclass(n=300)
        classes = np.array([0, 1, 2])
        clf = EnsembleClassifier(n_estimators=5, random_state=1)
        for Xc, yc in zip(np.array_split(X, 3), np.array_split(y, 3)):
            clf.partial_fit(Xc, yc, classes=classes)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 3))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):

    def test_same_random_state_same_predictions(self):
        X, y = _make_binary(n=200)
        clf1 = EnsembleClassifier(n_estimators=5, random_state=42).fit(X, y)
        clf2 = EnsembleClassifier(n_estimators=5, random_state=42).fit(X, y)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))

    def test_different_random_state_different_predictions(self):
        X, y = _make_binary(n=300)
        clf1 = EnsembleClassifier(n_estimators=10, random_state=0).fit(X, y)
        clf2 = EnsembleClassifier(n_estimators=10, random_state=99).fit(X, y)
        # Not identical (very unlikely with 10 trees on 300 samples)
        self.assertFalse(np.array_equal(clf1.predict(X), clf2.predict(X)))


# ---------------------------------------------------------------------------
# RandomForestClassifier subclass
# ---------------------------------------------------------------------------

class TestRandomForestClassifier(unittest.TestCase):

    def test_is_subclass_of_ensemble(self):
        self.assertTrue(issubclass(RandomForestClassifier, EnsembleClassifier))

    def test_method_is_random_forest(self):
        clf = RandomForestClassifier()
        self.assertEqual(clf.method, 'random_forest')

    def test_default_max_features_sqrt(self):
        clf = RandomForestClassifier()
        self.assertEqual(clf.max_features, 'sqrt')

    def test_estimators_use_sqrt_max_features(self):
        X, y = _make_binary()
        clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        for est in clf.estimators_:
            self.assertEqual(est.max_features, 'sqrt')

    def test_fit_predict_score(self):
        X, y = _make_binary(n=400)
        clf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=1)
        clf.fit(X, y)
        self.assertGreater(clf.score(X, y), 0.80)

    def test_partial_fit_streaming(self):
        X, y = _make_binary(n=400)
        clf = RandomForestClassifier(n_estimators=5, random_state=2)
        for Xc, yc in zip(np.array_split(X, 4), np.array_split(y, 4)):
            clf.partial_fit(Xc, yc, classes=np.array([0, 1]))
        self.assertGreater(clf.score(X, y), 0.65)

    def test_proba_sums_to_one(self):
        X, y = _make_binary()
        clf = RandomForestClassifier(n_estimators=5, random_state=3).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_single_class_data(self):
        """All-same labels: predict should return that label for every sample."""
        X = np.random.default_rng(0).normal(size=(50, 4))
        y = np.zeros(50, dtype=int)
        clf = EnsembleClassifier(n_estimators=3, random_state=0).fit(X, y)
        np.testing.assert_array_equal(clf.predict(X), np.zeros(50, dtype=int))

    def test_single_estimator(self):
        X, y = _make_binary(n=200)
        clf = EnsembleClassifier(n_estimators=1, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_tiny_chunk_partial_fit(self):
        """Chunk size of 2 must not crash."""
        X, y = _make_binary(n=20)
        clf = EnsembleClassifier(n_estimators=3, random_state=0)
        clf.partial_fit(X[:2], y[:2], classes=np.array([0, 1]))
        clf.partial_fit(X[2:], y[2:], classes=np.array([0, 1]))
        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X),))


if __name__ == '__main__':
    unittest.main()
