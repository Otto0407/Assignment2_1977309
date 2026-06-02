"""
tests/test_tree.py
------------------
Unit tests for framework.tree – DecisionTreeClassifier.

Covers:
  - Basic fit / predict / predict_proba
  - Depth limit enforcement
  - min_samples_split enforcement
  - max_features: None, 'sqrt', 'log2', int
  - criterion: gini vs entropy (both train correctly)
  - partial_fit accumulation and classes argument
  - Edge cases: single class, two samples, zero-variance feature
  - Numerical: gini/entropy values, proba sums to 1
  - Error handling: predict before fit
"""

import unittest
import numpy as np

from framework.tree import DecisionTreeClassifier, _Node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _make_multiclass_data(n=300, seed=1):
    """3-class data: label = argmax of first 3 features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.argmax(X[:, :3], axis=1)   # labels 0, 1, 2
    return X, y


# ---------------------------------------------------------------------------
# Basic fit / predict
# ---------------------------------------------------------------------------

class TestDecisionTreeFitPredict(unittest.TestCase):

    def test_fit_returns_self(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3)
        self.assertIs(clf.fit(X, y), clf)

    def test_predict_shape(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        self.assertEqual(clf.predict(X).shape, (len(X),))

    def test_predict_values_are_known_classes(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        unique_preds = set(clf.predict(X).tolist())
        self.assertTrue(unique_preds.issubset({0, 1}))

    def test_predict_proba_shape(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 2))

    def test_predict_proba_sums_to_one(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_training_accuracy_gini(self):
        """depth-5 gini tree should reach >85 % training accuracy."""
        X, y = _make_binary_data(n=500)
        clf = DecisionTreeClassifier(max_depth=5, criterion='gini').fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        self.assertGreater(acc, 0.85)

    def test_training_accuracy_entropy(self):
        """depth-5 entropy tree should also reach >85 % training accuracy."""
        X, y = _make_binary_data(n=500)
        clf = DecisionTreeClassifier(max_depth=5, criterion='entropy').fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        self.assertGreater(acc, 0.85)

    def test_multiclass_predict(self):
        X, y = _make_multiclass_data()
        clf = DecisionTreeClassifier(max_depth=5).fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X),))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1, 2}))

    def test_multiclass_predict_proba_shape(self):
        X, y = _make_multiclass_data()
        clf = DecisionTreeClassifier(max_depth=5).fit(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 3))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_predict_before_fit_raises(self):
        clf = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict(np.ones((3, 2)))

    def test_predict_proba_before_fit_raises(self):
        clf = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict_proba(np.ones((3, 2)))


# ---------------------------------------------------------------------------
# Depth / split constraints
# ---------------------------------------------------------------------------

class TestTreeConstraints(unittest.TestCase):

    def test_max_depth_1_has_at_most_two_leaves(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=1).fit(X, y)
        root = clf.tree_
        if not root.is_leaf:
            self.assertTrue(root.left.is_leaf)
            self.assertTrue(root.right.is_leaf)

    def test_max_depth_0_is_leaf(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=0).fit(X, y)
        self.assertTrue(clf.tree_.is_leaf)

    def test_min_samples_split_large_forces_leaf(self):
        """If min_samples_split > n, root must be a leaf."""
        X, y = _make_binary_data(n=50)
        clf = DecisionTreeClassifier(min_samples_split=1000).fit(X, y)
        self.assertTrue(clf.tree_.is_leaf)

    def test_min_samples_split_2_allows_splits(self):
        X, y = _make_binary_data(n=100)
        clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2).fit(X, y)
        self.assertFalse(clf.tree_.is_leaf)


# ---------------------------------------------------------------------------
# max_features variants
# ---------------------------------------------------------------------------

class TestMaxFeatures(unittest.TestCase):

    def _check_trains(self, max_features):
        X, y = _make_binary_data(n=300, seed=7)
        clf = DecisionTreeClassifier(
            max_depth=4, max_features=max_features, random_state=0
        ).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        self.assertGreater(acc, 0.75, msg=f"max_features={max_features!r} acc={acc:.3f}")

    def test_max_features_none(self):
        self._check_trains(None)

    def test_max_features_sqrt(self):
        self._check_trains('sqrt')

    def test_max_features_log2(self):
        self._check_trains('log2')

    def test_max_features_int(self):
        self._check_trains(2)

    def test_max_features_invalid_raises(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_features='invalid')
        with self.assertRaises(ValueError):
            clf.fit(X, y)


# ---------------------------------------------------------------------------
# partial_fit (streaming)
# ---------------------------------------------------------------------------

class TestPartialFit(unittest.TestCase):

    def test_partial_fit_accumulates_data(self):
        X, y = _make_binary_data(n=300)
        clf = DecisionTreeClassifier(max_depth=4)
        clf.partial_fit(X[:150], y[:150])
        clf.partial_fit(X[150:], y[150:])
        # After seeing all data, should match a full fit
        acc = np.mean(clf.predict(X) == y)
        self.assertGreater(acc, 0.80)

    def test_partial_fit_classes_argument_preserved(self):
        """classes kwarg must not be overwritten by np.unique(y_chunk)."""
        X, y = _make_binary_data(n=100)
        # Feed only class-0 samples in first chunk; class-1 in second
        mask0 = y == 0
        mask1 = y == 1
        clf = DecisionTreeClassifier(max_depth=3)
        clf.partial_fit(X[mask0], y[mask0], classes=np.array([0, 1]))
        # classes_ should still contain both classes
        self.assertIn(0, clf.classes_)
        self.assertIn(1, clf.classes_)
        clf.partial_fit(X[mask1], y[mask1], classes=np.array([0, 1]))
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape[1], 2)

    def test_partial_fit_multiple_chunks_improves_accuracy(self):
        X, y = _make_binary_data(n=400)
        clf = DecisionTreeClassifier(max_depth=4)
        accs = []
        for i in range(4):
            clf.partial_fit(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
            accs.append(np.mean(clf.predict(X[:i*100+100]) == y[:i*100+100]))
        # Final accuracy after all chunks should be reasonable
        self.assertGreater(accs[-1], 0.75)

    def test_partial_fit_returns_self(self):
        X, y = _make_binary_data(n=50)
        clf = DecisionTreeClassifier()
        self.assertIs(clf.partial_fit(X, y), clf)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_single_class_root_is_leaf(self):
        X = np.random.default_rng(0).normal(size=(30, 3))
        y = np.zeros(30, dtype=int)
        clf = DecisionTreeClassifier().fit(X, y)
        self.assertTrue(clf.tree_.is_leaf)
        np.testing.assert_array_equal(clf.predict(X), np.zeros(30))

    def test_two_samples(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([0, 1])
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(preds.shape, (2,))

    def test_zero_variance_feature(self):
        """Constant feature should not cause a crash or NaN predictions."""
        rng = np.random.default_rng(5)
        X = np.column_stack([np.ones(50), rng.normal(size=50)])
        y = (X[:, 1] > 0).astype(int)
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        preds = clf.predict(X)
        self.assertFalse(np.any(np.isnan(preds.astype(float))))

    def test_single_feature(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        y = (X[:, 0] > 10).astype(int)
        clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        self.assertGreater(acc, 0.85)


# ---------------------------------------------------------------------------
# Impurity functions
# ---------------------------------------------------------------------------

class TestImpurityFunctions(unittest.TestCase):

    def test_gini_pure(self):
        clf = DecisionTreeClassifier()
        self.assertAlmostEqual(clf._gini(np.array([0, 0, 0])), 0.0)

    def test_gini_equal_split(self):
        clf = DecisionTreeClassifier()
        g = clf._gini(np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(g, 0.5, places=6)

    def test_entropy_pure(self):
        clf = DecisionTreeClassifier()
        self.assertAlmostEqual(clf._entropy(np.array([1, 1, 1])), 0.0)

    def test_entropy_equal_split(self):
        clf = DecisionTreeClassifier()
        e = clf._entropy(np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(e, 1.0, places=6)   # log2(2) = 1 bit

    def test_gini_empty(self):
        clf = DecisionTreeClassifier()
        self.assertAlmostEqual(clf._gini(np.array([])), 0.0)

    def test_entropy_empty(self):
        clf = DecisionTreeClassifier()
        self.assertAlmostEqual(clf._entropy(np.array([])), 0.0)

    def test_invalid_criterion_raises(self):
        X, y = _make_binary_data(n=20)
        clf = DecisionTreeClassifier(criterion='invalid')
        with self.assertRaises(ValueError):
            clf.fit(X, y)


if __name__ == '__main__':
    unittest.main()
