"""
tests/test_tree.py
------------------
Unit tests for framework.tree – DecisionTreeClassifier.
"""

import unittest
import numpy as np

from framework.tree import DecisionTreeClassifier, _Node


def _make_binary_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestDecisionTreeFit(unittest.TestCase):

    def test_fit_returns_self(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3)
        result = clf.fit(X, y)
        self.assertIs(result, clf)

    def test_predict_shape(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(preds.shape, (len(X),))

    def test_predict_proba_shape(self):
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))

    def test_training_accuracy(self):
        """A depth-5 tree should achieve >85 % on training data."""
        X, y = _make_binary_data(n=500)
        clf = DecisionTreeClassifier(max_depth=5).fit(X, y)
        acc = np.mean(clf.predict(X) == y)
        self.assertGreater(acc, 0.85)

    def test_max_depth_limit(self):
        """Tree with max_depth=1 must have only root + two leaves."""
        X, y = _make_binary_data()
        clf = DecisionTreeClassifier(max_depth=1).fit(X, y)
        root = clf.tree_
        if not root.is_leaf:
            self.assertTrue(root.left.is_leaf)
            self.assertTrue(root.right.is_leaf)

    def test_single_class_input(self):
        """When all labels are the same, tree root should be a leaf."""
        X = np.random.default_rng(0).normal(size=(30, 3))
        y = np.zeros(30, dtype=int)
        clf = DecisionTreeClassifier().fit(X, y)
        self.assertTrue(clf.tree_.is_leaf)
        np.testing.assert_array_equal(clf.predict(X), np.zeros(30))

    def test_partial_fit_accumulates(self):
        X, y = _make_binary_data(n=300)
        clf = DecisionTreeClassifier(max_depth=4)
        clf.partial_fit(X[:150], y[:150])
        clf.partial_fit(X[150:], y[150:])
        acc = np.mean(clf.predict(X) == y)
        self.assertGreater(acc, 0.80)

    def test_gini_pure_node(self):
        clf = DecisionTreeClassifier()
        self.assertAlmostEqual(clf._gini(np.array([0, 0, 0])), 0.0)

    def test_entropy_pure_node(self):
        clf = DecisionTreeClassifier()
        self.assertAlmostEqual(clf._entropy(np.array([1, 1, 1])), 0.0)

    def test_predict_before_fit_raises(self):
        clf = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict(np.ones((3, 2)))


if __name__ == '__main__':
    unittest.main()
