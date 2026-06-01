"""
tests/test_preprocessing.py
----------------------------
Unit tests for framework.preprocessing – StandardScaler, MinMaxScaler,
Imputer, OneHotEncoder.
"""

import unittest
import numpy as np

from framework.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder,
)


class TestStandardScaler(unittest.TestCase):

    def _make_data(self, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=3.0, scale=5.0, size=(100, 4))

    def test_transform_zero_mean(self):
        X = self._make_data()
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), np.zeros(4), atol=1e-10)

    def test_transform_unit_variance(self):
        X = self._make_data()
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        np.testing.assert_allclose(Xt.var(axis=0), np.ones(4), atol=1e-6)

    def test_fit_transform_equivalent(self):
        X = self._make_data(seed=1)
        sc1 = StandardScaler()
        sc1.partial_fit(X)
        ref = sc1.transform(X)
        sc2 = StandardScaler()
        result = sc2.fit_transform(X)
        np.testing.assert_allclose(result, ref, rtol=1e-10)

    def test_inverse_transform_roundtrip(self):
        X = self._make_data(seed=2)
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        X_rec = sc.inverse_transform(Xt)
        np.testing.assert_allclose(X_rec, X, atol=1e-8)

    def test_incremental_partial_fit(self):
        """Two partial_fit calls should yield same mean as one batch fit."""
        X = self._make_data(seed=3)
        sc_batch = StandardScaler()
        sc_batch.partial_fit(X)

        sc_inc = StandardScaler()
        sc_inc.partial_fit(X[:50])
        sc_inc.partial_fit(X[50:])

        # Means should match
        np.testing.assert_allclose(
            sc_batch._mean, sc_inc._mean, rtol=1e-8,
        )

    def test_zero_variance_feature_no_divide_by_zero(self):
        X = np.column_stack([
            np.ones(20),
            np.random.default_rng(5).normal(size=20),
        ])
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        self.assertFalse(np.any(np.isnan(Xt)))
        self.assertFalse(np.any(np.isinf(Xt)))

    def test_transform_before_fit_raises(self):
        sc = StandardScaler()
        with self.assertRaises(RuntimeError):
            sc.transform(np.ones((5, 2)))


class TestMinMaxScaler(unittest.TestCase):

    def test_range_0_1(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        sc = MinMaxScaler(feature_range=(0, 1))
        sc.partial_fit(X)
        Xt = sc.transform(X)
        np.testing.assert_allclose(Xt.min(axis=0), [0.0, 0.0])
        np.testing.assert_allclose(Xt.max(axis=0), [1.0, 1.0])

    def test_custom_range(self):
        X = np.array([[0.0], [10.0]])
        sc = MinMaxScaler(feature_range=(-1, 1))
        sc.partial_fit(X)
        Xt = sc.transform(X)
        np.testing.assert_allclose(Xt.ravel(), [-1.0, 1.0])

    def test_inverse_roundtrip(self):
        X = np.random.default_rng(0).uniform(0, 100, (50, 3))
        sc = MinMaxScaler()
        sc.partial_fit(X)
        np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X, atol=1e-8)


class TestImputer(unittest.TestCase):

    def test_mean_imputation(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 2.0]])
        imp = Imputer(strategy='mean')
        imp.partial_fit(X)
        Xt = imp.transform(np.array([[np.nan, np.nan]]))
        self.assertAlmostEqual(Xt[0, 0], 3.0)  # mean of [1, 3, 5]

    def test_no_nan_passthrough(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        imp = Imputer(strategy='mean')
        imp.partial_fit(X)
        Xt = imp.transform(X.copy())
        np.testing.assert_allclose(Xt, X)

    def test_constant_imputation(self):
        X = np.array([[1.0, np.nan], [np.nan, 3.0]])
        imp = Imputer(strategy='constant', fill_value=-999.0)
        imp.partial_fit(X)
        Xt = imp.transform(np.array([[np.nan, np.nan]]))
        np.testing.assert_allclose(Xt, [[-999.0, -999.0]])


class TestOneHotEncoder(unittest.TestCase):

    def test_basic_encoding(self):
        X = np.array([['a'], ['b'], ['c']])
        enc = OneHotEncoder()
        enc.partial_fit(X)
        Xt = enc.transform(X)
        self.assertEqual(Xt.shape, (3, 3))
        self.assertEqual(Xt.sum(), 3)  # one hot per row

    def test_feature_names(self):
        X = np.array([['cat', '1'], ['dog', '2']])
        enc = OneHotEncoder()
        enc.partial_fit(X)
        names = enc.get_feature_names()
        self.assertIn('x0_cat', names)
        self.assertIn('x1_1', names)

    def test_incremental_categories(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([['a'], ['b']]))
        enc.partial_fit(np.array([['c']]))
        Xt = enc.transform(np.array([['c']]))
        self.assertEqual(Xt.shape[1], 3)  # a, b, c


if __name__ == '__main__':
    unittest.main()
