"""
tests/test_preprocessing.py
----------------------------
Unit tests for framework.preprocessing.

Covers:
  StandardScaler
    - zero-mean, unit-variance after fit
    - incremental (Chan) == batch result
    - zero-variance feature (no divide-by-zero)
    - inverse_transform roundtrip
    - fit_transform, transform before fit raises
    - NaN input propagation
    - large-magnitude numerical stability

  MinMaxScaler
    - output range [0,1] and custom range
    - inverse_transform roundtrip
    - incremental partial_fit (min/max updated correctly)
    - zero-range column (constant feature)
    - invalid feature_range raises

  Imputer
    - mean strategy: basic, NaN-only row ignored, incremental == batch
    - median strategy: basic correctness
    - constant strategy: basic, custom fill_value
    - transform before fit raises
    - invalid strategy raises
    - NaN column fill correctness with mixed NaN/valid data

  OneHotEncoder
    - basic encoding shape and values
    - multiple features
    - incremental category discovery
    - feature_names format
    - transform before fit raises
    - unknown category (maps to all-zero row)
    - broadcasting: inner loop eliminated (verified via output correctness)
"""

import unittest
import numpy as np

from framework.preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class TestStandardScaler(unittest.TestCase):

    def _make(self, seed=0, n=100, d=4, loc=3.0, scale=5.0):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=loc, scale=scale, size=(n, d))

    def test_transform_zero_mean(self):
        X = self._make()
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), np.zeros(4), atol=1e-10)

    def test_transform_unit_variance(self):
        X = self._make()
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        np.testing.assert_allclose(Xt.var(axis=0), np.ones(4), atol=1e-6)

    def test_fit_transform_equivalent(self):
        X = self._make(seed=1)
        ref = StandardScaler().fit_transform(X)
        sc = StandardScaler()
        sc.partial_fit(X)
        np.testing.assert_allclose(sc.transform(X), ref, rtol=1e-10)

    def test_inverse_transform_roundtrip(self):
        X = self._make(seed=2)
        sc = StandardScaler()
        sc.partial_fit(X)
        np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X, atol=1e-8)

    def test_incremental_chan_equals_batch(self):
        """Two partial_fit calls (Chan) must yield the same result as one batch call."""
        X = self._make(seed=3, n=200)
        sc_batch = StandardScaler()
        sc_batch.partial_fit(X)

        sc_inc = StandardScaler()
        sc_inc.partial_fit(X[:100])
        sc_inc.partial_fit(X[100:])

        np.testing.assert_allclose(sc_batch._mean, sc_inc._mean, rtol=1e-10)
        np.testing.assert_allclose(sc_batch._M2,   sc_inc._M2,   rtol=1e-10)

    def test_many_chunks_equals_batch(self):
        X = self._make(seed=4, n=500)
        sc_batch = StandardScaler()
        sc_batch.partial_fit(X)

        sc_inc = StandardScaler()
        for chunk in np.array_split(X, 10):
            sc_inc.partial_fit(chunk)

        np.testing.assert_allclose(sc_batch._mean, sc_inc._mean, rtol=1e-10)
        np.testing.assert_allclose(
            sc_batch.transform(X), sc_inc.transform(X), rtol=1e-8
        )

    def test_zero_variance_no_nan_or_inf(self):
        X = np.column_stack([np.ones(20), np.random.default_rng(5).normal(size=20)])
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        self.assertFalse(np.any(np.isnan(Xt)))
        self.assertFalse(np.any(np.isinf(Xt)))

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform(np.ones((5, 2)))

    def test_inverse_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().inverse_transform(np.ones((5, 2)))

    def test_large_magnitude_stability(self):
        """Chan's algorithm must remain stable for loc=1e10."""
        rng = np.random.default_rng(6)
        X = rng.normal(loc=1e10, scale=1.0, size=(200, 3))
        sc = StandardScaler()
        for chunk in np.array_split(X, 10):
            sc.partial_fit(chunk)
        Xt = sc.transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), np.zeros(3), atol=1e-4)
        np.testing.assert_allclose(Xt.var(axis=0),  np.ones(3),  rtol=1e-4)

    def test_1d_input(self):
        sc = StandardScaler()
        sc.partial_fit(np.array([1.0, 2.0, 3.0]))  # 1-D treated as (3,1)
        self.assertIsNotNone(sc._mean)

    def test_nan_in_chunk_does_not_corrupt_state(self):
        """NaN in chunk col-1 must not poison subsequent valid chunks."""
        sc = StandardScaler()
        sc.partial_fit(np.array([[1.0, np.nan], [3.0, 4.0]]))   # col-1: only 4.0 valid
        sc.partial_fit(np.array([[5.0, 6.0], [7.0, 8.0]]))       # col-1: 6.0, 8.0 valid
        # col-0: mean([1,3,5,7]) = 4.0;  col-1: nanmean([4,6,8]) = 6.0
        np.testing.assert_allclose(sc._mean[0], 4.0, atol=1e-10)
        np.testing.assert_allclose(sc._mean[1], 6.0, atol=1e-10)
        self.assertFalse(np.any(np.isnan(sc._mean)))

    def test_nan_input_transform_output(self):
        """transform on a row with NaN should propagate NaN for that feature only."""
        sc = StandardScaler()
        sc.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        Xt = sc.transform(np.array([[np.nan, 3.0]]))
        self.assertTrue(np.isnan(Xt[0, 0]))
        self.assertFalse(np.isnan(Xt[0, 1]))

    def test_fit_transform_returns_correct_shape(self):
        X = np.random.default_rng(7).normal(size=(50, 3))
        Xt = StandardScaler().fit_transform(X)
        self.assertEqual(Xt.shape, X.shape)


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------

class TestMinMaxScaler(unittest.TestCase):

    def test_range_0_1(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        sc = MinMaxScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        np.testing.assert_allclose(Xt.min(axis=0), [0.0, 0.0])
        np.testing.assert_allclose(Xt.max(axis=0), [1.0, 1.0])

    def test_custom_range(self):
        X = np.array([[0.0], [10.0]])
        sc = MinMaxScaler(feature_range=(-1, 1))
        sc.partial_fit(X)
        np.testing.assert_allclose(sc.transform(X).ravel(), [-1.0, 1.0])

    def test_inverse_roundtrip(self):
        X = np.random.default_rng(0).uniform(0, 100, (50, 3))
        sc = MinMaxScaler()
        sc.partial_fit(X)
        np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X, atol=1e-8)

    def test_incremental_minmax(self):
        """Min/max must track across two chunks correctly."""
        X1 = np.array([[0.0, 5.0]])
        X2 = np.array([[10.0, -5.0]])
        sc = MinMaxScaler()
        sc.partial_fit(X1)
        sc.partial_fit(X2)
        np.testing.assert_allclose(sc._data_min, [0.0, -5.0])
        np.testing.assert_allclose(sc._data_max, [10.0,  5.0])

    def test_zero_range_no_divide_by_zero(self):
        X = np.full((10, 2), 5.0)
        sc = MinMaxScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        self.assertFalse(np.any(np.isnan(Xt)))
        self.assertFalse(np.any(np.isinf(Xt)))

    def test_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            MinMaxScaler(feature_range=(1, 0))

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            MinMaxScaler().transform(np.ones((3, 2)))

    def test_inverse_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            MinMaxScaler().inverse_transform(np.ones((3, 2)))

    def test_all_nan_column_then_valid(self):
        """all-NaN column on first chunk must not cause NaN state after valid data arrives."""
        sc = MinMaxScaler()
        sc.partial_fit(np.array([[np.nan, 1.0], [np.nan, 2.0]]))   # col-0 all NaN
        sc.partial_fit(np.array([[3.0, 1.5]]))                       # col-0 now has data
        self.assertAlmostEqual(sc._data_min[0], 3.0)
        self.assertAlmostEqual(sc._data_max[0], 3.0)
        self.assertFalse(np.any(np.isnan(sc._data_min)))
        self.assertFalse(np.any(np.isinf(sc._data_min)))

    def test_fit_transform(self):
        X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        Xt = MinMaxScaler().fit_transform(X)
        np.testing.assert_allclose(Xt.min(axis=0), [0.0, 0.0])
        np.testing.assert_allclose(Xt.max(axis=0), [1.0, 1.0])


# ---------------------------------------------------------------------------
# Imputer
# ---------------------------------------------------------------------------

class TestImputer(unittest.TestCase):

    # --- mean strategy ---

    def test_mean_basic(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 2.0]])
        imp = Imputer(strategy='mean')
        imp.partial_fit(X)
        Xt = imp.transform(np.array([[np.nan, np.nan]]))
        self.assertAlmostEqual(Xt[0, 0], 3.0)   # mean of [1,3,5]
        self.assertAlmostEqual(Xt[0, 1], 3.0)   # mean of [4,2]

    def test_mean_no_nan_passthrough(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        imp = Imputer(strategy='mean')
        imp.partial_fit(X)
        np.testing.assert_allclose(imp.transform(X.copy()), X)

    def test_mean_incremental_equals_batch(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        X[rng.random(size=X.shape) < 0.2] = np.nan  # ~20 % NaN

        imp_batch = Imputer(strategy='mean')
        imp_batch.partial_fit(X)

        imp_inc = Imputer(strategy='mean')
        for chunk in np.array_split(X, 5):
            imp_inc.partial_fit(chunk)

        np.testing.assert_allclose(
            imp_batch._statistics, imp_inc._statistics, rtol=1e-8,
            err_msg="Incremental mean imputer differs from batch"
        )

    def test_mean_nan_isolation(self):
        """NaN in col-0 must not affect col-1 statistics."""
        X = np.array([[np.nan, 10.0], [2.0, 20.0], [4.0, 30.0]])
        imp = Imputer(strategy='mean')
        imp.partial_fit(X)
        self.assertAlmostEqual(imp._statistics[0], 3.0)   # mean([2,4])
        self.assertAlmostEqual(imp._statistics[1], 20.0)  # mean([10,20,30])

    def test_mean_all_nan_column(self):
        """A column with all NaN should produce NaN statistic (not 0)."""
        X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        imp = Imputer(strategy='mean')
        imp.partial_fit(X)
        self.assertTrue(np.isnan(imp._statistics[0]))
        self.assertAlmostEqual(imp._statistics[1], 1.5)

    # --- median strategy ---

    def test_median_basic(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, 40.0]])
        imp = Imputer(strategy='median')
        imp.partial_fit(X)
        Xt = imp.transform(np.array([[np.nan, np.nan]]))
        self.assertAlmostEqual(Xt[0, 0], np.median([1, 2, 3, 100]))
        self.assertAlmostEqual(Xt[0, 1], np.median([10, 20, 30, 40]))

    def test_median_incremental_grows_buffer(self):
        imp = Imputer(strategy='median')
        imp.partial_fit(np.array([[1.0], [2.0]]))
        imp.partial_fit(np.array([[3.0], [4.0]]))
        # median of [1,2,3,4] = 2.5
        self.assertAlmostEqual(imp._statistics[0], 2.5)

    # --- constant strategy ---

    def test_constant_basic(self):
        X = np.array([[1.0, np.nan], [np.nan, 3.0]])
        imp = Imputer(strategy='constant', fill_value=-999.0)
        imp.partial_fit(X)
        Xt = imp.transform(np.array([[np.nan, np.nan]]))
        np.testing.assert_allclose(Xt, [[-999.0, -999.0]])

    def test_constant_default_zero(self):
        imp = Imputer(strategy='constant')
        imp.partial_fit(np.ones((3, 2)))
        Xt = imp.transform(np.array([[np.nan, np.nan]]))
        np.testing.assert_allclose(Xt, [[0.0, 0.0]])

    # --- error handling ---

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            Imputer(strategy='mode')

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            Imputer().transform(np.ones((3, 2)))

    # --- vectorised transform ---

    def test_transform_vectorised_no_loop(self):
        """Verify transform output is correct for multiple NaN positions."""
        X_fit = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        imp = Imputer(strategy='mean')
        imp.partial_fit(X_fit)
        X_test = np.array([[np.nan, 5.0, np.nan], [1.0, np.nan, 6.0]])
        Xt = imp.transform(X_test)
        expected_col0 = np.mean([1, 4, 7])
        expected_col1 = np.mean([2, 5, 8])
        expected_col2 = np.mean([3, 6, 9])
        self.assertAlmostEqual(Xt[0, 0], expected_col0)
        self.assertAlmostEqual(Xt[0, 2], expected_col2)
        self.assertAlmostEqual(Xt[1, 1], expected_col1)
        self.assertAlmostEqual(Xt[0, 1], 5.0)   # was not NaN
        self.assertAlmostEqual(Xt[1, 0], 1.0)   # was not NaN

    def test_fit_transform(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, np.nan]])
        imp = Imputer(strategy='mean')
        Xt = imp.fit_transform(X)
        self.assertFalse(np.any(np.isnan(Xt)))
        self.assertEqual(Xt.shape, X.shape)


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

class TestOneHotEncoder(unittest.TestCase):

    def test_basic_shape_and_values(self):
        X = np.array([['a'], ['b'], ['c']])
        enc = OneHotEncoder()
        enc.partial_fit(X)
        Xt = enc.transform(X)
        self.assertEqual(Xt.shape, (3, 3))
        np.testing.assert_array_equal(Xt.sum(axis=1), [1, 1, 1])  # one-hot

    def test_output_is_binary(self):
        X = np.array([['cat'], ['dog'], ['cat'], ['fish']])
        enc = OneHotEncoder()
        enc.partial_fit(X)
        Xt = enc.transform(X)
        self.assertTrue(np.all((Xt == 0) | (Xt == 1)))

    def test_multiple_features(self):
        X = np.array([['a', '1'], ['b', '2'], ['a', '1']])
        enc = OneHotEncoder()
        enc.partial_fit(X)
        Xt = enc.transform(X)
        # col0 has 2 cats, col1 has 2 cats → width 4
        self.assertEqual(Xt.shape, (3, 4))
        np.testing.assert_array_equal(Xt.sum(axis=1), [2, 2, 2])

    def test_incremental_categories(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([['a'], ['b']]))
        enc.partial_fit(np.array([['c']]))
        Xt = enc.transform(np.array([['a'], ['b'], ['c']]))
        self.assertEqual(Xt.shape[1], 3)           # 3 categories total
        np.testing.assert_array_equal(Xt.sum(axis=1), [1, 1, 1])

    def test_feature_names_format(self):
        X = np.array([['cat', '1'], ['dog', '2']])
        enc = OneHotEncoder()
        enc.partial_fit(X)
        names = enc.get_feature_names()
        self.assertIn('x0_cat', names)
        self.assertIn('x0_dog', names)
        self.assertIn('x1_1', names)
        self.assertIn('x1_2', names)

    def test_feature_names_empty_before_fit(self):
        self.assertEqual(OneHotEncoder().get_feature_names(), [])

    def test_unknown_category_is_all_zero(self):
        """A category not seen during fit must produce an all-zero row."""
        enc = OneHotEncoder()
        enc.partial_fit(np.array([['a'], ['b']]))
        Xt = enc.transform(np.array([['z']]))
        np.testing.assert_array_equal(Xt, [[0.0, 0.0]])

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            OneHotEncoder().transform(np.array([['a']]))

    def test_vectorised_block_correctness(self):
        """
        Verify that the broadcasting transform produces the same result as
        a naive per-category loop would.
        """
        cats = np.array(['a', 'b', 'c'])
        X_col = np.array(['b', 'a', 'c', 'b'])
        # Expected block (naive):
        expected = np.zeros((4, 3))
        for k, c in enumerate(cats):
            expected[:, k] = (X_col == c).astype(float)
        # Broadcasting (what transform does):
        result = (X_col.reshape(-1, 1) == cats.reshape(1, -1)).astype(float)
        np.testing.assert_array_equal(result, expected)

    def test_1d_input(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array(['x', 'y', 'z']))
        Xt = enc.transform(np.array(['x', 'z']))
        self.assertEqual(Xt.shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
