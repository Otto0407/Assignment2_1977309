"""
tests/test_stats.py
-------------------
Unit tests for framework.stats.

Covers:
  - welford_update: correctness, chained updates, large numbers
  - chunk_mean / chunk_variance: basic, NaN, constant column, 1-D input
  - chunk_quantile: values, boundary q, invalid q, NaN
  - chunk_histogram: shapes, counts sum, zero-range, NaN exclusion, range_
  - StreamStats: incremental == batch, NaN per-feature isolation,
    reset, wrong shape, count==1 variance, large-magnitude stability,
    n_samples_seen, std, multi-chunk precision
"""

import unittest
import numpy as np

from framework.stats import (
    StreamStats,
    chunk_mean,
    chunk_variance,
    chunk_quantile,
    chunk_histogram,
    welford_update,
)


# ---------------------------------------------------------------------------
# welford_update
# ---------------------------------------------------------------------------

class TestWelfordUpdate(unittest.TestCase):

    def test_single_value_mean(self):
        count, mean, M2 = welford_update(0, 0.0, 0.0, 5.0)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(M2, 0.0)

    def test_two_values(self):
        """Mean of [2, 4] == 3; M2 == (2-3)^2 + (4-3)^2 == 2."""
        count, mean, M2 = welford_update(0, 0.0, 0.0, 2.0)
        count, mean, M2 = welford_update(count, mean, M2, 4.0)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(M2, 2.0)

    def test_chained_matches_numpy(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(size=50).tolist()
        count, mean, M2 = 0, 0.0, 0.0
        for v in vals:
            count, mean, M2 = welford_update(count, mean, M2, v)
        self.assertAlmostEqual(mean, np.mean(vals), places=10)
        self.assertAlmostEqual(M2 / count, np.var(vals), places=10)

    def test_large_numbers(self):
        """Welford should be numerically stable for large-magnitude data."""
        base = 1e12
        vals = [base + i for i in range(10)]
        count, mean, M2 = 0, 0.0, 0.0
        for v in vals:
            count, mean, M2 = welford_update(count, mean, M2, v)
        self.assertAlmostEqual(mean, np.mean(vals), places=3)
        self.assertAlmostEqual(M2 / count, np.var(vals), places=3)

    def test_zero_value(self):
        count, mean, M2 = welford_update(0, 0.0, 0.0, 0.0)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(mean, 0.0)
        self.assertAlmostEqual(M2, 0.0)


# ---------------------------------------------------------------------------
# chunk_mean
# ---------------------------------------------------------------------------

class TestChunkMean(unittest.TestCase):

    def test_basic_2d(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(chunk_mean(X), [2.0, 3.0])

    def test_single_row(self):
        X = np.array([[7.0, 8.0, 9.0]])
        np.testing.assert_allclose(chunk_mean(X), [7.0, 8.0, 9.0])

    def test_1d_input(self):
        np.testing.assert_allclose(chunk_mean(np.array([1.0, 3.0, 5.0])), [3.0])

    def test_constant_columns(self):
        np.testing.assert_allclose(chunk_mean(np.full((5, 3), 4.0)), [4.0, 4.0, 4.0])

    def test_nan_ignored(self):
        """NaN in one column must not affect the other column's mean."""
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]])
        result = chunk_mean(X)
        self.assertAlmostEqual(result[0], 3.0)   # mean of [1,3,5]
        self.assertAlmostEqual(result[1], 6.0)   # mean of [4,8]

    def test_all_nan_column_returns_nan(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        result = chunk_mean(X)
        self.assertTrue(np.isnan(result[0]))
        self.assertAlmostEqual(result[1], 1.5)

    def test_large_array(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(1000, 5))
        np.testing.assert_allclose(chunk_mean(X), X.mean(axis=0), rtol=1e-10)


# ---------------------------------------------------------------------------
# chunk_variance
# ---------------------------------------------------------------------------

class TestChunkVariance(unittest.TestCase):

    def test_population_variance(self):
        X = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(chunk_variance(X, ddof=0), [2 / 3], rtol=1e-6)

    def test_sample_variance(self):
        X = np.array([[2.0], [4.0], [4.0], [4.0], [5.0], [5.0], [7.0], [9.0]])
        np.testing.assert_allclose(chunk_variance(X, ddof=1), [4.571428], rtol=1e-4)

    def test_constant_column(self):
        np.testing.assert_allclose(chunk_variance(np.full((10, 2), 3.0)), [0.0, 0.0])

    def test_nan_ignored(self):
        """NaN rows should be excluded when computing variance."""
        X = np.array([[1.0], [2.0], [np.nan], [3.0], [4.0]])
        result = chunk_variance(X, ddof=0)
        expected = np.var([1, 2, 3, 4])
        np.testing.assert_allclose(result, [expected], rtol=1e-6)

    def test_matches_numpy(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 4))
        np.testing.assert_allclose(chunk_variance(X), X.var(axis=0), rtol=1e-10)


# ---------------------------------------------------------------------------
# chunk_quantile
# ---------------------------------------------------------------------------

class TestChunkQuantile(unittest.TestCase):

    def test_median(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        np.testing.assert_allclose(chunk_quantile(X, 0.5), [2.0, 20.0])

    def test_q0_is_min(self):
        X = np.arange(10, dtype=float).reshape(5, 2)
        np.testing.assert_allclose(chunk_quantile(X, 0.0), X.min(axis=0))

    def test_q1_is_max(self):
        X = np.arange(10, dtype=float).reshape(5, 2)
        np.testing.assert_allclose(chunk_quantile(X, 1.0), X.max(axis=0))

    def test_invalid_q_raises(self):
        with self.assertRaises(ValueError):
            chunk_quantile(np.ones((3, 2)), q=1.5)

    def test_negative_q_raises(self):
        with self.assertRaises(ValueError):
            chunk_quantile(np.ones((3, 2)), q=-0.1)

    def test_nan_ignored(self):
        X = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 8.0]])
        result = chunk_quantile(X, 0.5)
        self.assertAlmostEqual(result[0], 2.0)    # median of [1,2,3]
        self.assertAlmostEqual(result[1], 6.0)    # median of [4,8]

    def test_1d_input(self):
        X = np.array([10.0, 20.0, 30.0])
        np.testing.assert_allclose(chunk_quantile(X, 0.5), [20.0])


# ---------------------------------------------------------------------------
# chunk_histogram
# ---------------------------------------------------------------------------

class TestChunkHistogram(unittest.TestCase):

    def test_output_shapes(self):
        X = np.random.default_rng(0).uniform(0, 1, (100, 3))
        counts, edges = chunk_histogram(X, bins=10)
        self.assertEqual(counts.shape, (3, 10))
        self.assertEqual(edges.shape, (3, 11))

    def test_counts_sum_to_n(self):
        X = np.random.default_rng(1).normal(0, 1, (50, 2))
        counts, _ = chunk_histogram(X, bins=5)
        np.testing.assert_array_equal(counts.sum(axis=1), [50, 50])

    def test_zero_range_column_no_crash(self):
        """All-constant column must not crash; all counts should sum to n."""
        X = np.full((20, 2), 5.0)
        counts, edges = chunk_histogram(X, bins=5)
        self.assertEqual(counts.shape, (2, 5))
        np.testing.assert_array_equal(counts.sum(axis=1), [20, 20])

    def test_nan_excluded_from_counts(self):
        """NaN values must be excluded; counts sum to number of valid samples."""
        X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0], [5.0, 6.0]])
        counts, _ = chunk_histogram(X, bins=4)
        self.assertEqual(counts[0].sum(), 3)   # col-0: 3 valid values
        self.assertEqual(counts[1].sum(), 3)   # col-1: 3 valid values

    def test_explicit_range(self):
        X = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        counts, edges = chunk_histogram(X, bins=3, range_=(0.0, 9.0))
        self.assertEqual(edges.shape, (2, 4))
        self.assertAlmostEqual(edges[0, 0], 0.0)
        self.assertAlmostEqual(edges[0, -1], 9.0)

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        counts, edges = chunk_histogram(X, bins=5)
        self.assertEqual(counts.shape, (1, 5))
        self.assertEqual(counts.sum(), 5)


# ---------------------------------------------------------------------------
# StreamStats
# ---------------------------------------------------------------------------

class TestStreamStats(unittest.TestCase):

    def _make_data(self, seed=42):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=5.0, scale=2.0, size=(200, 4))

    # --- mean ---

    def test_mean_single_chunk_matches_numpy(self):
        X = self._make_data()
        ss = StreamStats(n_features=4)
        ss.update(X)
        np.testing.assert_allclose(ss.mean(), np.nanmean(X, axis=0), rtol=1e-6)

    def test_mean_incremental_matches_numpy(self):
        X = self._make_data()
        ss = StreamStats(n_features=4)
        for i in range(0, 200, 20):
            ss.update(X[i:i+20])
        np.testing.assert_allclose(ss.mean(), np.nanmean(X, axis=0), rtol=1e-6)

    def test_mean_single_row(self):
        ss = StreamStats(n_features=3)
        ss.update(np.array([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(ss.mean(), [1.0, 2.0, 3.0])

    # --- variance ---

    def test_variance_matches_numpy(self):
        X = self._make_data()
        ss = StreamStats(n_features=4)
        ss.update(X)
        np.testing.assert_allclose(ss.variance(), np.nanvar(X, axis=0), rtol=1e-5)

    def test_variance_count_1_is_zero(self):
        ss = StreamStats(n_features=2)
        ss.update(np.array([[3.0, 5.0]]))
        np.testing.assert_allclose(ss.variance(), [0.0, 0.0])

    def test_variance_constant_column(self):
        ss = StreamStats(n_features=1)
        ss.update(np.full((10, 1), 7.0))
        np.testing.assert_allclose(ss.variance(), [0.0])

    # --- std ---

    def test_std_is_sqrt_variance(self):
        X = self._make_data()
        ss = StreamStats(n_features=4)
        ss.update(X)
        np.testing.assert_allclose(ss.std(), np.sqrt(ss.variance()))

    # --- incremental == batch ---

    def test_incremental_equals_batch(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(100, 3))
        ss_inc = StreamStats(n_features=3)
        for chunk in np.array_split(X, 10):
            ss_inc.update(chunk)
        ss_bat = StreamStats(n_features=3)
        ss_bat.update(X)
        np.testing.assert_allclose(ss_inc.mean(), ss_bat.mean(), rtol=1e-8)
        np.testing.assert_allclose(ss_inc.variance(), ss_bat.variance(), rtol=1e-6)

    # --- NaN handling ---

    def test_nan_feature_isolated(self):
        """NaN in column-1 must not affect column-0 mean/variance."""
        X = np.array([
            [1.0, np.nan],
            [3.0, 2.0],
            [5.0, 4.0],
        ])
        ss = StreamStats(n_features=2)
        ss.update(X)
        self.assertAlmostEqual(ss.mean()[0], 3.0)           # mean of [1,3,5]
        self.assertAlmostEqual(ss.mean()[1], 3.0)           # mean of [2,4]
        self.assertAlmostEqual(ss.variance()[0], np.var([1, 3, 5]))
        self.assertAlmostEqual(ss.variance()[1], np.var([2, 4]))

    def test_all_nan_column_mean_is_nan(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        ss = StreamStats(n_features=2)
        ss.update(X)
        self.assertTrue(np.isnan(ss.mean()[0]))
        self.assertAlmostEqual(ss.mean()[1], 1.5)

    def test_full_nan_row_skipped(self):
        X = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
        ss = StreamStats(n_features=2)
        ss.update(X)
        np.testing.assert_allclose(ss.mean(), [2.0, 3.0])

    # --- reset ---

    def test_reset_zeroes_state(self):
        X = self._make_data()
        ss = StreamStats(n_features=4)
        ss.update(X)
        ss.reset()
        # After reset, no samples seen → mean is NaN (count=0), variance is 0
        self.assertTrue(np.all(np.isnan(ss.mean())))
        np.testing.assert_allclose(ss.variance(), np.zeros(4))
        np.testing.assert_array_equal(ss.n_samples_seen, np.zeros(4, dtype=int))

    def test_update_after_reset(self):
        ss = StreamStats(n_features=2)
        ss.update(np.array([[10.0, 20.0]]))
        ss.reset()
        ss.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(ss.mean(), [2.0, 3.0])

    # --- error handling ---

    def test_wrong_feature_count_raises(self):
        ss = StreamStats(n_features=3)
        with self.assertRaises(ValueError):
            ss.update(np.ones((5, 4)))

    # --- n_samples_seen ---

    def test_n_samples_seen_no_nan(self):
        ss = StreamStats(n_features=2)
        ss.update(np.ones((7, 2)))
        np.testing.assert_array_equal(ss.n_samples_seen, [7, 7])

    def test_n_samples_seen_with_nan(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
        ss = StreamStats(n_features=2)
        ss.update(X)
        np.testing.assert_array_equal(ss.n_samples_seen, [3, 2])

    # --- numerical stability ---

    def test_large_magnitude_stability(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=1e10, scale=1.0, size=(500, 3))
        ss = StreamStats(n_features=3)
        for chunk in np.array_split(X, 25):
            ss.update(chunk)
        np.testing.assert_allclose(ss.mean(), np.mean(X, axis=0), rtol=1e-6)
        np.testing.assert_allclose(ss.variance(), np.var(X, axis=0), rtol=1e-4)

    def test_many_chunks_precision(self):
        rng = np.random.default_rng(3)
        X = rng.normal(loc=1000.0, scale=0.01, size=(1000, 3))
        ss = StreamStats(n_features=3)
        for chunk in np.array_split(X, 50):
            ss.update(chunk)
        np.testing.assert_allclose(ss.mean(), np.mean(X, axis=0), rtol=1e-8)
        np.testing.assert_allclose(ss.variance(), np.var(X, axis=0), rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
