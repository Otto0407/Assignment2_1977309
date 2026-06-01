"""
tests/test_stats.py
-------------------
Unit tests for framework.stats – StreamStats, chunk_mean, chunk_variance,
chunk_quantile, chunk_histogram, welford_update.
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


class TestWelfordUpdate(unittest.TestCase):
    """Tests for the single-sample Welford update helper."""

    def test_single_value(self):
        """After one update the mean equals the value and M2 is 0."""
        count, mean, M2 = welford_update(0, 0.0, 0.0, 5.0)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(M2, 0.0)

    def test_two_values(self):
        """Mean of [2, 4] is 3; M2 == 2."""
        count, mean, M2 = welford_update(0, 0.0, 0.0, 2.0)
        count, mean, M2 = welford_update(count, mean, M2, 4.0)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(M2, 2.0)


class TestChunkMean(unittest.TestCase):

    def test_basic_2d(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = chunk_mean(X)
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_single_row(self):
        X = np.array([[7.0, 8.0, 9.0]])
        np.testing.assert_allclose(chunk_mean(X), [7.0, 8.0, 9.0])

    def test_1d_input(self):
        """1-D array treated as a single-feature (n, 1) chunk."""
        X = np.array([1.0, 3.0, 5.0])
        result = chunk_mean(X)
        np.testing.assert_allclose(result, [3.0])

    def test_all_same(self):
        X = np.full((5, 3), 4.0)
        np.testing.assert_allclose(chunk_mean(X), [4.0, 4.0, 4.0])


class TestChunkVariance(unittest.TestCase):

    def test_population_variance(self):
        X = np.array([[1.0], [2.0], [3.0]])
        var = chunk_variance(X, ddof=0)
        np.testing.assert_allclose(var, [2 / 3], rtol=1e-6)

    def test_sample_variance(self):
        X = np.array([[2.0], [4.0], [4.0], [4.0], [5.0], [5.0], [7.0], [9.0]])
        var = chunk_variance(X, ddof=1)
        np.testing.assert_allclose(var, [4.571428], rtol=1e-4)

    def test_constant_column(self):
        X = np.full((10, 2), 3.0)
        np.testing.assert_allclose(chunk_variance(X), [0.0, 0.0])


class TestChunkQuantile(unittest.TestCase):

    def test_median(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        np.testing.assert_allclose(chunk_quantile(X, 0.5), [2.0, 20.0])

    def test_min_max(self):
        X = np.arange(10, dtype=float).reshape(5, 2)
        np.testing.assert_allclose(chunk_quantile(X, 0.0), X.min(axis=0))
        np.testing.assert_allclose(chunk_quantile(X, 1.0), X.max(axis=0))

    def test_invalid_q(self):
        with self.assertRaises(ValueError):
            chunk_quantile(np.ones((3, 2)), q=1.5)


class TestChunkHistogram(unittest.TestCase):

    def test_output_shapes(self):
        X = np.random.default_rng(0).uniform(0, 1, (100, 3))
        counts, edges = chunk_histogram(X, bins=10)
        self.assertEqual(counts.shape, (3, 10))
        self.assertEqual(edges.shape, (3, 11))

    def test_counts_sum_to_n(self):
        X = np.random.default_rng(1).normal(0, 1, (50, 2))
        counts, _ = chunk_histogram(X, bins=5)
        for row in counts:
            self.assertEqual(row.sum(), 50)


class TestStreamStats(unittest.TestCase):

    def _make_data(self):
        rng = np.random.default_rng(42)
        return rng.normal(loc=5.0, scale=2.0, size=(200, 4))

    def test_mean_matches_numpy(self):
        X = self._make_data()
        ss = StreamStats(n_features=4)
        for i in range(0, 200, 20):
            ss.update(X[i:i+20])
        np.testing.assert_allclose(ss.mean(), X.mean(axis=0), rtol=1e-6)

    def test_variance_matches_numpy(self):
        X = self._make_data()
        ss = StreamStats(n_features=4)
        ss.update(X)
        np.testing.assert_allclose(ss.variance(), X.var(axis=0), rtol=1e-5)

    def test_std_is_sqrt_variance(self):
        X = self._make_data()
        ss = StreamStats(n_features=4)
        ss.update(X)
        np.testing.assert_allclose(ss.std(), np.sqrt(ss.variance()))

    def test_reset_zeroes_state(self):
        X = self._make_data()
        ss = StreamStats(n_features=4)
        ss.update(X)
        ss.reset()
        np.testing.assert_allclose(ss.mean(), np.zeros(4))
        np.testing.assert_allclose(ss.variance(), np.zeros(4))

    def test_single_row(self):
        X = np.array([[1.0, 2.0, 3.0]])
        ss = StreamStats(n_features=3)
        ss.update(X)
        np.testing.assert_allclose(ss.mean(), [1.0, 2.0, 3.0])

    def test_wrong_feature_count_raises(self):
        ss = StreamStats(n_features=3)
        with self.assertRaises(ValueError):
            ss.update(np.ones((5, 4)))

    def test_incremental_equals_batch(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(100, 3))
        ss_inc = StreamStats(n_features=3)
        for chunk in np.array_split(X, 10):
            ss_inc.update(chunk)
        ss_batch = StreamStats(n_features=3)
        ss_batch.update(X)
        np.testing.assert_allclose(ss_inc.mean(), ss_batch.mean(), rtol=1e-8)
        np.testing.assert_allclose(ss_inc.variance(), ss_batch.variance(), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
