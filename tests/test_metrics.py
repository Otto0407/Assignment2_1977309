"""
tests/test_metrics.py
---------------------
Unit tests for framework.metrics – streaming classes and standalone functions.

Covers:
  - accuracy_score: perfect, zero, partial, empty
  - precision_score: binary, multiclass, all averages, edge cases
  - recall_score: same
  - f1_score: harmonic mean consistency, zero case
  - confusion_matrix: binary, multiclass, diagonal, shape
  - roc_auc_score: perfect, random, all-same labels, single sample
  - Streaming Accuracy: cumulative, windowed, reset, window overflow
  - Streaming Precision / Recall / F1: update + result + reset
  - Streaming ConfusionMatrix: accumulation, reset, out-of-range labels
  - Streaming AUC: update + result + reset
  - Vectorisation: confusion_matrix bincount correctness
"""

import unittest
import numpy as np

from framework.metrics import (
    Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUC,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)


# ---------------------------------------------------------------------------
# accuracy_score
# ---------------------------------------------------------------------------

class TestAccuracyScore(unittest.TestCase):

    def test_perfect(self):
        y = np.array([0, 1, 2, 1, 0])
        self.assertAlmostEqual(accuracy_score(y, y), 1.0)

    def test_zero(self):
        self.assertAlmostEqual(accuracy_score([0, 0, 0], [1, 1, 1]), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(accuracy_score([0, 1, 0, 1], [0, 0, 0, 1]), 0.75)

    def test_empty(self):
        self.assertAlmostEqual(accuracy_score([], []), 0.0)

    def test_single_correct(self):
        self.assertAlmostEqual(accuracy_score([1], [1]), 1.0)

    def test_single_wrong(self):
        self.assertAlmostEqual(accuracy_score([1], [0]), 0.0)


# ---------------------------------------------------------------------------
# precision_score
# ---------------------------------------------------------------------------

class TestPrecisionScore(unittest.TestCase):

    def test_perfect_macro(self):
        y = np.array([0, 1, 2])
        self.assertAlmostEqual(precision_score(y, y, average='macro'), 1.0)

    def test_binary_macro(self):
        # TP=1, FP=0 for class 1; TP=2, FP=0 for class 0 → 1.0
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        p = precision_score(y_true, y_pred, average='macro')
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_micro_equals_accuracy(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0])
        p_micro = precision_score(y_true, y_pred, average='micro')
        acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(p_micro, acc)

    def test_weighted(self):
        y_true = np.array([0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        p = precision_score(y_true, y_pred, average='weighted')
        self.assertAlmostEqual(p, 1.0)

    def test_zero_support_class_excluded(self):
        # Class 2 never appears in y_true; macro should not be affected
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 2, 0, 2])
        # precision for class 0: TP=2, FP=0 → 1.0; class 1: TP=0, FP=0 → 0.0
        p = precision_score(y_true, y_pred, average='macro')
        self.assertAlmostEqual(p, 0.5)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(precision_score([], [], average='macro'), 0.0)


# ---------------------------------------------------------------------------
# recall_score
# ---------------------------------------------------------------------------

class TestRecallScore(unittest.TestCase):

    def test_perfect(self):
        y = np.array([0, 1, 2])
        self.assertAlmostEqual(recall_score(y, y, average='macro'), 1.0)

    def test_micro_equals_accuracy(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0])
        r_micro = recall_score(y_true, y_pred, average='micro')
        acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(r_micro, acc)

    def test_weighted_perfect(self):
        y = np.array([0, 0, 1, 1, 1])
        self.assertAlmostEqual(recall_score(y, y, average='weighted'), 1.0)

    def test_zero_recall_class(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        r = recall_score(y_true, y_pred, average='macro')
        # class 0: recall=1.0, class 1: recall=0.0 → macro=0.5
        self.assertAlmostEqual(r, 0.5)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(recall_score([], [], average='macro'), 0.0)


# ---------------------------------------------------------------------------
# f1_score
# ---------------------------------------------------------------------------

class TestF1Score(unittest.TestCase):

    def test_perfect(self):
        y = np.array([0, 1, 2])
        self.assertAlmostEqual(f1_score(y, y), 1.0)

    def test_zero(self):
        # No correct predictions → f1=0
        y_true = np.array([0, 0])
        y_pred = np.array([1, 1])
        self.assertAlmostEqual(f1_score(y_true, y_pred), 0.0)

    def test_harmonic_mean_consistency(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        expected = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        self.assertAlmostEqual(f1_score(y_true, y_pred), expected, places=8)

    def test_micro_f1_equals_accuracy(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0])
        f1_micro = f1_score(y_true, y_pred, average='micro')
        acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(f1_micro, acc)

    def test_weighted(self):
        y = np.array([0, 0, 1, 1, 1])
        self.assertAlmostEqual(f1_score(y, y, average='weighted'), 1.0)


# ---------------------------------------------------------------------------
# confusion_matrix (standalone, vectorised)
# ---------------------------------------------------------------------------

class TestConfusionMatrix(unittest.TestCase):

    def test_binary_shape(self):
        cm = confusion_matrix([0, 1, 0, 1], [0, 1, 1, 0])
        self.assertEqual(cm.shape, (2, 2))

    def test_total_count(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.sum(), 4)

    def test_perfect_diagonal(self):
        y = np.array([0, 1, 2])
        cm = confusion_matrix(y, y)
        np.testing.assert_array_equal(np.diag(cm), [1, 1, 1])
        self.assertEqual(cm.sum() - cm.trace(), 0)

    def test_all_wrong_binary(self):
        cm = confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
        self.assertEqual(cm.trace(), 0)
        self.assertEqual(cm.sum(), 4)

    def test_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 2, 0])
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm.sum(), 6)
        # 3 correct
        self.assertEqual(cm.trace(), 3)

    def test_single_class(self):
        cm = confusion_matrix([0, 0, 0], [0, 0, 0])
        np.testing.assert_array_equal(cm, [[3]])

    def test_bincount_matches_loop(self):
        """Vectorised result must match a naive loop implementation."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 4, size=100)
        y_pred = rng.integers(0, 4, size=100)
        cm_vec = confusion_matrix(y_true, y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n = len(classes)
        cm_loop = np.zeros((n, n), dtype=int)
        c2i = {c: i for i, c in enumerate(classes)}
        for t, p in zip(y_true, y_pred):
            cm_loop[c2i[t], c2i[p]] += 1
        np.testing.assert_array_equal(cm_vec, cm_loop)


# ---------------------------------------------------------------------------
# roc_auc_score (vectorised)
# ---------------------------------------------------------------------------

class TestRocAucScore(unittest.TestCase):

    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc_score(y_true, y_score), 1.0)

    def test_worst(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(roc_auc_score(y_true, y_score), 0.0)

    def test_random_around_half(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        y_score = rng.uniform(0, 1, size=200)
        auc = roc_auc_score(y_true, y_score)
        self.assertGreater(auc, 0.3)
        self.assertLess(auc, 0.7)

    def test_all_positive_returns_zero(self):
        # Cannot compute AUC without both classes
        self.assertAlmostEqual(roc_auc_score([1, 1, 1], [0.9, 0.8, 0.7]), 0.0)

    def test_all_negative_returns_zero(self):
        self.assertAlmostEqual(roc_auc_score([0, 0, 0], [0.1, 0.5, 0.9]), 0.0)

    def test_tie_handling(self):
        # Tied scores should not crash
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        auc = roc_auc_score(y_true, y_score)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_large_input(self):
        rng = np.random.default_rng(7)
        n = 10_000
        y_true = rng.integers(0, 2, size=n)
        y_score = y_true * 0.6 + rng.uniform(0, 0.4, size=n)
        auc = roc_auc_score(y_true, y_score)
        self.assertGreater(auc, 0.85)


# ---------------------------------------------------------------------------
# Streaming Accuracy
# ---------------------------------------------------------------------------

class TestStreamingAccuracy(unittest.TestCase):

    def test_cumulative_basic(self):
        acc = Accuracy()
        acc.update([0, 1, 1], [0, 1, 0])
        self.assertAlmostEqual(acc.result(), 2 / 3)

    def test_cumulative_multiple_updates(self):
        acc = Accuracy()
        acc.update([0, 1], [0, 1])   # 2 correct
        acc.update([0, 1], [1, 0])   # 0 correct
        self.assertAlmostEqual(acc.result(), 0.5)

    def test_windowed_basic(self):
        acc = Accuracy(window_size=4)
        acc.update([1, 1, 1, 1], [1, 1, 1, 0])
        self.assertAlmostEqual(acc.result(), 0.75)

    def test_windowed_overflow(self):
        acc = Accuracy(window_size=4)
        acc.update([0, 0, 0, 0], [1, 1, 1, 1])   # 0 correct
        acc.update([1, 1, 1, 1], [1, 1, 1, 1])   # 4 correct, evicts previous 4
        self.assertAlmostEqual(acc.result(), 1.0)

    def test_windowed_partial_fill(self):
        acc = Accuracy(window_size=10)
        acc.update([0, 1, 0], [0, 1, 0])
        self.assertAlmostEqual(acc.result(), 1.0)

    def test_reset_cumulative(self):
        acc = Accuracy()
        acc.update([0, 1], [1, 0])
        acc.reset()
        self.assertAlmostEqual(acc.result(), 0.0)

    def test_reset_windowed(self):
        acc = Accuracy(window_size=5)
        acc.update([0, 1], [0, 0])
        acc.reset()
        self.assertAlmostEqual(acc.result(), 0.0)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(Accuracy().result(), 0.0)
        self.assertAlmostEqual(Accuracy(window_size=5).result(), 0.0)


# ---------------------------------------------------------------------------
# Streaming Precision
# ---------------------------------------------------------------------------

class TestStreamingPrecision(unittest.TestCase):

    def test_perfect_after_update(self):
        prec = Precision()
        y = np.array([0, 1, 2])
        prec.update(y, y)
        self.assertAlmostEqual(prec.result(), 1.0)

    def test_windowed(self):
        prec = Precision(window_size=4)
        prec.update([0, 0, 1, 1], [0, 0, 1, 1])  # perfect window
        self.assertAlmostEqual(prec.result(), 1.0)

    def test_reset(self):
        prec = Precision()
        prec.update([0, 1], [0, 1])
        prec.reset()
        self.assertAlmostEqual(prec.result(), 0.0)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(Precision().result(), 0.0)


# ---------------------------------------------------------------------------
# Streaming Recall
# ---------------------------------------------------------------------------

class TestStreamingRecall(unittest.TestCase):

    def test_perfect_after_update(self):
        rec = Recall()
        y = np.array([0, 1, 2])
        rec.update(y, y)
        self.assertAlmostEqual(rec.result(), 1.0)

    def test_reset(self):
        rec = Recall()
        rec.update([0, 1], [0, 1])
        rec.reset()
        self.assertAlmostEqual(rec.result(), 0.0)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(Recall().result(), 0.0)


# ---------------------------------------------------------------------------
# Streaming F1Score
# ---------------------------------------------------------------------------

class TestStreamingF1(unittest.TestCase):

    def test_perfect(self):
        f1 = F1Score()
        y = np.array([0, 1, 1, 0])
        f1.update(y, y)
        self.assertAlmostEqual(f1.result(), 1.0)

    def test_result_in_range(self):
        f1 = F1Score()
        f1.update([0, 1, 1, 0], [0, 1, 0, 0])
        result = f1.result()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_multiple_updates_consistent(self):
        f1_stream = F1Score()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        f1_stream.update(y_true[:3], y_pred[:3])
        f1_stream.update(y_true[3:], y_pred[3:])
        expected = f1_score(y_true, y_pred)
        self.assertAlmostEqual(f1_stream.result(), expected, places=8)

    def test_reset(self):
        f1 = F1Score()
        f1.update([0, 1], [0, 1])
        f1.reset()
        self.assertAlmostEqual(f1.result(), 0.0)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(F1Score().result(), 0.0)


# ---------------------------------------------------------------------------
# Streaming ConfusionMatrix
# ---------------------------------------------------------------------------

class TestStreamingConfusionMatrix(unittest.TestCase):

    def test_accumulated_shape(self):
        cm = ConfusionMatrix(n_classes=2)
        cm.update([0, 1, 0, 1], [0, 1, 1, 0])
        self.assertEqual(cm.result().shape, (2, 2))

    def test_accumulated_total(self):
        cm = ConfusionMatrix(n_classes=2)
        cm.update([0, 1, 0, 1], [0, 1, 1, 0])
        self.assertEqual(cm.result().sum(), 4)

    def test_multiple_chunks(self):
        cm = ConfusionMatrix(n_classes=2)
        cm.update([0, 1], [0, 1])
        cm.update([0, 1], [1, 0])
        self.assertEqual(cm.result().sum(), 4)

    def test_perfect_diagonal(self):
        cm = ConfusionMatrix(n_classes=3)
        cm.update([0, 1, 2], [0, 1, 2])
        np.testing.assert_array_equal(np.diag(cm.result()), [1, 1, 1])

    def test_out_of_range_ignored(self):
        cm = ConfusionMatrix(n_classes=2)
        cm.update([0, 1, 5], [0, 1, 5])   # 5 is out of range
        self.assertEqual(cm.result().sum(), 2)

    def test_reset_clears(self):
        cm = ConfusionMatrix(n_classes=2)
        cm.update([0, 1], [0, 1])
        cm.reset()
        np.testing.assert_array_equal(cm.result(), np.zeros((2, 2)))

    def test_matches_standalone(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 3, size=50)
        y_pred = rng.integers(0, 3, size=50)
        cm_stream = ConfusionMatrix(n_classes=3)
        cm_stream.update(y_true, y_pred)
        cm_batch = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(cm_stream.result(), cm_batch)

    def test_incremental_matches_batch(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 3, size=60)
        y_pred = rng.integers(0, 3, size=60)
        cm_stream = ConfusionMatrix(n_classes=3)
        for yt, yp in zip(np.array_split(y_true, 6), np.array_split(y_pred, 6)):
            cm_stream.update(yt, yp)
        cm_batch = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(cm_stream.result(), cm_batch)


# ---------------------------------------------------------------------------
# Streaming AUC
# ---------------------------------------------------------------------------

class TestStreamingAUC(unittest.TestCase):

    def test_perfect(self):
        auc = AUC()
        auc.update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(auc.result(), 1.0)

    def test_multiple_updates(self):
        auc_stream = AUC()
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
        auc_stream.update(y_true[:3], y_score[:3])
        auc_stream.update(y_true[3:], y_score[3:])
        expected = roc_auc_score(y_true, y_score)
        self.assertAlmostEqual(auc_stream.result(), expected, places=8)

    def test_windowed(self):
        auc = AUC(window_size=4)
        auc.update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(auc.result(), 1.0)

    def test_reset(self):
        auc = AUC()
        auc.update([0, 1], [0.1, 0.9])
        auc.reset()
        self.assertAlmostEqual(auc.result(), 0.0)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(AUC().result(), 0.0)


if __name__ == '__main__':
    unittest.main()
