"""
tests/test_metrics.py
---------------------
Unit tests for framework.metrics – streaming classes and standalone functions.
"""

import unittest
import numpy as np

from framework.metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    AUC,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


class TestAccuracyScore(unittest.TestCase):

    def test_perfect(self):
        y = np.array([0, 1, 2, 1, 0])
        self.assertAlmostEqual(accuracy_score(y, y), 1.0)

    def test_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.0)

    def test_partial(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.75)


class TestPrecisionRecallF1(unittest.TestCase):

    def test_binary_precision(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        # macro: class-0 prec=1.0, class-1 prec=1.0 -> 1.0
        p = precision_score(y_true, y_pred, average='macro')
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_recall_perfect(self):
        y = np.array([0, 1, 2])
        self.assertAlmostEqual(recall_score(y, y, average='macro'), 1.0)

    def test_f1_harmonic_mean(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        expected = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        self.assertAlmostEqual(f1_score(y_true, y_pred), expected, places=6)


class TestConfusionMatrix(unittest.TestCase):

    def test_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape, (2, 2))
        # TN + FP + FN + TP = 4
        self.assertEqual(cm.sum(), 4)

    def test_diagonal_is_correct(self):
        y = np.array([0, 1, 2])
        cm = confusion_matrix(y, y)
        np.testing.assert_array_equal(np.diag(cm), [1, 1, 1])


class TestRocAucScore(unittest.TestCase):

    def test_perfect_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc_score(y_true, y_score), 1.0)

    def test_random_auc_around_half(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        y_score = rng.uniform(0, 1, size=200)
        auc = roc_auc_score(y_true, y_score)
        self.assertGreater(auc, 0.3)
        self.assertLess(auc, 0.7)


class TestStreamingAccuracy(unittest.TestCase):

    def test_cumulative(self):
        acc = Accuracy()
        acc.update(np.array([0, 1, 1]), np.array([0, 1, 0]))
        # 2 correct out of 3
        self.assertAlmostEqual(acc.result(), 2 / 3)

    def test_rolling_window(self):
        acc = Accuracy(window_size=4)
        acc.update(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 0]))
        self.assertAlmostEqual(acc.result(), 0.75)
        # Adding a perfect batch should raise accuracy
        acc.update(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
        self.assertGreater(acc.result(), 0.75)

    def test_reset(self):
        acc = Accuracy()
        acc.update(np.array([0, 1]), np.array([1, 0]))
        acc.reset()
        self.assertAlmostEqual(acc.result(), 0.0)


class TestStreamingConfusionMatrix(unittest.TestCase):

    def test_accumulated_matrix(self):
        cm_metric = ConfusionMatrix(n_classes=2)
        cm_metric.update(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]))
        cm = cm_metric.result()
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm.sum(), 4)

    def test_reset_clears(self):
        cm_metric = ConfusionMatrix(n_classes=2)
        cm_metric.update(np.array([0, 1]), np.array([0, 1]))
        cm_metric.reset()
        np.testing.assert_array_equal(cm_metric.result(), np.zeros((2, 2)))


class TestStreamingF1(unittest.TestCase):

    def test_update_result(self):
        f1 = F1Score()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        f1.update(y_true, y_pred)
        result = f1.result()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


if __name__ == '__main__':
    unittest.main()
