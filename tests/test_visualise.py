"""
tests/test_visualise.py
-----------------------
Unit tests for framework.visualise.

所有測試使用 matplotlib Agg 後端（非互動），以 save_path 寫入暫存 PNG
確認圖形能正常產生，並驗證錯誤輸入時的防護行為。

涵蓋：
  plot_metric_over_time
    - 基本呼叫、儲存成檔案
    - 空列表不崩潰
    - 單一值不崩潰
    - title / ylabel 客製化
    - 大量資料點

  compare_models
    - 基本呼叫、儲存成檔案
    - metric1/metric2 長度不同 → ValueError
    - labels 少於 2 → ValueError
    - 單一 chunk

  plot_predictions_vs_ground_truth
    - 基本呼叫、儲存成檔案
    - y_true/y_pred 長度不同 → ValueError（明確訊息）
    - 單一樣本
    - 多類別

  plot_confusion_matrix
    - 基本 2×2、儲存成檔案
    - 自訂 class_names
    - 全零矩陣（不崩潰）
    - 3×3 多類別
    - class_names 長度不符 → ValueError（明確訊息）
    - class_names=None 自動填充
"""

import os
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')   # 測試環境不開視窗
import numpy as np

from framework.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
    plot_confusion_matrix,
)


def _tmppath(suffix='.png'):
    """建立一個暫存路徑（自動清理由呼叫者負責）。"""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.close()
    return f.name


def _saved_ok(path: str) -> bool:
    """檔案存在且大小 > 0。"""
    return os.path.exists(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# plot_metric_over_time
# ---------------------------------------------------------------------------

class TestPlotMetricOverTime(unittest.TestCase):

    def test_basic_saves_file(self):
        path = _tmppath()
        try:
            plot_metric_over_time([0.5, 0.6, 0.7, 0.8], save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_empty_list_no_crash(self):
        path = _tmppath()
        try:
            plot_metric_over_time([], save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_single_value_no_crash(self):
        path = _tmppath()
        try:
            plot_metric_over_time([0.9], save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_custom_title_ylabel(self):
        path = _tmppath()
        try:
            plot_metric_over_time(
                [0.1, 0.5, 0.9],
                title='My Title',
                ylabel='Loss',
                save_path=path,
            )
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_large_sequence(self):
        path = _tmppath()
        try:
            values = list(np.linspace(0, 1, 100))
            plot_metric_over_time(values, save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_returns_none(self):
        path = _tmppath()
        try:
            ret = plot_metric_over_time([0.5], save_path=path)
            self.assertIsNone(ret)
        finally:
            if os.path.exists(path): os.unlink(path)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels(unittest.TestCase):

    def test_basic_saves_file(self):
        path = _tmppath()
        try:
            compare_models(
                [0.5, 0.6, 0.7],
                [0.55, 0.65, 0.72],
                labels=['Tree', 'RF'],
                save_path=path,
            )
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_unequal_lengths_raises(self):
        with self.assertRaises(ValueError) as ctx:
            compare_models([0.5, 0.6, 0.7], [0.5, 0.6], ['A', 'B'],
                           save_path='/dev/null')
        self.assertIn('same length', str(ctx.exception))

    def test_too_few_labels_raises(self):
        with self.assertRaises(ValueError) as ctx:
            compare_models([0.5], [0.6], ['OnlyOne'], save_path='/dev/null')
        self.assertIn('two', str(ctx.exception))

    def test_single_chunk(self):
        path = _tmppath()
        try:
            compare_models([0.5], [0.6], ['A', 'B'], save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_custom_title_ylabel(self):
        path = _tmppath()
        try:
            compare_models(
                [0.5, 0.6], [0.4, 0.5],
                labels=['Model 1', 'Model 2'],
                title='Comparison',
                ylabel='Accuracy',
                save_path=path,
            )
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_returns_none(self):
        path = _tmppath()
        try:
            ret = compare_models([0.5], [0.6], ['A', 'B'], save_path=path)
            self.assertIsNone(ret)
        finally:
            if os.path.exists(path): os.unlink(path)


# ---------------------------------------------------------------------------
# plot_predictions_vs_ground_truth
# ---------------------------------------------------------------------------

class TestPlotPredictionsVsGroundTruth(unittest.TestCase):

    def test_basic_saves_file(self):
        path = _tmppath()
        try:
            plot_predictions_vs_ground_truth(
                np.array([0, 1, 1, 0, 1]),
                np.array([0, 1, 0, 0, 1]),
                save_path=path,
            )
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_unequal_lengths_raises_clear_message(self):
        with self.assertRaises(ValueError) as ctx:
            plot_predictions_vs_ground_truth(
                np.array([0, 1, 2]),
                np.array([0, 1]),
                save_path='/dev/null',
            )
        msg = str(ctx.exception)
        # 錯誤訊息應清楚說明長度不符（非 matplotlib 內部訊息）
        self.assertTrue('y_true' in msg or 'same length' in msg or '3' in msg)

    def test_single_sample(self):
        path = _tmppath()
        try:
            plot_predictions_vs_ground_truth(
                np.array([1]),
                np.array([0]),
                save_path=path,
            )
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_multiclass(self):
        path = _tmppath()
        try:
            plot_predictions_vs_ground_truth(
                np.array([0, 1, 2, 0, 1, 2]),
                np.array([0, 2, 2, 1, 1, 0]),
                title='Multiclass',
                save_path=path,
            )
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_perfect_predictions(self):
        path = _tmppath()
        try:
            y = np.array([0, 1, 0, 1, 1])
            plot_predictions_vs_ground_truth(y, y, save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_returns_none(self):
        path = _tmppath()
        try:
            ret = plot_predictions_vs_ground_truth(
                np.array([0, 1]), np.array([0, 1]), save_path=path
            )
            self.assertIsNone(ret)
        finally:
            if os.path.exists(path): os.unlink(path)


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix(unittest.TestCase):

    def test_basic_2x2_saves_file(self):
        path = _tmppath()
        try:
            cm = np.array([[50, 5], [3, 42]])
            plot_confusion_matrix(cm, save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_custom_class_names(self):
        path = _tmppath()
        try:
            cm = np.array([[10, 2], [1, 7]])
            plot_confusion_matrix(cm, class_names=['cat', 'dog'], save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_all_zero_matrix_no_crash(self):
        """全零矩陣：thresh=0，text 顏色判斷不應崩潰。"""
        path = _tmppath()
        try:
            plot_confusion_matrix(np.zeros((2, 2), dtype=int), save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_3x3_multiclass(self):
        path = _tmppath()
        try:
            cm = np.array([[10, 1, 0], [2, 8, 1], [0, 1, 9]])
            plot_confusion_matrix(cm, class_names=['A', 'B', 'C'], save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_class_names_length_mismatch_raises(self):
        """class_names 長度不符應給出清楚的 ValueError。"""
        cm = np.eye(3, dtype=int)
        with self.assertRaises(ValueError) as ctx:
            plot_confusion_matrix(cm, class_names=['a', 'b'], save_path='/dev/null')
        msg = str(ctx.exception)
        self.assertTrue('3' in msg or 'class_names' in msg or 'confusion matrix' in msg.lower())

    def test_none_class_names_auto_filled(self):
        """class_names=None 時應自動填充 '0', '1', ..."""
        path = _tmppath()
        try:
            plot_confusion_matrix(np.eye(3, dtype=int), class_names=None, save_path=path)
            self.assertTrue(_saved_ok(path))
        finally:
            if os.path.exists(path): os.unlink(path)

    def test_returns_none(self):
        path = _tmppath()
        try:
            ret = plot_confusion_matrix(np.eye(2, dtype=int), save_path=path)
            self.assertIsNone(ret)
        finally:
            if os.path.exists(path): os.unlink(path)


if __name__ == '__main__':
    unittest.main()
