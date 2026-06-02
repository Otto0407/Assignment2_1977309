"""
tests/test_integration_pipeline_metrics.py
------------------------------------------
整合測試：確認 pipeline.py 與 metrics.py 的銜接正確。

涵蓋場景：
  1. pipe.score 與 accuracy_score 完全一致
  2. 串流場景：每 chunk 以 Accuracy 追蹤，結果合理
  3. ConfusionMatrix 串流累積 == batch 一次計算
  4. predict_proba → roc_auc_score 正確計算
  5. 串流 AUC 追蹤（每 chunk 更新）
  6. F1 == harmonic mean of precision/recall
  7. 串流 Precision 多 chunk == batch Precision
  8. NaN 資料 → Imputer pipeline → metrics 不出現 NaN
"""

import unittest
import numpy as np

from framework.pipeline import Pipeline
from framework.preprocessing import StandardScaler, MinMaxScaler, Imputer
from framework.tree import DecisionTreeClassifier
from framework.ensemble import RandomForestClassifier
from framework.metrics import (
    Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUC,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)


def _make_data(n=400, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=5.0, scale=2.0, size=(n, 4))
    y = (X[:, 0] + X[:, 2] > 10).astype(int)
    return X, y


class TestPipelineMetricsIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = _make_data()
        cls.rng = np.random.default_rng(99)

    # ------------------------------------------------------------------
    # 1. pipe.score 與 accuracy_score 完全一致
    # ------------------------------------------------------------------

    def test_pipe_score_matches_accuracy_score(self):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=4, random_state=0)),
        ])
        pipe.fit(self.X, self.y)
        pipe_score = pipe.score(self.X, self.y)
        metric_score = accuracy_score(self.y, pipe.predict(self.X))
        self.assertAlmostEqual(pipe_score, metric_score, places=12)

    # ------------------------------------------------------------------
    # 2. 串流 Accuracy 追蹤
    # ------------------------------------------------------------------

    def test_streaming_accuracy_tracking(self):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=3, random_state=1)),
        ])
        acc_metric = Accuracy()
        for Xc, yc in zip(np.array_split(self.X, 5), np.array_split(self.y, 5)):
            pipe.partial_fit(Xc, yc, classes=np.array([0, 1]))
            acc_metric.update(yc, pipe.predict(Xc))
        self.assertGreater(acc_metric.result(), 0.5)

    # ------------------------------------------------------------------
    # 3. ConfusionMatrix 串流累積 == batch
    # ------------------------------------------------------------------

    def test_confusion_matrix_streaming_equals_batch(self):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=5, random_state=2)),
        ])
        pipe.fit(self.X, self.y)
        cm_stream = ConfusionMatrix(n_classes=2)
        for Xc, yc in zip(np.array_split(self.X, 5), np.array_split(self.y, 5)):
            cm_stream.update(yc, pipe.predict(Xc))
        cm_batch = confusion_matrix(self.y, pipe.predict(self.X))
        np.testing.assert_array_equal(cm_stream.result(), cm_batch)

    # ------------------------------------------------------------------
    # 4. predict_proba → roc_auc_score
    # ------------------------------------------------------------------

    def test_predict_proba_to_auc(self):
        pipe = Pipeline([
            ('mm', MinMaxScaler()),
            ('rf', RandomForestClassifier(n_estimators=10, random_state=3)),
        ])
        pipe.fit(self.X, self.y)
        scores = pipe.predict_proba(self.X)[:, 1]
        auc = roc_auc_score(self.y, scores)
        self.assertGreater(auc, 0.8)
        self.assertLessEqual(auc, 1.0)

    # ------------------------------------------------------------------
    # 5. 串流 AUC 追蹤
    # ------------------------------------------------------------------

    def test_streaming_auc_tracking(self):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=5, random_state=4)),
        ])
        auc_metric = AUC()
        for Xc, yc in zip(np.array_split(self.X, 4), np.array_split(self.y, 4)):
            pipe.partial_fit(Xc, yc, classes=np.array([0, 1]))
            auc_metric.update(yc, pipe.predict_proba(Xc)[:, 1])
        result = auc_metric.result()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    # ------------------------------------------------------------------
    # 6. F1 == harmonic mean of P/R
    # ------------------------------------------------------------------

    def test_f1_equals_harmonic_mean_of_precision_recall(self):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=5, random_state=2)),
        ])
        pipe.fit(self.X, self.y)
        preds = pipe.predict(self.X)
        p = precision_score(self.y, preds)
        r = recall_score(self.y, preds)
        f1 = f1_score(self.y, preds)
        expected = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        self.assertAlmostEqual(f1, expected, places=8)

    # ------------------------------------------------------------------
    # 7. 串流 Precision 多 chunk == batch
    # ------------------------------------------------------------------

    def test_streaming_precision_matches_batch(self):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=4, random_state=5)),
        ])
        pipe.fit(self.X, self.y)
        prec_stream = Precision(average='macro')
        for Xc, yc in zip(np.array_split(self.X, 4), np.array_split(self.y, 4)):
            prec_stream.update(yc, pipe.predict(Xc))
        prec_batch = precision_score(self.y, pipe.predict(self.X), average='macro')
        self.assertAlmostEqual(prec_stream.result(), prec_batch, places=8)

    # ------------------------------------------------------------------
    # 8. NaN 資料 → Imputer pipeline → metrics 不出現 NaN
    # ------------------------------------------------------------------

    def test_nan_data_imputer_pipeline_metrics_no_nan(self):
        X_nan = self.X.copy()
        X_nan[self.rng.random(size=self.X.shape) < 0.1] = np.nan
        pipe = Pipeline([
            ('imp', Imputer(strategy='mean')),
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=3, random_state=6)),
        ])
        pipe.fit(X_nan, self.y)
        preds = pipe.predict(X_nan)
        acc = accuracy_score(self.y, preds)
        self.assertFalse(np.isnan(acc))
        self.assertGreater(acc, 0.5)
        cm = confusion_matrix(self.y, preds)
        self.assertEqual(cm.sum(), len(self.y))

    # ------------------------------------------------------------------
    # 9. windowed Accuracy 在串流場景中運作正確
    # ------------------------------------------------------------------

    def test_windowed_accuracy_in_streaming(self):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=3, random_state=7)),
        ])
        pipe.fit(self.X, self.y)
        acc_win = Accuracy(window_size=50)
        for Xc, yc in zip(np.array_split(self.X, 8), np.array_split(self.y, 8)):
            acc_win.update(yc, pipe.predict(Xc))
        result = acc_win.result()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
        self.assertLessEqual(len(acc_win._window), 50)

    # ------------------------------------------------------------------
    # 10. multiclass：pipe + metrics 形狀與數值正確
    # ------------------------------------------------------------------

    def test_multiclass_pipeline_metrics(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(300, 4))
        y = np.argmax(X[:, :3], axis=1)   # 3 classes
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=5, random_state=8)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        cm = confusion_matrix(y, preds)
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm.sum(), len(y))
        f1 = f1_score(y, preds, average='macro')
        self.assertGreater(f1, 0.3)


if __name__ == '__main__':
    unittest.main()
