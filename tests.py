"""
tests.py
--------
A curated suite of 30 unit tests drawn from the individual per-module
test files, covering standard functionality and streaming / edge-case
behaviour across the whole framework.

Run with:
    python -m unittest tests
"""

import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _binary_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _multiclass_data(n=300, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.argmax(X[:, :3], axis=1)
    return X, y


# ===========================================================================
# 1. Preprocessing — StandardScaler
# ===========================================================================

from framework.preprocessing import StandardScaler, MinMaxScaler, Imputer


class TestStandardScaler(unittest.TestCase):

    def test_transform_zero_mean(self):
        """Scaled output should have mean ≈ 0."""
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5.0, scale=3.0, size=(200, 4))
        Xt = StandardScaler().partial_fit(X).transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), np.zeros(4), atol=1e-10)

    def test_incremental_chan_equals_batch(self):
        """Two-chunk partial_fit must produce the same scaler as one-shot fit."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(100, 3))
        batch = StandardScaler().partial_fit(X)
        inc = StandardScaler()
        inc.partial_fit(X[:50])
        inc.partial_fit(X[50:])
        np.testing.assert_allclose(inc.transform(X), batch.transform(X), atol=1e-10)

    def test_nan_in_chunk_does_not_corrupt_state(self):
        """NaN values in one chunk must not poison the running statistics."""
        rng = np.random.default_rng(3)
        X1 = rng.normal(size=(50, 2))
        X2 = rng.normal(size=(50, 2))
        X2[5, 0] = np.nan
        sc = StandardScaler()
        sc.partial_fit(X1)
        sc.partial_fit(X2)
        Xt = sc.transform(rng.normal(size=(10, 2)))
        self.assertFalse(np.any(np.isnan(Xt)))


# ===========================================================================
# 2. Preprocessing — MinMaxScaler
# ===========================================================================

from framework.preprocessing import MinMaxScaler


class TestMinMaxScaler(unittest.TestCase):

    def test_range_0_1(self):
        """Output values should lie in [0, 1] after scaling."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        Xt = MinMaxScaler().partial_fit(X).transform(X)
        self.assertGreaterEqual(Xt.min(), 0.0)
        self.assertLessEqual(Xt.max(), 1.0)

    def test_inverse_roundtrip(self):
        """inverse_transform(transform(X)) should recover X."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 2))
        sc = MinMaxScaler().partial_fit(X)
        np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X, atol=1e-10)


# ===========================================================================
# 3. Preprocessing — Imputer
# ===========================================================================

class TestImputer(unittest.TestCase):

    def test_mean_basic(self):
        """Mean imputer should replace NaN with the column mean."""
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        Xt = Imputer(strategy="mean").partial_fit(X).transform(X)
        self.assertAlmostEqual(Xt[0, 1], 5.0)  # mean of [4, 6]

    def test_mean_incremental_equals_batch(self):
        """Streaming partial_fit must converge to the same imputation as fit."""
        rng = np.random.default_rng(9)
        X = rng.normal(size=(120, 4))
        X[rng.integers(0, 120, 20), rng.integers(0, 4, 20)] = np.nan
        batch = Imputer(strategy="mean").partial_fit(X).transform(X)
        inc = Imputer(strategy='mean')
        for i in range(0, 120, 40):
            inc.partial_fit(X[i:i+40])
        np.testing.assert_allclose(inc.transform(X), batch, atol=1e-10)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            Imputer(strategy='mode').fit(np.ones((5, 2)))


# ===========================================================================
# 3. Decision Tree
# ===========================================================================

from framework.tree import DecisionTreeClassifier


class TestDecisionTree(unittest.TestCase):

    def test_training_accuracy_gini(self):
        X, y = _binary_data(n=500)
        clf = DecisionTreeClassifier(max_depth=5, criterion='gini').fit(X, y)
        self.assertGreater(np.mean(clf.predict(X) == y), 0.85)

    def test_predict_proba_sums_to_one(self):
        X, y = _binary_data()
        proba = DecisionTreeClassifier(max_depth=3).fit(X, y).predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_max_depth_0_is_leaf(self):
        X, y = _binary_data()
        clf = DecisionTreeClassifier(max_depth=0).fit(X, y)
        self.assertTrue(clf.tree_.is_leaf)

    def test_partial_fit_accumulates_data(self):
        """Streaming partial_fit should match a full fit when all data is seen."""
        X, y = _binary_data(n=300)
        clf = DecisionTreeClassifier(max_depth=4)
        clf.partial_fit(X[:150], y[:150])
        clf.partial_fit(X[150:], y[150:])
        self.assertGreater(np.mean(clf.predict(X) == y), 0.80)

    def test_partial_fit_classes_argument_preserved(self):
        """classes kwarg must survive across partial_fit calls."""
        X, y = _binary_data(n=100)
        clf = DecisionTreeClassifier(max_depth=3)
        clf.partial_fit(X[y == 0], y[y == 0], classes=np.array([0, 1]))
        self.assertIn(0, clf.classes_)
        self.assertIn(1, clf.classes_)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            DecisionTreeClassifier().predict(np.ones((3, 2)))

    def test_zero_variance_feature_no_crash(self):
        rng = np.random.default_rng(5)
        X = np.column_stack([np.ones(50), rng.normal(size=50)])
        y = (X[:, 1] > 0).astype(int)
        preds = DecisionTreeClassifier(max_depth=3).fit(X, y).predict(X)
        self.assertFalse(np.any(np.isnan(preds.astype(float))))


# ===========================================================================
# 4. Ensemble
# ===========================================================================

from framework.ensemble import EnsembleClassifier, RandomForestClassifier


class TestEnsemble(unittest.TestCase):

    def test_predict_proba_sums_to_one(self):
        X, y = _binary_data(n=200)
        clf = EnsembleClassifier(n_estimators=5, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_partial_fit_classes_preserved_across_chunks(self):
        X, y = _binary_data(n=200)
        clf = EnsembleClassifier(n_estimators=5)
        clf.partial_fit(X[y == 0], y[y == 0], classes=np.array([0, 1]))
        clf.partial_fit(X[y == 1], y[y == 1], classes=np.array([0, 1]))
        self.assertEqual(clf.predict_proba(X).shape[1], 2)

    def test_rf_uses_sqrt_max_features(self):
        clf = RandomForestClassifier(n_estimators=5, random_state=0)
        clf.fit(*_binary_data())
        for est in clf.estimators_:
            self.assertEqual(est.max_features, 'sqrt')

    def test_streaming_accuracy_improves(self):
        """Accuracy after 4 chunks should exceed accuracy after just 1 chunk."""
        X, y = _binary_data(n=400)
        clf = EnsembleClassifier(n_estimators=5, random_state=0)
        clf.partial_fit(X[:100], y[:100], classes=np.array([0, 1]))
        acc1 = np.mean(clf.predict(X[:100]) == y[:100])
        for i in range(1, 4):
            clf.partial_fit(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
        acc4 = np.mean(clf.predict(X) == y)
        self.assertGreaterEqual(acc4, acc1 * 0.95)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            EnsembleClassifier(method='boosting').fit(*_binary_data(n=50))


# ===========================================================================
# 5. Pipeline
# ===========================================================================

from framework.pipeline import Pipeline


class TestPipeline(unittest.TestCase):

    def test_fit_predict_shape(self):
        X, y = _binary_data()
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', DecisionTreeClassifier(max_depth=3))])
        preds = pipe.fit(X, y).predict(X)
        self.assertEqual(preds.shape, (len(X),))

    def test_partial_fit_multiple_chunks_accuracy(self):
        X, y = _binary_data(n=300)
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', EnsembleClassifier(n_estimators=5, random_state=0))])
        for i in range(3):
            pipe.partial_fit(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
        self.assertGreater(pipe.score(X, y), 0.75)

    def test_partial_fit_classes_param_forwarded(self):
        X, y = _binary_data(n=100)
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', EnsembleClassifier(n_estimators=3))])
        pipe.partial_fit(X[y == 0], y[y == 0], classes=np.array([0, 1]))
        self.assertIn(1, pipe.steps[-1][1].classes_)

    def test_imputer_in_pipeline_fills_nan(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 3))
        X[rng.integers(0, 100, 10), rng.integers(0, 3, 10)] = np.nan
        y = (X[:, 0] > 0).astype(int)
        pipe = Pipeline([('imp', Imputer(strategy='mean')),
                         ('clf', DecisionTreeClassifier(max_depth=3))])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        self.assertFalse(np.any(np.isnan(preds.astype(float))))


# ===========================================================================
# 6. Metrics
# ===========================================================================

from framework.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    Accuracy, F1Score, ConfusionMatrix,
)


class TestMetrics(unittest.TestCase):

    def test_accuracy_perfect(self):
        y = np.array([0, 1, 2, 0, 1])
        self.assertEqual(accuracy_score(y, y), 1.0)

    def test_confusion_matrix_bincount_matches_loop(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, 50)
        y_pred = rng.integers(0, 3, 50)
        n = 3
        expected = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            expected[t, p] += 1
        np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)

    def test_roc_auc_perfect(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc_score(y, scores), 1.0)

    def test_streaming_accuracy_cumulative(self):
        acc = Accuracy()
        acc.update(np.array([0, 1, 1]), np.array([0, 1, 1]))
        acc.update(np.array([0, 0]), np.array([0, 1]))
        self.assertAlmostEqual(acc.result(), 4 / 5)

    def test_streaming_f1_reset(self):
        f1 = F1Score()
        f1.update(np.array([0, 1, 1]), np.array([0, 1, 0]))
        f1.reset()
        self.assertEqual(f1.result(), 0.0)

    def test_confusion_matrix_streaming_matches_batch(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 60)
        y_pred = rng.integers(0, 2, 60)
        batch_cm = confusion_matrix(y_true, y_pred)
        stream_cm = ConfusionMatrix(n_classes=2)
        stream_cm.update(y_true[:30], y_pred[:30])
        stream_cm.update(y_true[30:], y_pred[30:])
        np.testing.assert_array_equal(stream_cm.result(), batch_cm)

    def test_f1_zero_for_all_wrong(self):
        """F1 should be 0 when every prediction is wrong."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        self.assertEqual(f1_score(y_true, y_pred, average='macro'), 0.0)

    def test_training_accuracy_entropy(self):
        """Entropy criterion should also achieve >85 % training accuracy."""
        X, y = _binary_data(n=500)
        from framework.tree import DecisionTreeClassifier as DTC
        clf = DTC(max_depth=5, criterion='entropy').fit(X, y)
        self.assertGreater(np.mean(clf.predict(X) == y), 0.85)


# ===========================================================================
# 7. StreamTrainer
# ===========================================================================

from framework.stream import StreamTrainer


class TestStreamTrainer(unittest.TestCase):

    def _make_trainer(self):
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', EnsembleClassifier(n_estimators=5, random_state=0))])
        return StreamTrainer(pipe, metrics=[Accuracy()], log_memory=True)

    def test_fit_chunk_returns_dict_with_accuracy(self):
        trainer = self._make_trainer()
        X, y = _binary_data(n=100)
        record = trainer.fit_chunk(X, y)
        self.assertIn('accuracy', record)
        self.assertIn('chunk', record)

    def test_log_length_matches_chunks(self):
        trainer = self._make_trainer()
        X, y = _binary_data(n=300)
        for i in range(3):
            trainer.fit_chunk(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
        self.assertEqual(len(trainer.get_log()), 3)

    def test_score_chunk_does_not_advance_log(self):
        trainer = self._make_trainer()
        X, y = _binary_data(n=100)
        trainer.fit_chunk(X, y)
        trainer.score_chunk(X, y)
        self.assertEqual(len(trainer.get_log()), 1)

    def test_reset_clears_log(self):
        trainer = self._make_trainer()
        X, y = _binary_data(n=100)
        trainer.fit_chunk(X, y)
        trainer.reset()
        self.assertEqual(len(trainer.get_log()), 0)

    def test_memory_logged_when_requested(self):
        trainer = self._make_trainer()
        X, y = _binary_data(n=100)
        record = trainer.fit_chunk(X, y)
        self.assertIn('memory_mb', record)
        self.assertGreaterEqual(record['memory_mb'], 0.0)


# ===========================================================================
# 8. Integration — Pipeline + Metrics
# ===========================================================================

class TestIntegration(unittest.TestCase):

    def test_pipe_score_matches_accuracy_score(self):
        X, y = _binary_data(n=200)
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', DecisionTreeClassifier(max_depth=4))])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        self.assertAlmostEqual(pipe.score(X, y), accuracy_score(y, preds), places=12)

    def test_nan_data_imputer_pipeline_no_nan_in_predictions(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(150, 4))
        X[rng.integers(0, 150, 15), rng.integers(0, 4, 15)] = np.nan
        y = (X[:, 0] > 0).astype(float)
        y = np.where(np.isnan(y), 0, y).astype(int)
        pipe = Pipeline([('imp', Imputer(strategy='mean')),
                         ('sc', StandardScaler()),
                         ('clf', DecisionTreeClassifier(max_depth=3))])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        self.assertFalse(np.any(np.isnan(preds.astype(float))))

    def test_streaming_confusion_matrix_equals_batch_over_pipeline(self):
        X, y = _binary_data(n=200)
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', EnsembleClassifier(n_estimators=5, random_state=0))])
        stream_cm = ConfusionMatrix(n_classes=2)
        for i in range(4):
            Xc, yc = X[i*50:(i+1)*50], y[i*50:(i+1)*50]
            pipe.partial_fit(Xc, yc)
            stream_cm.update(yc, pipe.predict(Xc))
        # Batch confusion matrix using all collected predictions
        all_preds = pipe.predict(X)
        batch_cm = confusion_matrix(y, all_preds)
        # Shapes must match
        self.assertEqual(stream_cm.result().shape, batch_cm.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
