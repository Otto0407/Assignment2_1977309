"""
tests.py
--------
40 unit tests across the whole framework.

Run with:
    pytest tests.py -v
"""

import pytest
import numpy as np

from framework.preprocessing import StandardScaler, MinMaxScaler, Imputer
from framework.tree import DecisionTreeClassifier
from framework.ensemble import EnsembleClassifier, RandomForestClassifier
from framework.pipeline import Pipeline
from framework.stream import StreamTrainer
from framework.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    Accuracy, F1Score, ConfusionMatrix,
)


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

@pytest.fixture
def binary_data_large():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

@pytest.fixture
def rf_pipeline():
    return Pipeline([
        ('sc', StandardScaler()),
        ('clf', EnsembleClassifier(n_estimators=5, random_state=0)),
    ])

@pytest.fixture
def stream_trainer(rf_pipeline):
    return StreamTrainer(rf_pipeline, metrics=[Accuracy()])


class TestStandardScaler:

    def test_transform_zero_mean(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5.0, scale=3.0, size=(200, 4))
        Xt = StandardScaler().partial_fit(X).transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), np.zeros(4), atol=1e-10)

    def test_incremental_chan_equals_batch(self):
        """Two-chunk partial_fit must produce the same result as one-shot."""
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
        assert not np.any(np.isnan(Xt))


class TestMinMaxScaler:

    def test_range_0_1(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        Xt = MinMaxScaler().partial_fit(X).transform(X)
        assert Xt.min() >= 0.0
        assert Xt.max() <= 1.0


class TestImputer:

    def test_mean_basic(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        Xt = Imputer(strategy='mean').partial_fit(X).transform(X)
        assert Xt[0, 1] == pytest.approx(5.0)  # mean of [4, 6]

    def test_mean_incremental_equals_batch(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(120, 4))
        X[rng.integers(0, 120, 20), rng.integers(0, 4, 20)] = np.nan
        batch = Imputer(strategy='mean').partial_fit(X).transform(X)
        inc = Imputer(strategy='mean')
        for i in range(0, 120, 40):
            inc.partial_fit(X[i:i+40])
        np.testing.assert_allclose(inc.transform(X), batch, atol=1e-10)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            Imputer(strategy='mode').partial_fit(np.ones((5, 2)))


class TestDecisionTree:

    def test_training_accuracy_gini(self, binary_data_large):
        X, y = binary_data_large
        clf = DecisionTreeClassifier(max_depth=5, criterion='gini').fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.85

    def test_training_accuracy_entropy(self, binary_data_large):
        X, y = binary_data_large
        clf = DecisionTreeClassifier(max_depth=5, criterion='entropy').fit(X, y)
        assert np.mean(clf.predict(X) == y) > 0.85

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y = binary_data
        proba = DecisionTreeClassifier(max_depth=3).fit(X, y).predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_max_depth_0_is_leaf(self, binary_data):
        X, y = binary_data
        clf = DecisionTreeClassifier(max_depth=0).fit(X, y)
        assert clf.tree_.is_leaf

    def test_partial_fit_accumulates_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = DecisionTreeClassifier(max_depth=4)
        clf.partial_fit(X[:150], y[:150])
        clf.partial_fit(X[150:], y[150:])
        assert np.mean(clf.predict(X) == y) > 0.80

    def test_partial_fit_classes_argument_preserved(self, binary_data):
        X, y = binary_data
        clf = DecisionTreeClassifier(max_depth=3)
        clf.partial_fit(X[y == 0], y[y == 0], classes=np.array([0, 1]))
        assert 0 in clf.classes_
        assert 1 in clf.classes_

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            DecisionTreeClassifier().predict(np.ones((3, 2)))

    def test_zero_variance_feature_no_crash(self):
        rng = np.random.default_rng(5)
        X = np.column_stack([np.ones(50), rng.normal(size=50)])
        y = (X[:, 1] > 0).astype(int)
        preds = DecisionTreeClassifier(max_depth=3).fit(X, y).predict(X)
        assert not np.any(np.isnan(preds.astype(float)))


class TestEnsemble:

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y = binary_data
        clf = EnsembleClassifier(n_estimators=5, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_partial_fit_classes_preserved_across_chunks(self, binary_data):
        X, y = binary_data
        clf = EnsembleClassifier(n_estimators=5)
        clf.partial_fit(X[y == 0], y[y == 0], classes=np.array([0, 1]))
        clf.partial_fit(X[y == 1], y[y == 1], classes=np.array([0, 1]))
        assert clf.predict_proba(X).shape[1] == 2

    def test_rf_uses_sqrt_max_features(self, binary_data):
        X, y = binary_data
        clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        for est in clf.estimators_:
            assert est.max_features == 'sqrt'

    def test_streaming_accuracy_improves(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = EnsembleClassifier(n_estimators=5, random_state=0)
        clf.partial_fit(X[:100], y[:100], classes=np.array([0, 1]))
        acc1 = np.mean(clf.predict(X[:100]) == y[:100])
        for i in range(1, 4):
            clf.partial_fit(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
        acc4 = np.mean(clf.predict(X) == y)
        assert acc4 >= acc1 * 0.95

    def test_invalid_method_raises(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError):
            EnsembleClassifier(method='boosting').fit(X[:50], y[:50])


class TestPipeline:

    def test_fit_predict_shape(self, binary_data):
        X, y = binary_data
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', DecisionTreeClassifier(max_depth=3))])
        preds = pipe.fit(X, y).predict(X)
        assert preds.shape == (len(X),)

    def test_partial_fit_multiple_chunks_accuracy(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', EnsembleClassifier(n_estimators=5, random_state=0))])
        for i in range(3):
            pipe.partial_fit(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
        assert pipe.score(X, y) > 0.75

    def test_partial_fit_classes_param_forwarded(self, binary_data):
        X, y = binary_data
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', EnsembleClassifier(n_estimators=3))])
        pipe.partial_fit(X[y == 0], y[y == 0], classes=np.array([0, 1]))
        assert 1 in pipe.steps[-1][1].classes_

    def test_imputer_in_pipeline_fills_nan(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 3))
        X[rng.integers(0, 100, 10), rng.integers(0, 3, 10)] = np.nan
        y = (X[:, 0] > 0).astype(int)
        pipe = Pipeline([('imp', Imputer(strategy='mean')),
                         ('clf', DecisionTreeClassifier(max_depth=3))])
        preds = pipe.fit(X, y).predict(X)
        assert not np.any(np.isnan(preds.astype(float)))


class TestMetrics:

    def test_accuracy_perfect(self):
        y = np.array([0, 1, 2, 0, 1])
        assert accuracy_score(y, y) == 1.0

    def test_confusion_matrix_bincount_matches_loop(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, 50)
        y_pred = rng.integers(0, 3, 50)
        expected = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred):
            expected[t, p] += 1
        np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)

    def test_roc_auc_perfect(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert roc_auc_score(y, scores) == pytest.approx(1.0)

    def test_streaming_accuracy_cumulative(self):
        acc = Accuracy()
        acc.update(np.array([0, 1, 1]), np.array([0, 1, 1]))
        acc.update(np.array([0, 0]), np.array([0, 1]))
        assert acc.result() == pytest.approx(4 / 5)

    def test_streaming_f1_reset(self):
        f1 = F1Score()
        f1.update(np.array([0, 1, 1]), np.array([0, 1, 0]))
        f1.reset()
        assert f1.result() == 0.0

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
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        assert f1_score(y_true, y_pred, average='macro') == 0.0

    def test_roc_auc_worst(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert roc_auc_score(y, scores) == pytest.approx(0.0)


class TestStreamTrainer:

    def test_fit_chunk_returns_dict_with_accuracy(self, stream_trainer, binary_data):
        X, y = binary_data
        record = stream_trainer.fit_chunk(X, y)
        assert 'accuracy' in record
        assert 'chunk' in record

    def test_log_length_matches_chunks(self, stream_trainer, binary_data):
        X, y = binary_data
        for i in range(3):
            n = len(X) // 3
            stream_trainer.fit_chunk(X[i*n:(i+1)*n], y[i*n:(i+1)*n])
        assert len(stream_trainer.get_log()) == 3

    def test_score_chunk_does_not_advance_log(self, stream_trainer, binary_data):
        X, y = binary_data
        stream_trainer.fit_chunk(X, y)
        stream_trainer.score_chunk(X, y)
        assert len(stream_trainer.get_log()) == 1

    def test_reset_clears_log(self, stream_trainer, binary_data):
        X, y = binary_data
        stream_trainer.fit_chunk(X, y)
        stream_trainer.reset()
        assert len(stream_trainer.get_log()) == 0

class TestIntegration:

    def test_pipe_score_matches_accuracy_score(self, binary_data):
        X, y = binary_data
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', DecisionTreeClassifier(max_depth=4))])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert pipe.score(X, y) == pytest.approx(accuracy_score(y, preds))

    def test_nan_data_imputer_pipeline_no_nan_in_predictions(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(150, 4))
        X[rng.integers(0, 150, 15), rng.integers(0, 4, 15)] = np.nan
        y = (rng.normal(size=150) > 0).astype(int)
        pipe = Pipeline([('imp', Imputer(strategy='mean')),
                         ('sc', StandardScaler()),
                         ('clf', DecisionTreeClassifier(max_depth=3))])
        preds = pipe.fit(X, y).predict(X)
        assert not np.any(np.isnan(preds.astype(float)))

    def test_streaming_confusion_matrix_shape_over_pipeline(self, binary_data):
        X, y = binary_data
        pipe = Pipeline([('sc', StandardScaler()),
                         ('clf', EnsembleClassifier(n_estimators=5, random_state=0))])
        stream_cm = ConfusionMatrix(n_classes=2)
        for i in range(4):
            n = len(X) // 4
            Xc, yc = X[i*n:(i+1)*n], y[i*n:(i+1)*n]
            pipe.partial_fit(Xc, yc)
            stream_cm.update(yc, pipe.predict(Xc))
        assert stream_cm.result().shape == confusion_matrix(y, pipe.predict(X)).shape
