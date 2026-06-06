"""
tests/test_stream.py
--------------------
Unit tests for framework.stream – StreamTrainer.
"""

import unittest
import numpy as np

from framework.pipeline import Pipeline
from framework.preprocessing import StandardScaler
from framework.tree import DecisionTreeClassifier
from framework.ensemble import RandomForestClassifier
from framework.metrics import Accuracy, F1Score
from framework.stream import StreamTrainer


def _make_pipeline(seed=0):
    return Pipeline([
        ('sc', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth=3, random_state=seed)),
    ])


def _make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y


class TestStreamTrainer(unittest.TestCase):

    def test_fit_chunk_returns_dict(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipeline(), metrics=[Accuracy()])
        record = trainer.fit_chunk(X[:50], y[:50])
        self.assertIn('chunk', record)
        self.assertIn('accuracy', record)

    def test_chunk_index_increments(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipeline())
        trainer.fit_chunk(X[:50], y[:50])
        trainer.fit_chunk(X[50:100], y[50:100])
        log = trainer.get_log()
        self.assertEqual(log[0]['chunk'], 0)
        self.assertEqual(log[1]['chunk'], 1)

    def test_log_length_matches_chunks(self):
        X, y = _make_data(n=300)
        trainer = StreamTrainer(_make_pipeline())
        for i in range(3):
            trainer.fit_chunk(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
        self.assertEqual(len(trainer.get_log()), 3)

    def test_score_chunk_does_not_advance_log(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipeline())
        trainer.fit_chunk(X[:100], y[:100])
        trainer.score_chunk(X[100:200], y[100:200])
        self.assertEqual(len(trainer.get_log()), 1)

    def test_reset_clears_log(self):
        X, y = _make_data()
        trainer = StreamTrainer(_make_pipeline())
        trainer.fit_chunk(X[:50], y[:50])
        trainer.reset()
        self.assertEqual(len(trainer.get_log()), 0)
        self.assertEqual(trainer._chunk_idx, 0)

    def test_multiple_metrics(self):
        X, y = _make_data()
        trainer = StreamTrainer(
            _make_pipeline(),
            metrics=[Accuracy(), F1Score()],
        )
        record = trainer.fit_chunk(X[:100], y[:100])
        self.assertIn('accuracy', record)
        self.assertIn('f1score', record)

    def test_accuracy_improves_over_chunks(self):
        """Accuracy on training data should be reasonable after enough chunks."""
        X, y = _make_data(n=500)
        trainer = StreamTrainer(_make_pipeline())
        for i in range(5):
            trainer.fit_chunk(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
        log = trainer.get_log()
        # Last chunk accuracy should be >0.5 for linearly separable data
        self.assertGreater(log[-1]['accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main()
