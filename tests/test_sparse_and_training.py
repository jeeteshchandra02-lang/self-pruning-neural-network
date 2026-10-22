"""Integration and true sparse-vs-masked-dense equivalence checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nn import MLP
from optim import Adam
from prune import GlobalPruner, SparseMLP, target_for_epoch
from train.runner import train_model


class SparseAndTrainingTests(unittest.TestCase):
    def test_sparse_equals_masked_dense_and_roundtrip(self):
        rng = np.random.default_rng(19)
        model = MLP(dims=(3, 7, 4, 2), seed=17)
        optimizer = Adam(model.parameters())
        GlobalPruner(model, optimizer, 'magnitude').prune_to(0.65)
        x = rng.normal(size=(5, 3))
        sparse = SparseMLP.from_model(model)
        np.testing.assert_allclose(sparse.predict_logits(x), model.predict_logits(x),
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(sparse.nnz(), model.active_weights())
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'model.npz')
            sparse.save(path)
            restored = SparseMLP.load(path)
            np.testing.assert_allclose(restored.predict_logits(x), model.predict_logits(x),
                                       rtol=1e-12, atol=1e-12)

    def test_mask_always_reaches_schedule_target(self):
        targets = [target_for_epoch(i, 25, .90) for i in range(25)]
        self.assertEqual(targets[-1], .9)
        self.assertTrue(all(a <= b for a, b in zip(targets, targets[1:])))
        self.assertTrue(all(0 <= a <= .9 for a in targets))

    def test_few_epoch_training_stable_and_sparsity_correct(self):
        _, _, curve, row, _ = train_model(epochs=5, seed=1, target=.50)
        self.assertTrue(all(np.isfinite(r['training_loss']) for r in curve))
        self.assertGreater(row['test_accuracy'], 0.6)
        self.assertEqual(row['active_weights'],
                         row['total_weights'] - round(.5 * row['total_weights']))
        self.assertGreater(curve[0]['training_loss'], curve[-1]['training_loss'])


if __name__ == '__main__':
    unittest.main()
