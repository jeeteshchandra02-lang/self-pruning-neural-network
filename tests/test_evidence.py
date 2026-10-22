"""Committed empirical artifacts must agree with calculations, not hand claims."""
import csv
import unittest
from pathlib import Path
import numpy as np

from prune import SparseMLP
from train.runner import ROOT
from train.sweep import summarize, generate_claim


class EvidenceTests(unittest.TestCase):
    def test_paired_sweep_csv_summary_claim_checkpoint_agree(self):
        raw = ROOT / 'artifacts' / 'raw'
        with (raw / 'sweep_results.csv').open(newline='') as f:
            data = list(csv.DictReader(f))
        numeric = {'seed', 'target_sparsity', 'test_accuracy', 'achieved_sparsity',
                   'total_weights', 'active_weights', 'sparse_macs_per_sample',
                   'dense_macs_per_sample', 'validation_accuracy'}
        converted = [{key: (int(value) if key in ('seed', 'total_weights', 'active_weights', 'sparse_macs_per_sample', 'dense_macs_per_sample') else
                            float(value) if key in numeric else value)
                      for key, value in record.items()}
                     for record in data]
        self.assertEqual(len(converted), 5 * 5 * 2)
        keys = {(r['criterion'], r['target_sparsity'], r['seed']) for r in converted}
        self.assertEqual(len(keys), len(converted))
        for r in converted:
            self.assertEqual(r['active_weights'], r['sparse_macs_per_sample'])
            self.assertEqual(r['total_weights'], r['dense_macs_per_sample'])
            self.assertAlmostEqual(1 - r['active_weights'] / r['total_weights'],
                                   r['achieved_sparsity'])
        with (raw / 'sweep_summary.csv').open(newline='') as f:
            saved_summary = list(csv.DictReader(f))
        expected = summarize(converted)
        self.assertEqual(len(expected), len(saved_summary))
        for predicted, saved in zip(expected, saved_summary):
            for key, value in predicted.items():
                if isinstance(value, str):
                    self.assertEqual(value, saved[key])
                else:
                    self.assertAlmostEqual(float(value), float(saved[key]), places=10)
        self.assertEqual((raw / 'falsifiable_claim.txt').read_text(),
                         generate_claim(converted))
        sparse = SparseMLP.load(str(ROOT / 'artifacts' / 'sparse_model.npz'))
        self.assertEqual(sparse.nnz(), 1123)
        self.assertEqual(sparse.layers[0].in_features, 64)
        self.assertEqual(sparse.layers[-1].out_features, 10)
        self.assertTrue(np.all(np.isfinite(sparse.predict_logits(
            np.zeros((2, 64))))))


if __name__ == '__main__':
    unittest.main()
