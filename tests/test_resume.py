"""Bit-for-bit epoch-boundary resume versus uninterrupted training."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from train.runner import TrainingSession


class ResumeTests(unittest.TestCase):
    def test_full_optimizer_rng_mask_and_regrowth_resume_exact(self):
        kwargs = dict(seed=2, epochs=6, target=.6, batch_size=96,
                      regrow_swaps=3)
        full = TrainingSession(**kwargs)
        while full.epoch < full.epochs:
            full.run_epoch()
        broken = TrainingSession(**kwargs)
        for _ in range(3):
            broken.run_epoch()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'training_state.npz'
            broken.save_checkpoint(path)
            resumed = TrainingSession.load_checkpoint(path)
            self.assertEqual(resumed.epoch, 3)
            while resumed.epoch < resumed.epochs:
                resumed.run_epoch()
            for key, expected in full.model.state_dict().items():
                np.testing.assert_array_equal(resumed.model.state_dict()[key], expected)
            for key, expected in full.optimizer.state_dict().items():
                np.testing.assert_array_equal(resumed.optimizer.state_dict()[key], expected)
            for key, expected in full.pruner.state_dict().items():
                np.testing.assert_array_equal(resumed.pruner.state_dict()[key], expected)
            self.assertEqual(full.curve, resumed.curve)
            self.assertEqual(full.swaps_completed, resumed.swaps_completed)
            self.assertEqual(full.finish()[3], resumed.finish()[3])

    def test_modified_data_rejected(self):
        session = TrainingSession(epochs=3)
        session.run_epoch()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'checkpoint.npz'
            session.save_checkpoint(path)
            broken = [(a.copy(), b.copy()) for a, b in session.dataset]
            broken[0][0][0, 0] += 0.5
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                TrainingSession.load_checkpoint(path, dataset=broken)


if __name__ == '__main__':
    unittest.main()
