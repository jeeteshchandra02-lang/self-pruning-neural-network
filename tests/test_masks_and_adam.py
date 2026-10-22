"""Key adversarial checks: masked weights, poisoned state and revival."""
import unittest
import numpy as np

from engine import Tensor
from nn import MLP, Linear
from optim import Adam
from prune import GlobalPruner


class MaskAdamTests(unittest.TestCase):
    def test_masked_forward_and_gradient(self):
        layer = Linear(2, 2, np.random.default_rng(0))
        layer.weight.data[:] = np.array([[2., 30.], [5., 7.]])
        layer.bias.data[:] = 0
        layer.mask[0, 1] = False
        x = np.array([[1., 2.], [3., 4.]])
        output = layer(Tensor(x)).sum()
        output.backward()
        effective = layer.weight.data * layer.mask
        np.testing.assert_allclose(layer.dense_logits(x), x @ effective)
        expected = np.array([[4., 0.], [6., 6.]])
        np.testing.assert_array_equal(layer.weight.grad, expected)
        np.testing.assert_array_equal(layer.bias.grad, np.array([2., 2.]))
        layer.weight.data[0, 1] = 9_999_999
        np.testing.assert_allclose(layer.dense_logits(x), x @ effective)

    def test_poisoned_moments_never_move_frozen_weight(self):
        p = Tensor(np.array([[0.9, 0.2]]), requires_grad=True)
        mask = np.array([[True, True]])
        adam = Adam([(p, mask)], lr=0.01)
        p.grad[:] = np.array([[2., -3.]])
        adam.step()
        state = adam.state_for(p)
        self.assertTrue(np.any(state.first != 0))
        removed = np.array([[False, True]])
        mask[removed] = False
        # Deliberately poison history; reset_coordinates MUST clear all state.
        state.first[removed] = 12345.
        state.second[removed] = 54321.
        state.age[removed] = 800
        adam.reset_coordinates(p, removed)
        for _ in range(5):
            p.zero_grad()
            p.grad[0, 0] = 1.0
            adam.step()
            self.assertEqual(p.data[0, 1], 0.0)
            self.assertEqual(p.grad[0, 1], 0.0)
            self.assertEqual(state.first[0, 1], 0.0)
            self.assertEqual(state.second[0, 1], 0.0)
            self.assertEqual(state.age[0, 1], 0)

    def test_adam_rejects_an_illegal_masked_gradient(self):
        p = Tensor([[1.0, 2.0]], requires_grad=True)
        mask = np.array([[True, False]])
        adam = Adam([(p, mask)])
        p.grad[:] = [[1.0, 9.0]]
        with self.assertRaisesRegex(AssertionError, 'Masked parameter'):
            adam.step()

    def test_revived_coordinate_first_step_matches_fresh_adam(self):
        p = Tensor(np.array([[8.0]]), requires_grad=True)
        mask = np.array([[True]])
        adam = Adam([(p, mask)], lr=0.03)
        for _ in range(4):
            p.grad[:] = 2.0
            adam.step()
        where = np.array([[True]])
        mask[where] = False
        adam.reset_coordinates(p, where)
        for _ in range(3):
            p.zero_grad()
            adam.step()
        mask[where] = True
        adam.reset_coordinates(p, where)  # documented revival = weight zero
        self.assertEqual(adam.state_for(p).age[0, 0], 0)
        p.grad[:] = 4.0
        adam.step()
        expected = -0.03 * 4.0 / (abs(4.0) + 1e-8)
        self.assertAlmostEqual(p.data[0, 0], expected, places=12)
        self.assertEqual(adam.state_for(p).age[0, 0], 1)

    def test_pruner_budget_with_equal_scores(self):
        model = MLP(dims=(2, 4, 2), seed=5)
        optimizer = Adam(model.parameters())
        pruner = GlobalPruner(model, optimizer, criterion='taylor',
                              min_incoming=0, min_outgoing=0)
        # Scores all tied at 0, must STILL hit exact integer budget.
        count = pruner.prune_to(0.5)
        self.assertEqual(count, model.total_weights() // 2)
        self.assertEqual(model.active_weights(), model.total_weights() // 2)
        self.assertEqual(pruner.prune_to(0.5), 0)
        self.assertTrue(all(np.all(layer.weight.data[~layer.mask] == 0)
                            for layer in model.linear_layers()))
        self.assertEqual(pruner.prune_to(0.75),
                         round(.75 * model.total_weights()) - count)

    def test_prune_then_revive_connection_gradient_is_nonzero(self):
        model = MLP(dims=(2, 2), seed=3)
        optimizer = Adam(model.parameters())
        pruner = GlobalPruner(model, optimizer)
        layer = model.linear_layers()[0]
        layer.mask[0, 0] = False
        optimizer.reset_coordinates(layer.weight,
                                    np.array([[True, False], [False, False]]))
        x = Tensor([[2., 0.]])
        model.zero_grad()
        layer(x).sum().backward()
        self.assertEqual(layer.weight.grad[0, 0], 0.0)
        pruner.revive(layer, (0, 0))
        model.zero_grad()
        layer(x).sum().backward()
        self.assertEqual(layer.weight.grad[0, 0], 2.0)
        optimizer.step()
        self.assertNotEqual(layer.weight.data[0, 0], 0.0)
        self.assertEqual(optimizer.state_for(layer.weight).age[0, 0], 1)


if __name__ == '__main__':
    unittest.main()
