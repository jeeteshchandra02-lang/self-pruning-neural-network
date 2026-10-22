"""Interviewer-style regression tests, not only easy happy paths."""
import unittest
import numpy as np

from engine import Tensor, cross_entropy
from nn import MLP, Linear
from optim import Adam
from prune import GlobalPruner


class AdversarialTests(unittest.TestCase):
    def test_illegal_prune_after_forward_refused_and_backward_recovers(self):
        model = MLP(dims=(2, 4, 2), seed=0)
        pruner = GlobalPruner(model, Adam(model.parameters()),
                              min_incoming=0, min_outgoing=0)
        out = model(Tensor([[1., 2.]]))
        with self.assertRaisesRegex(RuntimeError, 'in flight|before backward'):
            pruner.prune_to(0.25)
        with self.assertRaisesRegex(RuntimeError, 'in flight|before backward'):
            pruner.optimizer.step()
        out.sum().backward()
        pruner.prune_to(0.25)
        self.assertEqual(model.total_weights() - model.active_weights(), 4)

    def test_direct_mask_edit_before_backward_is_detected_without_grads(self):
        layer = Linear(2, 2, np.random.default_rng(0))
        graph = layer(Tensor([[1., 2.]])).sum()
        layer.mask[0, 0] = False
        with self.assertRaisesRegex(RuntimeError, 'changed between forward and backward'):
            graph.backward()
        np.testing.assert_array_equal(layer.weight.grad, np.zeros((2, 2)))
        layer.mask[0, 0] = True
        graph.backward()  # restored snapshot: same untouched graph is safe
        self.assertEqual(layer.weight._pending_graphs, 0)

    def test_direct_weight_mutation_before_backward_is_detected(self):
        layer = Linear(2, 3, np.random.default_rng(2))
        original = layer.weight.data.copy()
        graph = layer(Tensor([[0.5, -2.]])).sum()
        layer.weight.data[0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, 'changed between forward and backward'):
            graph.backward()
        layer.weight.data[:] = original
        graph.backward()

    def test_masked_weight_and_input_finite_difference(self):
        layer = Linear(2, 2, np.random.default_rng(9))
        layer.mask[1, 0] = False
        layer.weight.data[1, 0] = 0.0
        x = Tensor([[.7, -.4]], requires_grad=True)
        layer(x).sum().backward()
        saved_weight_gradient = layer.weight.grad.copy()
        saved_input_gradient = x.grad.copy()
        eps = 1e-6
        def f():
            return float(layer.dense_logits(x.data).sum())
        for row in range(2):
            for col in range(2):
                original = layer.weight.data[row, col]
                layer.weight.data[row, col] = original + eps
                positive = f()
                layer.weight.data[row, col] = original - eps
                negative = f()
                layer.weight.data[row, col] = original
                np.testing.assert_allclose(saved_weight_gradient[row, col],
                                           (positive - negative) / (2 * eps), atol=1e-9)
        for col in range(2):
            original = x.data[0, col]
            x.data[0, col] = original + eps
            positive = f()
            x.data[0, col] = original - eps
            negative = f()
            x.data[0, col] = original
            np.testing.assert_allclose(saved_input_gradient[0, col],
                                       (positive - negative) / (2 * eps), atol=1e-9)
        self.assertEqual(saved_weight_gradient[1, 0], 0.)
        self.assertNotEqual(layer.latest_effective_gradient[1, 0], 0.)

    def test_masked_input_gradient_and_counterfactual_gradient(self):
        layer = Linear(2, 2, np.random.default_rng(5))
        layer.weight.data[:] = [[2., 999.], [3., 7.]]
        layer.mask[0, 1] = False
        x = Tensor([[1., 4.]], requires_grad=True)
        output = layer(x)
        output.sum().backward()
        np.testing.assert_array_equal(layer.weight.grad, [[1., 0.], [4., 4.]])
        np.testing.assert_array_equal(x.grad, [[2., 10.]])
        # The deleted edge can have nonzero *effective* gradient: input value 1.
        self.assertEqual(layer.latest_effective_gradient[0, 1], 1.)
        self.assertEqual(layer.weight.grad[0, 1], 0.)

    def test_dead_neuron_protection_and_atomic_infeasible_budget(self):
        model = MLP(dims=(2, 4, 2), seed=7)
        pruner = GlobalPruner(model, Adam(model.parameters()),
                              min_incoming=1, min_outgoing=1)
        before = model.state_dict()
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            pruner.prune_to(0.75)  # 4 live < minimum 6 incoming
        for name in before:
            np.testing.assert_array_equal(model.state_dict()[name], before[name])
        pruner.prune_to(0.50)
        report = model.connectivity_report()
        self.assertTrue(all(r['zero_incoming'] == 0 for r in report))
        self.assertTrue(all(r['zero_outgoing'] == 0 for r in report[1:]))

    def test_counterfactual_regrowth_keeps_hard_budget_and_clears_adam(self):
        model = MLP(dims=(3, 6, 3), seed=4)
        adam = Adam(model.parameters(), lr=.02)
        pruner = GlobalPruner(model, adam, min_incoming=1, min_outgoing=1)
        pruner.prune_to(.4)
        model.zero_grad()
        x = Tensor([[1., .5, -2.], [-.3, 1., 2.]])
        cross_entropy(model(x), np.array([0, 2])).backward()
        pruner.observe_gradients()
        self.assertTrue(any(np.any(layer.regrowth_ema[~layer.mask] > 0)
                            for layer in model.linear_layers()))
        before = [layer.mask.copy() for layer in model.linear_layers()]
        before_count = model.active_weights()
        changed = pruner.regrow_swap(2)
        self.assertEqual(changed, 2)
        self.assertEqual(model.active_weights(), before_count)
        self.assertEqual(model.sparsity(), 1 - before_count / model.total_weights())
        for layer, original_mask in zip(model.linear_layers(), before):
            altered = original_mask != layer.mask
            state = adam.state_for(layer.weight)
            self.assertTrue(np.all(state.first[altered] == 0))
            self.assertTrue(np.all(state.second[altered] == 0))
            self.assertTrue(np.all(state.age[altered] == 0))
            self.assertTrue(np.all(layer.weight.data[~layer.mask] == 0))
        report = model.connectivity_report()
        self.assertTrue(all(r['zero_incoming'] == 0 for r in report))
        self.assertTrue(all(r['zero_outgoing'] == 0 for r in report[1:]))

    def test_zero_taylor_does_not_mean_zero_damage(self):
        w = Tensor(1., requires_grad=True)
        loss = (w - 1.) * (w - 1.)
        loss.backward()
        self.assertEqual(abs(float(w.data * w.grad)), 0.)
        self.assertEqual((0. - 1.) ** 2 - float(loss.data), 1.)

    def test_repeated_shared_subgraph_gradients_and_backward_reset(self):
        x = Tensor([1., 2.], requires_grad=True)
        node = x * x
        loss = (node + node).sum()
        loss.backward()
        np.testing.assert_array_equal(x.grad, [4., 8.])
        loss.backward()  # leaf gradient accumulates, intermediate does NOT double
        np.testing.assert_array_equal(x.grad, [8., 16.])

    def test_adam_bad_gradient_does_not_half_update_model(self):
        p = Tensor([1.], requires_grad=True)
        q = Tensor([2.], requires_grad=True)
        adam = Adam([(p, None), (q, None)])
        p.grad[:] = 1.
        q.grad[:] = np.inf
        with self.assertRaises(FloatingPointError):
            adam.step()
        self.assertEqual(float(p.data[0]), 1.)
        self.assertEqual(adam.state_for(p).age[0], 0)

    def test_single_edge_saliency_audit_uses_heldout_validation(self):
        from train.runner import TrainingSession
        from train.audit import measure_single_edge_damage
        session = TrainingSession(epochs=2, seed=0, target=.5)
        for _ in range(2):
            session.run_epoch()
        before = session.model.state_dict()
        rows = measure_single_edge_damage(session, n_edges=7, val_batch=32)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(np.isfinite(r['absolute_prediction_error']) for r in rows))
        for key, saved in before.items():
            np.testing.assert_array_equal(session.model.state_dict()[key], saved)

    def test_extreme_logits_stay_finite(self):
        logits = Tensor([[10000., -10000.], [-1e4, 1e4]], requires_grad=True)
        loss = cross_entropy(logits, np.array([0, 1]))
        self.assertTrue(np.isfinite(float(loss.data)))
        loss.backward()
        self.assertTrue(np.all(np.isfinite(logits.grad)))


if __name__ == '__main__':
    unittest.main()
