"""Central finite differences against EVERY implemented differentiable op."""
import unittest
import numpy as np

from engine import Tensor, cross_entropy


def verify_gradients(fn, inputs, tolerance=2e-5, epsilon=1e-5):
    inputs = [np.array(x, dtype=np.float64) for x in inputs]
    tensors = [Tensor(x, requires_grad=True) for x in inputs]
    result = fn(*tensors)
    if result.data.size != 1:
        result = result.sum()
    result.backward()
    analytic = [t.grad.copy() for t in tensors]
    for arg in range(len(inputs)):
        approximate = np.zeros_like(inputs[arg])
        for position in np.ndindex(inputs[arg].shape):
            positive = [x.copy() for x in inputs]
            negative = [x.copy() for x in inputs]
            positive[arg][position] += epsilon
            negative[arg][position] -= epsilon
            left = fn(*[Tensor(x) for x in positive])
            right = fn(*[Tensor(x) for x in negative])
            approximate[position] = (left.data.sum() - right.data.sum()) / (2 * epsilon)
        np.testing.assert_allclose(analytic[arg], approximate,
                                   atol=tolerance, rtol=tolerance)


class GradientCheckTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(117)

    def test_add_broadcast(self):
        verify_gradients(lambda a, b: a + b,
                         [self.rng.normal(size=(3, 4)), self.rng.normal(size=(4,))])
        verify_gradients(lambda a, b: a + b,
                         [self.rng.normal(size=(3, 4)), self.rng.normal(size=(1, 4))])
        verify_gradients(lambda a, b: a + b,
                         [self.rng.normal(size=(2, 1)), self.rng.normal(size=(1, 4))])
        verify_gradients(lambda a, b: a + b,
                         [self.rng.normal(size=(2, 3)), np.array(0.5)])

    def test_sub_and_reverse_sub(self):
        verify_gradients(lambda a, b: a - b,
                         [self.rng.normal(size=(3, 4)), self.rng.normal(size=(4,))])
        verify_gradients(lambda a: 3.0 - a, [self.rng.normal(size=(2, 3))])
        verify_gradients(lambda a: -a, [self.rng.normal(size=(2, 3))])

    def test_multiply_and_reverse_multiply(self):
        verify_gradients(lambda a, b: a * b,
                         [self.rng.normal(size=(3, 1)), self.rng.normal(size=(1, 4))])
        verify_gradients(lambda a: 2.5 * a, [self.rng.normal(size=(2, 3))])

    def test_divide_and_reverse_divide(self):
        verify_gradients(lambda a, b: a / b,
                         [self.rng.normal(size=(3, 4)),
                          self.rng.uniform(0.5, 2.0, size=(4,))])
        verify_gradients(lambda a: 2.0 / a, [self.rng.uniform(0.4, 2.0, size=(2, 3))])

    def test_matrix_multiply(self):
        verify_gradients(lambda a, b: a @ b,
                         [self.rng.normal(size=(3, 5)),
                          self.rng.normal(size=(5, 4))])

    def test_sum_reduction(self):
        x = self.rng.normal(size=(2, 3, 4))
        verify_gradients(lambda a: a.sum(), [x])
        verify_gradients(lambda a: a.sum(axis=1), [x])
        verify_gradients(lambda a: a.sum(axis=(0, 2)), [x])
        verify_gradients(lambda a: a.sum(axis=-1, keepdims=True), [x])

    def test_mean_reduction(self):
        x = self.rng.normal(size=(2, 3, 4))
        verify_gradients(lambda a: a.mean(), [x])
        verify_gradients(lambda a: a.mean(axis=0), [x])
        verify_gradients(lambda a: a.mean(axis=(0, 2), keepdims=True), [x])

    def test_relu_away_from_nondifferentiable_zero(self):
        verify_gradients(lambda a: a.relu(),
                         [np.array([[-3.0, -1.0], [1.5, 4.0]])])

    def test_tanh(self):
        verify_gradients(lambda a: a.tanh(), [self.rng.normal(size=(3, 4))])

    def test_exp(self):
        verify_gradients(lambda a: a.exp(), [self.rng.normal(size=(3, 4))])

    def test_combined_softmax_cross_entropy(self):
        targets = np.array([0, 3, 1], dtype=np.int64)
        verify_gradients(lambda a: cross_entropy(a, targets),
                         [self.rng.normal(size=(3, 4))])

    def test_graph_accumulation_diamond(self):
        x = Tensor(np.array([1.3, -2.1, 0.5]), requires_grad=True)
        (x * x + x).sum().backward()
        np.testing.assert_allclose(x.grad, 2 * x.data + 1)
        # A second backward, after explicitly resetting the leaf, must give
        # the SAME result (not accumulated stale intermediates).
        x.zero_grad()
        (x * x + x).sum().backward()
        np.testing.assert_allclose(x.grad, 2 * x.data + 1)
        y = (x * x + x).sum()
        x.zero_grad()
        y.backward()
        y.backward()  # leaf accumulates from reused graph; intermediates reset
        np.testing.assert_allclose(x.grad, 2 * (2 * x.data + 1))

    def test_stable_large_logits(self):
        x = Tensor(np.array([[1000., -1000.], [-1000., 1000.]]), requires_grad=True)
        loss = cross_entropy(x, np.array([0, 1]))
        self.assertTrue(np.isfinite(loss.data).all())
        loss.backward()
        self.assertTrue(np.isfinite(x.grad).all())
        self.assertLess(float(loss.data), 1e-10)

    def test_bad_shapes_and_labels_rejected(self):
        with self.assertRaises(ValueError):
            (Tensor([1., 2.], True) @ Tensor([1., 2.])).sum()
        with self.assertRaises(ValueError):
            cross_entropy(Tensor(np.ones((2, 3)), True), np.array([-1, 0]))
        with self.assertRaises(TypeError):
            cross_entropy(Tensor(np.ones((2, 3)), True), np.array([1.0, 0.0]))
        with self.assertRaises(ValueError):
            Tensor(np.ones((2, 2)), True).backward()


if __name__ == '__main__':
    unittest.main()
