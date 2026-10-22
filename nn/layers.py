"""Composable neural-network components, with genuine binary connection masks."""
from __future__ import annotations

import numpy as np

from engine import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int,
                 rng: np.random.Generator, *, output_layer: bool = False):
        if in_features <= 0 or out_features <= 0:
            raise ValueError('Layer dimensions must be positive')
        # He initialization for layers followed by ReLU; smaller Xavier-like
        # variance for final logits (no ReLU follows the output layer).
        scale = np.sqrt((1.0 if output_layer else 2.0) / in_features)
        self.weight = Tensor(rng.normal(0, scale, (in_features, out_features)),
                             requires_grad=True, name='weight')
        self.bias = Tensor(np.zeros(out_features), requires_grad=True, name='bias')
        self.mask = np.ones(self.weight.data.shape, dtype=bool)
        self.saliency_ema = np.zeros_like(self.weight.data)
        self.regrowth_ema = np.zeros_like(self.weight.data)
        self.latest_effective_gradient = np.zeros_like(self.weight.data)

    def __call__(self, x: Tensor) -> Tensor:
        # A fresh constant-mask Tensor snapshots topology for this forward graph.
        # dL/d(weight) = mask * dL/d(effective_weight); masked grad is zero.
        if self.weight._pending_graphs:
            raise RuntimeError('A differentiable forward pass is awaiting backward')
        saved_mask = self.mask.copy()
        saved_weight = self.weight.data.copy()
        saved_bias = self.bias.data.copy()
        effective_weight = self.weight * Tensor(saved_mask.astype(np.float64))
        result = x @ effective_weight + self.bias
        self.weight._pending_graphs += 1

        def verify_unchanged():
            if (not np.array_equal(self.mask, saved_mask)
                    or not np.array_equal(self.weight.data, saved_weight)
                    or not np.array_equal(self.bias.data, saved_bias)):
                raise RuntimeError('Mask/parameter changed between forward and backward')

        def complete():
            # dL/d(W_eff) is distinct from dL/dW = mask * dL/d(W_eff).
            # Dormant edges can have a nonzero counterfactual gradient even
            # when their stored parameter gradient is exactly zero.
            self.latest_effective_gradient[:] = effective_weight.grad
            self.weight._pending_graphs -= 1

        result._backward_guards.append(verify_unchanged)
        result._backward_finished.append(complete)
        return result

    def weight_macs(self) -> int:
        return int(np.count_nonzero(self.mask))

    def dense_logits(self, x: np.ndarray) -> np.ndarray:
        return x @ (self.weight.data * self.mask) + self.bias.data

    def parameters(self) -> list[tuple[Tensor, np.ndarray | None]]:
        return [(self.weight, self.mask), (self.bias, None)]


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()


class MLP:
    def __init__(self, dims: tuple[int, ...] = (64, 96, 48, 10), seed: int = 0):
        if len(dims) < 2:
            raise ValueError('MLP requires input and output dimensions')
        rng = np.random.default_rng(seed)
        self.layers: list[Linear | ReLU] = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1], rng,
                                      output_layer=(i == len(dims) - 2)))
            if i != len(dims) - 2:
                self.layers.append(ReLU())

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def linear_layers(self) -> list[Linear]:
        return [layer for layer in self.layers if isinstance(layer, Linear)]

    def parameters(self) -> list[tuple[Tensor, np.ndarray | None]]:
        result: list[tuple[Tensor, np.ndarray | None]] = []
        for layer in self.linear_layers():
            result.extend(layer.parameters())
        return result

    def zero_grad(self) -> None:
        for tensor, _ in self.parameters():
            tensor.zero_grad()

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Dense masked evaluation, for accuracy and correctness reference only."""
        for layer in self.layers:
            if isinstance(layer, Linear):
                x = layer.dense_logits(x)
            else:
                x = np.maximum(x, 0.0)
        return x

    def assert_idle(self) -> None:
        if any(layer.weight._pending_graphs for layer in self.linear_layers()):
            raise RuntimeError('Cannot change topology/optimizer while forward graph is in flight')

    def connectivity_report(self) -> list[dict]:
        """Layer-wise active edges and neurons with no incoming/outgoing edges."""
        layers = self.linear_layers()
        records = []
        for idx, layer in enumerate(layers):
            incoming = np.count_nonzero(layer.mask, axis=0)
            outgoing = np.count_nonzero(layer.mask, axis=1)
            records.append(dict(layer=idx, active=int(layer.mask.sum()),
                                total=int(layer.mask.size),
                                zero_incoming=int(np.count_nonzero(incoming == 0)),
                                zero_outgoing=int(np.count_nonzero(outgoing == 0))
                                if idx > 0 else None))
        return records

    def total_weights(self) -> int:
        return sum(layer.mask.size for layer in self.linear_layers())

    def active_weights(self) -> int:
        return sum(layer.weight_macs() for layer in self.linear_layers())

    def sparsity(self) -> float:
        return 1.0 - self.active_weights() / self.total_weights()

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {}
        for idx, layer in enumerate(self.linear_layers()):
            state[f'weight_{idx}'] = layer.weight.data.copy()
            state[f'bias_{idx}'] = layer.bias.data.copy()
            state[f'mask_{idx}'] = layer.mask.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.assert_idle()
        layers = self.linear_layers()
        # Validate ALL layers first so malformed checkpoints cannot partially
        # replace a running model's parameters.
        for idx, layer in enumerate(layers):
            for name, dest in [('weight', layer.weight.data),
                               ('bias', layer.bias.data), ('mask', layer.mask)]:
                source = np.asarray(state[f'{name}_{idx}'])
                if source.shape != dest.shape:
                    raise ValueError(f'State shape mismatch for {name}_{idx}')
                if name == 'mask' and source.dtype != np.dtype(bool):
                    raise ValueError('Masks must be strictly boolean')
                if name != 'mask' and not np.all(np.isfinite(source)):
                    raise ValueError('State parameters must be finite')
            if np.any(np.asarray(state[f'weight_{idx}'])[
                      ~np.asarray(state[f'mask_{idx}'])] != 0):
                raise ValueError('Dormant checkpoint weights must be exactly zero')
        for idx, layer in enumerate(layers):
            layer.weight.data[:] = state[f'weight_{idx}']
            layer.bias.data[:] = state[f'bias_{idx}']
            layer.mask[:] = state[f'mask_{idx}']
