"""Inference-only, genuinely sparse edge traversal. No dense-times-zero matmul."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from nn import MLP


@dataclass(frozen=True)
class SparseLinear:
    # CSC-style per-output-neuron active input indices.
    indices: np.ndarray
    indptr: np.ndarray
    values: np.ndarray
    bias: np.ndarray
    in_features: int
    out_features: int

    @classmethod
    def from_layer(cls, layer) -> 'SparseLinear':
        mask = layer.mask
        indices, values, starts = [], [], [0]
        for col in range(mask.shape[1]):
            rows = np.flatnonzero(mask[:, col])
            indices.extend(rows.tolist())
            values.extend(layer.weight.data[rows, col].tolist())
            starts.append(len(indices))
        return cls(np.asarray(indices, dtype=np.int32),
                   np.asarray(starts, dtype=np.int32),
                   np.asarray(values, dtype=np.float64),
                   layer.bias.data.copy(), mask.shape[0], mask.shape[1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError('Invalid sparse Linear input dimensions')
        y = np.broadcast_to(self.bias, (len(x), self.out_features)).copy()
        for col in range(self.out_features):
            first, last = int(self.indptr[col]), int(self.indptr[col + 1])
            if first != last:
                # Operates only on live entries, unlike x @ (W * mask).
                y[:, col] += x[:, self.indices[first:last]] @ self.values[first:last]
        return y

    @property
    def nnz(self) -> int:
        return int(self.values.size)

    def validate(self) -> None:
        if self.indptr.shape != (self.out_features + 1,):
            raise ValueError('Invalid indptr shape')
        if self.indptr[0] != 0 or self.indptr[-1] != self.nnz:
            raise ValueError('Invalid sparse offset limits')
        if np.any(np.diff(self.indptr) < 0):
            raise ValueError('Non-monotonic indptr')
        if self.indices.shape != self.values.shape:
            raise ValueError('Invalid indices/values shapes')
        if np.any(self.indices < 0) or np.any(self.indices >= self.in_features):
            raise ValueError('Sparse indices out of bounds')
        if self.bias.shape != (self.out_features,):
            raise ValueError('Invalid bias shape')
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError('Invalid sparse layer dimensions')
        for col in range(self.out_features):
            entries = self.indices[self.indptr[col]:self.indptr[col + 1]]
            if np.any(np.diff(entries) <= 0):
                raise ValueError('Sparse row indices must be unique and sorted')
        if not np.all(np.isfinite(self.values)) or not np.all(np.isfinite(self.bias)):
            raise ValueError('Non-finite checkpoint')


class SparseMLP:
    def __init__(self, layers: list[SparseLinear]):
        self.layers = layers
        for layer in layers:
            layer.validate()
        if not layers:
            raise ValueError('Sparse MLP requires at least one layer')
        for previous, current in zip(layers, layers[1:]):
            if previous.out_features != current.in_features:
                raise ValueError('Incompatible adjacent sparse layers')

    @classmethod
    def from_model(cls, model: MLP) -> 'SparseMLP':
        return cls([SparseLinear.from_layer(layer) for layer in model.linear_layers()])

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    def nnz(self) -> int:
        return sum(layer.nnz for layer in self.layers)

    def save(self, path: str) -> None:
        arrays = {'n_layers': np.array(len(self.layers), dtype=np.int64)}
        for idx, layer in enumerate(self.layers):
            for key in ('indices', 'indptr', 'values', 'bias', 'in_features', 'out_features'):
                arrays[f'{idx}_{key}'] = np.asarray(getattr(layer, key))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> 'SparseMLP':
        with np.load(path, allow_pickle=False) as data:
            count = int(data['n_layers'])
            if not 1 <= count <= 50:
                raise ValueError('Unexpected layer count')
            layers = []
            for idx in range(count):
                layer = SparseLinear(
                    data[f'{idx}_indices'].copy(), data[f'{idx}_indptr'].copy(),
                    data[f'{idx}_values'].copy(), data[f'{idx}_bias'].copy(),
                    int(data[f'{idx}_in_features']), int(data[f'{idx}_out_features']))
                layers.append(layer)
        return cls(layers)
