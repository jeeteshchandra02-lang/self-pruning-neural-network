"""Small, auditable, float64 reverse-mode differentiation engine.

Supported matmul is intentionally 2-D only (the task uses matrix-shaped batches).
Graphs are built afresh for each forward pass. No parameter or mask is mutated
between a forward pass and its corresponding backward pass.
"""
from __future__ import annotations

import numpy as np


def sum_to_shape(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Undo NumPy broadcasting by accumulating over replicated coordinates."""
    result = np.asarray(gradient, dtype=np.float64)
    while result.ndim > len(shape):
        result = result.sum(axis=0)
    for axis, dimension in enumerate(shape):
        if dimension == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    if result.shape != shape:
        raise ValueError(f"cannot reduce gradient {gradient.shape} to {shape}")
    return result


def _tensor(value: object) -> 'Tensor':
    return value if isinstance(value, Tensor) else Tensor(value, requires_grad=False)


class Tensor:
    def __init__(self, data: object, requires_grad: bool = False,
                 parents: tuple['Tensor', ...] = (), name: str = ''):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.requires_grad = bool(requires_grad)
        self.parents = tuple(parents)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._backward = lambda: None
        # Graph-lifecycle hooks: validate saved parameter/mask versions BEFORE
        # any derivative is computed, and release forward leases AFTER backward.
        self._backward_guards = []
        self._backward_finished = []
        self._pending_graphs = 0  # used by Linear parameter leaves
        self.name = name

    @staticmethod
    def _result(data: object, parents: tuple['Tensor', ...]) -> 'Tensor':
        return Tensor(data, requires_grad=any(p.requires_grad for p in parents),
                      parents=parents)

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def backward(self, gradient: object | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError('Cannot backpropagate a tensor without gradients')
        if gradient is None:
            if self.data.size != 1:
                raise ValueError('Non-scalar backward requires an explicit upstream gradient')
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(gradient, dtype=np.float64)
            if seed.shape != self.data.shape:
                raise ValueError('Upstream gradient shape mismatch')

        topo: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            topo.append(node)

        visit(self)
        # Fail closed before changing any gradients if an in-flight model graph
        # saw a topology or parameter edit since its forward pass.
        for node in topo:
            for check in node._backward_guards:
                check()
        # Clear only intermediate gradients: leaf gradients are deliberately
        # accumulated until the training loop calls zero_grad().
        for node in topo:
            if node.parents and node.grad is not None:
                node.grad.fill(0.0)
        self.grad += seed
        for node in reversed(topo):
            node._backward()
        # Store counterfactual effective-edge gradients and release leases only
        # after every gradient of the graph is fully accumulated.
        for node in topo:
            completed, node._backward_finished = node._backward_finished, []
            for callback in completed:
                callback()

    def __add__(self, other: object) -> 'Tensor':
        other = _tensor(other)
        out = self._result(self.data + other.data, (self, other))

        def backward() -> None:
            if self.requires_grad:
                self.grad += sum_to_shape(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += sum_to_shape(out.grad, other.data.shape)
        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self) -> 'Tensor':
        return self * -1.0

    def __sub__(self, other: object) -> 'Tensor':
        return self + (-_tensor(other))

    def __rsub__(self, other: object) -> 'Tensor':
        return _tensor(other) - self

    def __mul__(self, other: object) -> 'Tensor':
        other = _tensor(other)
        a, b = self.data.copy(), other.data.copy()  # forward-time snapshots
        out = self._result(a * b, (self, other))

        def backward() -> None:
            if self.requires_grad:
                self.grad += sum_to_shape(out.grad * b, self.data.shape)
            if other.requires_grad:
                other.grad += sum_to_shape(out.grad * a, other.data.shape)
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> 'Tensor':
        other = _tensor(other)
        a, b = self.data.copy(), other.data.copy()
        out = self._result(a / b, (self, other))

        def backward() -> None:
            if self.requires_grad:
                self.grad += sum_to_shape(out.grad / b, self.data.shape)
            if other.requires_grad:
                other.grad += sum_to_shape(-out.grad * a / (b * b), other.data.shape)
        out._backward = backward
        return out

    def __rtruediv__(self, other: object) -> 'Tensor':
        return _tensor(other) / self

    def __matmul__(self, other: object) -> 'Tensor':
        other = _tensor(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError('Only 2-D matrix multiplication is supported')
        a, b = self.data.copy(), other.data.copy()
        out = self._result(a @ b, (self, other))

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad @ b.T
            if other.requires_grad:
                other.grad += a.T @ out.grad
        out._backward = backward
        return out

    @staticmethod
    def _axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
        if axis is None:
            return tuple(range(ndim))
        raw = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(a + ndim if a < 0 else a for a in raw)
        if len(set(axes)) != len(axes) or any(a < 0 or a >= ndim for a in axes):
            raise ValueError('Invalid reduction axes')
        return tuple(sorted(axes))

    def sum(self, axis: int | tuple[int, ...] | None = None,
            keepdims: bool = False) -> 'Tensor':
        axes = self._axes(axis, self.data.ndim)
        out = self._result(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def backward() -> None:
            if self.requires_grad:
                g = out.grad
                if not keepdims:
                    for ax in axes:
                        g = np.expand_dims(g, ax)
                self.grad += np.broadcast_to(g, self.data.shape)
        out._backward = backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None,
             keepdims: bool = False) -> 'Tensor':
        axes = self._axes(axis, self.data.ndim)
        count = int(np.prod([self.data.shape[a] for a in axes]))
        if count == 0:
            raise ValueError('Cannot mean-reduce an empty axis')
        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self) -> 'Tensor':
        x = self.data.copy()
        out = self._result(np.maximum(x, 0.0), (self,))

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (x > 0)
        out._backward = backward
        return out

    def tanh(self) -> 'Tensor':
        y = np.tanh(self.data)
        out = self._result(y, (self,))

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (1.0 - y * y)
        out._backward = backward
        return out

    def exp(self) -> 'Tensor':
        y = np.exp(self.data)
        out = self._result(y, (self,))

        def backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * y
        out._backward = backward
        return out

    def __repr__(self) -> str:
        return f'Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})'


def cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Stable combined log-sum-exp cross entropy; expects 2-D logits, 1-D labels."""
    z = logits.data
    labels = np.asarray(labels)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
        raise ValueError('Expected nonempty logits of shape (batch, classes>=2)')
    if labels.ndim != 1 or labels.shape[0] != z.shape[0]:
        raise ValueError('Expected one label per row')
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError('Class labels must be integers')
    if np.any((labels < 0) | (labels >= z.shape[1])):
        raise ValueError('Class label out of bounds')
    if not np.all(np.isfinite(z)):
        raise ValueError('Logits must be finite')
    shifted = z - np.max(z, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    total = exponentials.sum(axis=1, keepdims=True)
    prob = exponentials / total
    loss = np.mean(np.log(total[:, 0]) - shifted[np.arange(len(labels)), labels])
    out = Tensor(loss, requires_grad=logits.requires_grad, parents=(logits,))

    def backward() -> None:
        if logits.requires_grad:
            delta = prob.copy()
            delta[np.arange(len(labels)), labels] -= 1.0
            logits.grad += out.grad * delta / len(labels)
    out._backward = backward
    return out
