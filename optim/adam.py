"""From-scratch Adam with topology-aware, per-coordinate bias correction."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from engine import Tensor


@dataclass
class AdamState:
    first: np.ndarray
    second: np.ndarray
    age: np.ndarray


class Adam:
    def __init__(self, params: list[tuple[Tensor, np.ndarray | None]],
                 lr: float = 0.003, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        if lr <= 0 or epsilon <= 0 or not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError('Invalid Adam hyperparameters')
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state: dict[int, AdamState] = {}
        for param, mask in params:
            if not param.requires_grad:
                raise ValueError('Adam expects differentiable parameters')
            if mask is not None and mask.shape != param.data.shape:
                raise ValueError('Mask shape mismatch')
            self.state[id(param)] = AdamState(
                np.zeros_like(param.data), np.zeros_like(param.data),
                np.zeros(param.data.shape, dtype=np.int64))

    def state_for(self, param: Tensor) -> AdamState:
        return self.state[id(param)]

    def reset_coordinates(self, param: Tensor, where: np.ndarray) -> None:
        """Must be used on both prune and revive: no historical momentum."""
        if where.shape != param.data.shape:
            raise ValueError('Coordinate mask shape mismatch')
        if param._pending_graphs:
            raise RuntimeError('Cannot reset optimizer during pending backward')
        state = self.state_for(param)
        param.data[where] = 0.0
        state.first[where] = 0.0
        state.second[where] = 0.0
        state.age[where] = 0
        if param.grad is not None:
            param.grad[where] = 0.0

    def step(self) -> None:
        # Preflight every tensor before touching ANY parameter. Otherwise a bad
        # gradient in the last layer could leave earlier layers half updated.
        for param, mask in self.params:
            if param._pending_graphs:
                raise RuntimeError('Cannot step optimizer before backward completes')
            if param.grad is None:
                continue
            if mask is not None:
                if mask.shape != param.data.shape or mask.dtype != np.dtype(bool):
                    raise ValueError('Invalid active mask')
                if not np.all(param.grad[~mask] == 0):
                    raise AssertionError('Masked parameter has nonzero gradient')
            if not np.all(np.isfinite(param.grad)):
                raise FloatingPointError('Non-finite gradient')
        for param, mask in self.params:
            if param.grad is None:
                continue
            active = np.ones(param.data.shape, dtype=bool) if mask is None else mask
            state = self.state_for(param)
            if mask is not None:
                self.reset_coordinates(param, ~mask)
            g = param.grad[active]
            state.age[active] += 1
            state.first[active] = self.beta1 * state.first[active] + (1 - self.beta1) * g
            state.second[active] = self.beta2 * state.second[active] + (1 - self.beta2) * g * g
            ages = state.age[active]
            m = state.first[active] / (1 - self.beta1 ** ages)
            v = state.second[active] / (1 - self.beta2 ** ages)
            param.data[active] -= self.lr * m / (np.sqrt(v) + self.epsilon)
            if mask is not None:
                param.data[~mask] = 0.0
            if not np.all(np.isfinite(param.data)):
                raise FloatingPointError('Non-finite parameter')

    def state_dict(self) -> dict[str, np.ndarray]:
        """No id-based keys in persistence; follow stable parameter order."""
        result = {}
        for idx, (param, _) in enumerate(self.params):
            state = self.state_for(param)
            for name in ('first', 'second', 'age'):
                result[f'adam_{idx}_{name}'] = getattr(state, name).copy()
        return result

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for idx, (param, _) in enumerate(self.params):
            if param._pending_graphs:
                raise RuntimeError('Cannot load optimizer state during pending backward')
            record = self.state_for(param)
            for name in ('first', 'second', 'age'):
                dest = getattr(record, name)
                source = np.asarray(state[f'adam_{idx}_{name}'])
                if source.shape != dest.shape or source.dtype.kind not in 'fiu':
                    raise ValueError(f'Invalid Adam state {idx}/{name}')
                if name == 'age' and (np.any(source < 0)
                                      or not np.issubdtype(source.dtype, np.integer)):
                    raise ValueError('Invalid Adam step ages')
                if name != 'age' and not np.all(np.isfinite(source)):
                    raise ValueError('Nonfinite optimizer state')
                dest[:] = source
        for param, mask in self.params:
            if mask is None:
                continue
            state_obj = self.state_for(param)
            if (np.any(param.data[~mask] != 0)
                    or np.any(state_obj.first[~mask] != 0)
                    or np.any(state_obj.second[~mask] != 0)
                    or np.any(state_obj.age[~mask] != 0)):
                raise ValueError('Nonzero parameters or optimizer history on dormant edges')
