"""Exact-budget pruning, connectivity guards, and gradient-based edge regrowth."""
from __future__ import annotations

import numpy as np
from nn import MLP, Linear
from optim import Adam


class GlobalPruner:
    def __init__(self, model: MLP, optimizer: Adam, criterion: str = 'taylor',
                 smoothing: float = 0.90, *, min_incoming: int = 1,
                 min_outgoing: int = 1):
        if criterion not in ('taylor', 'magnitude'):
            raise ValueError('criterion must be taylor or magnitude')
        if not 0 <= smoothing < 1:
            raise ValueError('smoothing must be in [0, 1)')
        if min_incoming < 0 or min_outgoing < 0:
            raise ValueError('Minimum degrees must be nonnegative')
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.smoothing = smoothing
        self.min_incoming = int(min_incoming)
        # Incoming connectivity of each hidden neuron is governed by its own
        # layer; outgoing connectivity of hidden neurons in later layers.
        self.min_outgoing = int(min_outgoing)

    def observe_gradients(self) -> None:
        """Record active Taylor |w*dL/dw| and dormant effective-edge gradients.

        Called after backward and before the optimizer step. Stored-weight
        gradients are exactly zero on dormant edges; regrowth scores come
        separately from the gradient of the *effective* pre-mask weight.
        """
        for layer in self.model.linear_layers():
            if layer.weight.grad is None:
                continue
            active = layer.mask
            inactive = ~active
            if self.criterion == 'taylor':
                score = np.abs(layer.weight.data * layer.weight.grad)
                layer.saliency_ema[active] = (self.smoothing * layer.saliency_ema[active]
                                               + (1 - self.smoothing) * score[active])
            layer.saliency_ema[inactive] = 0.0
            edge_score = np.abs(layer.latest_effective_gradient)
            layer.regrowth_ema[inactive] = (self.smoothing * layer.regrowth_ema[inactive]
                                           + (1 - self.smoothing) * edge_score[inactive])
            layer.regrowth_ema[active] = 0.0

    def _importance(self, layer: Linear) -> np.ndarray:
        return (np.abs(layer.weight.data) if self.criterion == 'magnitude'
                else layer.saliency_ema)

    def _layout(self):
        starts = []
        offset = 0
        for layer in self.model.linear_layers():
            starts.append(offset)
            offset += layer.mask.size
        return starts

    def _locate(self, flat_id: int, starts: list[int]):
        layers = self.model.linear_layers()
        for idx, (layer, start) in enumerate(zip(layers, starts)):
            if start <= flat_id < start + layer.mask.size:
                local = flat_id - start
                row, col = np.unravel_index(local, layer.mask.shape)
                return idx, int(row), int(col)
        raise ValueError('Unknown edge index')

    def _candidate_prunes(self, count: int, *, disallowed: set[int] | None = None):
        """Select removable active coordinates with exact-k stable tie breaks.

        Degree limits protect all outputs and all hidden-neuron outgoing paths.
        The first layer's raw input features do not need an outgoing edge.
        Selection is prepared without mutating masks, so infeasible requests
        fail atomically, rather than leaving half a model pruned.
        """
        starts = self._layout()
        layers = self.model.linear_layers()
        ids, scores = [], []
        for layer, start in zip(layers, starts):
            local = np.flatnonzero(layer.mask.ravel())
            ids.extend((local + start).tolist())
            scores.extend(self._importance(layer).ravel()[local].tolist())
        ids = np.asarray(ids, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        ranked = np.lexsort((ids, scores))
        # Resolve id -> layer/row/column once; avoids per-edge unravel calls
        # during the potentially large cubic pruning ramp.
        layer_idx = np.searchsorted(np.asarray(starts), ids, side='right') - 1
        local = ids - np.asarray(starts)[layer_idx]
        widths = np.asarray([layer.mask.shape[1] for layer in layers])
        rows = local // widths[layer_idx]
        cols = local % widths[layer_idx]
        incoming = [np.count_nonzero(layer.mask, axis=0).astype(int)
                    for layer in layers]
        outgoing = [np.count_nonzero(layer.mask, axis=1).astype(int)
                    for layer in layers]
        selected = []
        for rank in ranked:
            edge = int(ids[rank])
            if disallowed and edge in disallowed:
                continue
            li, row, col = int(layer_idx[rank]), int(rows[rank]), int(cols[rank])
            if incoming[li][col] <= self.min_incoming:
                continue
            if li > 0 and outgoing[li][row] <= self.min_outgoing:
                continue
            incoming[li][col] -= 1
            outgoing[li][row] -= 1
            selected.append(edge)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f'Exact-k budget infeasible under degree constraints: '
                             f'requested {count}, only {len(selected)} removable')
        return selected

    def _drop(self, selected: list[int]) -> None:
        starts = self._layout()
        for li, layer in enumerate(self.model.linear_layers()):
            local = [v - starts[li] for v in selected
                     if starts[li] <= v < starts[li] + layer.mask.size]
            if not local:
                continue
            where = np.zeros(layer.mask.size, dtype=bool)
            where[local] = True
            where = where.reshape(layer.mask.shape)
            layer.mask[where] = False
            layer.saliency_ema[where] = 0.0
            layer.regrowth_ema[where] = 0.0
            self.optimizer.reset_coordinates(layer.weight, where)

    def _grow(self, selected: list[int]) -> None:
        starts = self._layout()
        for li, layer in enumerate(self.model.linear_layers()):
            local = [v - starts[li] for v in selected
                     if starts[li] <= v < starts[li] + layer.mask.size]
            if not local:
                continue
            where = np.zeros(layer.mask.size, dtype=bool)
            where[local] = True
            where = where.reshape(layer.mask.shape)
            if np.any(layer.mask[where]):
                raise ValueError('Can only revive dormant connections')
            self.optimizer.reset_coordinates(layer.weight, where)
            layer.saliency_ema[where] = 0.0
            layer.regrowth_ema[where] = 0.0
            layer.mask[where] = True

    def prune_to(self, sparsity: float) -> int:
        self.model.assert_idle()
        if not 0 <= sparsity < 1:
            raise ValueError('Sparsity must be between 0 and 1 (exclusive)')
        total = self.model.total_weights()
        desired = int(round(sparsity * total))
        existing = total - self.model.active_weights()
        to_remove = desired - existing
        if to_remove <= 0:
            return 0
        selected = self._candidate_prunes(to_remove)
        self._drop(selected)
        assert total - self.model.active_weights() == desired
        return to_remove

    def regrow_swap(self, count: int) -> int:
        """Exchange dormant high-gradient edges for low-importance live edges.

        Requires observed counterfactual effective-weight gradients. Does not
        alter the number of active connections or bypass minimum degree rules.
        This is a simple optional RigL-inspired exchange, not full RigL.
        """
        self.model.assert_idle()
        if count < 0:
            raise ValueError('Swap count must be >= 0')
        if count == 0:
            return 0
        starts = self._layout()
        ids, scores = [], []
        for layer, start in zip(self.model.linear_layers(), starts):
            local = np.flatnonzero(~layer.mask.ravel())
            values = layer.regrowth_ema.ravel()[local]
            eligible = values > 0  # no evidence => no speculative regrowth
            ids.extend((local[eligible] + start).tolist())
            scores.extend(values[eligible].tolist())
        if not ids:
            return 0
        order = np.lexsort((np.asarray(ids), -np.asarray(scores)))
        selected_grow = [ids[j] for j in order[:count]]
        selected_prune = self._candidate_prunes(len(selected_grow))
        before = self.model.active_weights()
        self._drop(selected_prune)
        self._grow(selected_grow)
        assert self.model.active_weights() == before
        return len(selected_grow)

    def revive(self, layer: Linear, coordinates: tuple[int, int]) -> None:
        """Manual revival may change the budget: use regrow_swap for a swap."""
        self.model.assert_idle()
        if layer not in self.model.linear_layers():
            raise ValueError('Layer not in model')
        row, col = coordinates
        if not (0 <= row < layer.mask.shape[0] and 0 <= col < layer.mask.shape[1]):
            raise IndexError('Invalid connection coordinates')
        if layer.mask[row, col]:
            raise ValueError('Connection is already active')
        where = np.zeros_like(layer.mask)
        where[row, col] = True
        self.optimizer.reset_coordinates(layer.weight, where)
        layer.saliency_ema[where] = 0.0
        layer.regrowth_ema[where] = 0.0
        layer.mask[where] = True

    def state_dict(self) -> dict[str, np.ndarray]:
        result = {}
        for idx, layer in enumerate(self.model.linear_layers()):
            result[f'saliency_{idx}'] = layer.saliency_ema.copy()
            result[f'regrowth_{idx}'] = layer.regrowth_ema.copy()
        return result

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.model.assert_idle()
        for idx, layer in enumerate(self.model.linear_layers()):
            for name, dest in [('saliency', layer.saliency_ema),
                               ('regrowth', layer.regrowth_ema)]:
                values = np.asarray(state[f'{name}_{idx}'])
                if values.shape != dest.shape or not np.all(np.isfinite(values)):
                    raise ValueError('Invalid pruning checkpoint state')
                if np.any(values < 0):
                    raise ValueError('Pruning scores must be nonnegative')
                dest[:] = values
