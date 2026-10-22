"""Hand-written mini-batch training, evaluation, and reproducible timing."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter_ns
import platform
import hashlib

import numpy as np

from data import load_data
from engine import Tensor, cross_entropy
from nn import MLP
from optim import Adam
from prune import GlobalPruner, SparseMLP, target_for_epoch


ROOT = Path(__file__).resolve().parents[1]


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(model.predict_logits(x), axis=1) == y))


def batch_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    # Independent NumPy check / monitor; training itself uses our engine.
    shifted = logits - logits.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=1))
                         - shifted[np.arange(len(labels)), labels]))


def data_fingerprint(dataset) -> str:
    """Ensure resumed training uses exactly the same data and split."""
    digest = hashlib.sha256()
    for split in dataset:
        for array in split:
            a = np.ascontiguousarray(array)
            digest.update(str((a.dtype.str, a.shape)).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


class TrainingSession:
    """Epoch-boundary resumable training, with all RNG and optimizer state.

    The sparsity schedule depends on the TOTAL planned number of epochs;
    changing it while resuming is deliberately rejected.
    """
    def __init__(self, *, seed: int = 0, epochs: int = 32, batch_size: int = 64,
                 lr: float = 0.003, target: float = 0.0, criterion: str = 'taylor',
                 dataset=None, regrow_swaps: int = 0,
                 min_incoming: int = 1, min_outgoing: int = 1):
        if epochs <= 0 or batch_size <= 0 or regrow_swaps < 0:
            raise ValueError('Invalid epoch, batch, or regrowth count')
        if not 0 <= target < 1:
            raise ValueError('Invalid target sparsity')
        self.dataset = load_data() if dataset is None else dataset
        self.model = MLP(seed=seed)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.pruner = GlobalPruner(self.model, self.optimizer, criterion=criterion,
                                   min_incoming=min_incoming,
                                   min_outgoing=min_outgoing)
        self.rng = np.random.default_rng(seed + 100_000)
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.target = target
        self.criterion = criterion
        self.regrow_swaps = regrow_swaps
        self.min_incoming = min_incoming
        self.min_outgoing = min_outgoing
        self.epoch = 0
        self.curve = []
        self.swaps_completed = 0

    def run_epoch(self, *, verbose: bool = False) -> dict:
        if self.epoch >= self.epochs:
            raise RuntimeError('Training already reached planned epoch limit')
        (x_train, y_train), (x_val, y_val), _ = self.dataset
        indices = self.rng.permutation(len(x_train))
        training_loss = 0.0
        total_seen = 0
        for start in range(0, len(x_train), self.batch_size):
            selection = indices[start:start + self.batch_size]
            x = Tensor(x_train[selection], requires_grad=False)
            self.model.zero_grad()
            logits = self.model(x)
            loss = cross_entropy(logits, y_train[selection])
            if not np.isfinite(loss.data).all():
                raise FloatingPointError('Loss is not finite')
            loss.backward()
            self.pruner.observe_gradients()
            self.optimizer.step()
            training_loss += float(loss.data) * len(selection)
            total_seen += len(selection)
        desired = target_for_epoch(self.epoch, self.epochs, self.target)
        self.pruner.prune_to(desired)  # topology changes only between steps
        # Optional fixed-budget regrowth: after the first masking epoch, a
        # dormant edge can accumulate effective-gradient evidence in batches.
        if self.regrow_swaps and self.model.active_weights() < self.model.total_weights():
            self.swaps_completed += self.pruner.regrow_swap(self.regrow_swaps)
        record = dict(epoch=self.epoch + 1, training_loss=training_loss / total_seen,
                      val_loss=batch_loss(self.model.predict_logits(x_val), y_val),
                      train_accuracy=accuracy(self.model, x_train, y_train),
                      val_accuracy=accuracy(self.model, x_val, y_val),
                      sparsity=self.model.sparsity())
        self.curve.append(record)
        self.epoch += 1
        if verbose and (self.epoch % max(1, self.epochs // 8) == 0
                        or self.epoch == 1):
            print(f"  epoch={self.epoch:02d} train_loss={record['training_loss']:.4f} "
                  f"val_acc={record['val_accuracy']:.3f} "
                  f"sparsity={record['sparsity']:.3f}", flush=True)
        return record

    def finish(self):
        if self.epoch != self.epochs:
            raise RuntimeError('Finish requires all planned training epochs')
        model = self.model
        dataset = self.dataset
        _, (x_val, y_val), (x_test, y_test) = dataset
        count = int(round(self.target * model.total_weights()))
        assert model.total_weights() - model.active_weights() == count
        sparse_model = SparseMLP.from_model(model)
        dense_output = model.predict_logits(x_test)
        sparse_output = sparse_model.predict_logits(x_test)
        np.testing.assert_allclose(sparse_output, dense_output, rtol=1e-9, atol=1e-9)
        report = model.connectivity_report()
        result = dict(seed=self.seed, criterion=self.criterion,
                      target_sparsity=self.target, achieved_sparsity=model.sparsity(),
                      total_weights=model.total_weights(),
                      active_weights=model.active_weights(),
                      dense_macs_per_sample=model.total_weights(),
                      sparse_macs_per_sample=sparse_model.nnz(),
                      dense_weight_flops_per_sample=2 * model.total_weights(),
                      sparse_weight_flops_per_sample=2 * sparse_model.nnz(),
                      validation_accuracy=accuracy(model, x_val, y_val),
                      test_accuracy=accuracy(model, x_test, y_test),
                      test_examples=int(len(y_test)),
                      final_training_loss=self.curve[-1]['training_loss'],
                      final_validation_loss=self.curve[-1]['val_loss'],
                      total_disconnected_incoming=sum(r['zero_incoming'] for r in report),
                      total_disconnected_hidden_outgoing=sum(r['zero_outgoing'] or 0
                                                             for r in report),
                      regrow_swaps_completed=self.swaps_completed,
                      min_incoming=self.min_incoming,
                      min_outgoing=self.min_outgoing)
        assert result['active_weights'] == result['sparse_macs_per_sample']
        return model, sparse_model, self.curve.copy(), result, dataset

    def save_checkpoint(self, path: str | Path) -> None:
        """Full training checkpoint, unlike the separate sparse inference export."""
        self.model.assert_idle()
        payload = {**self.model.state_dict(), **self.optimizer.state_dict(),
                   **self.pruner.state_dict()}
        metadata = dict(format_version=1, epoch=self.epoch, epochs=self.epochs,
                        seed=self.seed, batch_size=self.batch_size, lr=self.lr,
                        target=self.target, criterion=self.criterion,
                        regrow_swaps=self.regrow_swaps,
                        swaps_completed=self.swaps_completed,
                        min_incoming=self.min_incoming,
                        min_outgoing=self.min_outgoing,
                        data_sha256=data_fingerprint(self.dataset),
                        rng_state=self.rng.bit_generator.state,
                        curve=self.curve)
        payload['metadata_json'] = np.asarray(json.dumps(metadata))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, dataset=None):
        # Never unpickle arbitrary objects from a model checkpoint.
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json'].item()))
            if metadata['format_version'] != 1:
                raise ValueError('Unsupported checkpoint version')
            state = {name: data[name].copy() for name in data.files
                     if name != 'metadata_json'}
        session = cls(seed=metadata['seed'], epochs=metadata['epochs'],
                      batch_size=metadata['batch_size'], lr=metadata['lr'],
                      target=metadata['target'], criterion=metadata['criterion'],
                      regrow_swaps=metadata['regrow_swaps'],
                      min_incoming=metadata['min_incoming'],
                      min_outgoing=metadata['min_outgoing'], dataset=dataset)
        if metadata['data_sha256'] != data_fingerprint(session.dataset):
            raise ValueError('Checkpoint data/split fingerprint mismatch')
        if not 0 <= metadata['epoch'] <= session.epochs:
            raise ValueError('Invalid checkpoint epoch')
        if len(metadata['curve']) != metadata['epoch']:
            raise ValueError('Training history does not match epoch count')
        session.model.load_state_dict(state)
        session.optimizer.load_state_dict(state)
        session.pruner.load_state_dict(state)
        session.rng.bit_generator.state = metadata['rng_state']
        session.epoch = metadata['epoch']
        session.curve = metadata['curve']
        session.swaps_completed = metadata['swaps_completed']
        return session


def train_model(*, seed: int = 0, epochs: int = 32, batch_size: int = 64,
                lr: float = 0.003, target: float = 0.0, criterion: str = 'taylor',
                dataset=None, verbose: bool = False, regrow_swaps: int = 0,
                min_incoming: int = 1, min_outgoing: int = 1):
    session = TrainingSession(seed=seed, epochs=epochs, batch_size=batch_size,
                              lr=lr, target=target, criterion=criterion,
                              dataset=dataset, regrow_swaps=regrow_swaps,
                              min_incoming=min_incoming, min_outgoing=min_outgoing)
    for _ in range(epochs):
        session.run_epoch(verbose=verbose)
    return session.finish()

def benchmark(dense: MLP, sparse: SparseMLP, dataset,
              repeats: int = 30, warmup: int = 5) -> dict[str, float]:
    """Report measured latency, not assumed speedups. Export happens before timing."""
    x_test = dataset[2][0]
    result = {}
    for batch in (1, 32):
        x = np.ascontiguousarray(x_test[:batch])
        for label, fn in [('dense', dense.predict_logits),
                          ('sparse', sparse.predict_logits)]:
            for _ in range(warmup):
                fn(x)
            ms = []
            for _ in range(repeats):
                start = perf_counter_ns()
                fn(x)
                ms.append((perf_counter_ns() - start) / 1e6)
            result[f'{label}_batch{batch}_median_ms'] = float(np.median(ms))
            result[f'{label}_batch{batch}_p95_ms'] = float(np.percentile(ms, 95))
    return result


def save_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No records to save')
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning(curve: list[dict], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    epochs = [r['epoch'] for r in curve]
    axes[0].plot(epochs, [r['training_loss'] for r in curve], label='train loss')
    axes[0].plot(epochs, [r['val_loss'] for r in curve], label='validation loss')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Loss')
    axes[0].legend()
    axes[1].plot(epochs, [r['train_accuracy'] for r in curve], label='train acc')
    axes[1].plot(epochs, [r['val_accuracy'] for r in curve], label='validation acc')
    axes[1].plot(epochs, [r['sparsity'] for r in curve], label='weight sparsity', ls='--')
    axes[1].set(xlabel='Epoch', ylabel='Fraction', ylim=(0, 1.05), title='Accuracy and pruning')
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def environment() -> dict:
    from importlib.metadata import version
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__, 'sklearn_loader_version': version('scikit-learn'),
            'blas_note': 'Runtime and BLAS thread settings affect measured latency.'}
