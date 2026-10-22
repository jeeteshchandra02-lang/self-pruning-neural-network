"""Falsify/validate Taylor's LOCAL approximation against actual edge ablation.

This is a diagnostic on a held-out VALIDATION subset. Test labels are never
used to choose pruning scores or hyperparameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from engine import Tensor, cross_entropy
from train.runner import ROOT, TrainingSession, batch_loss, save_records


def measure_single_edge_damage(session: TrainingSession, n_edges: int = 80,
                               val_batch: int = 128, seed: int = 10) -> list[dict]:
    if n_edges <= 0 or val_batch <= 0:
        raise ValueError('n_edges and val_batch must be positive')
    model = session.model
    model.assert_idle()
    (x_val, y_val) = session.dataset[1]
    x, y = x_val[:val_batch], y_val[:val_batch]
    model.zero_grad()
    cross_entropy(model(Tensor(x)), y).backward()
    base = batch_loss(model.predict_logits(x), y)
    available = [(li, *tuple(int(j) for j in coord))
                 for li, layer in enumerate(model.linear_layers())
                 for coord in np.argwhere(layer.mask)]
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(available), size=min(n_edges, len(available)),
                          replace=False)
    rows = []
    for pick in selected:
        li, row, col = available[int(pick)]
        layer = model.linear_layers()[li]
        w = float(layer.weight.data[row, col])
        g = float(layer.weight.grad[row, col])
        # Because mask is toggled for an inference-only dense evaluation,
        # the analytic graph and optimizer state are never modified.
        layer.mask[row, col] = False
        try:
            new_loss = batch_loss(model.predict_logits(x), y)
        finally:
            layer.mask[row, col] = True
        true_change = new_loss - base
        prediction = -w * g  # signed first-order ΔL, not an equality
        rows.append(dict(layer=li, row=row, col=col, weight=w, gradient=g,
                         taylor_signed_prediction=prediction,
                         taylor_absolute_importance=abs(w * g),
                         true_validation_loss_change=true_change,
                         absolute_prediction_error=abs(prediction - true_change)))
    return rows


def plot_damage(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    predicted = [r['taylor_signed_prediction'] for r in rows]
    observed = [r['true_validation_loss_change'] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(predicted, observed, s=15, alpha=.75)
    lo, hi = min(predicted + observed), max(predicted + observed)
    ax.plot([lo, hi], [lo, hi], ls='--', label='Exact agreement')
    ax.set(xlabel='Signed first-order prediction: -weight × gradient',
           ylabel='Actual validation-loss change after deleting one edge',
           title='Taylor importance is an approximation, not a guarantee')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--sparsity', type=float, default=.90)
    parser.add_argument('--samples', type=int, default=80)
    args = parser.parse_args()
    session = TrainingSession(seed=args.seed, epochs=args.epochs,
                              target=args.sparsity)
    for _ in range(session.epochs):
        session.run_epoch()
    rows = measure_single_edge_damage(session, n_edges=args.samples)
    path = ROOT / 'artifacts'
    save_records(path / 'raw' / 'saliency_audit.csv', rows)
    plot_damage(rows, path / 'plots' / 'saliency_vs_loss.png')
    print(f'Examined {len(rows)} active edges against actual validation-loss '
          f'changes. Mean absolute error: '
          f'{np.mean([r["absolute_prediction_error"] for r in rows]):.6g}.')
    print('Saved artifacts/raw/saliency_audit.csv and artifacts/plots/saliency_vs_loss.png')


if __name__ == '__main__':
    main()
