"""Part 4: paired-seed Pareto sweeps with raw CSV and an honest measured cost."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import load_data
from train.runner import (ROOT, train_model, benchmark, save_records,
                          plot_learning, environment)


def plot_pareto(results: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for method in sorted(set(r['criterion'] for r in results)):
        subset = [r for r in results if r['criterion'] == method]
        targets = sorted(set(r['target_sparsity'] for r in subset))
        x, means, stds = [], [], []
        for target in targets:
            rows = [r for r in subset if r['target_sparsity'] == target]
            x.append(float(np.mean([r['achieved_sparsity'] for r in rows])) * 100)
            accuracy = [r['test_accuracy'] * 100 for r in rows]
            means.append(float(np.mean(accuracy)))
            stds.append(float(np.std(accuracy, ddof=1)) if len(rows) > 1 else 0)
        ax.errorbar(x, means, yerr=stds, marker='o', capsize=3, label=method)
    ax.set(xlabel='Achieved weight sparsity (%)', ylabel='Test accuracy (%)',
           title='Accuracy–sparsity trade-off (error bars: seed std dev)',
           xlim=(-2, 100), ylim=(0, 105))
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(results: list[dict]) -> list[dict]:
    summary = []
    for criterion in sorted(set(r['criterion'] for r in results)):
        for target in sorted(set(r['target_sparsity'] for r in results)):
            rows = [r for r in results if r['criterion'] == criterion
                    and r['target_sparsity'] == target]
            if not rows:
                continue
            accuracies = [r['test_accuracy'] for r in rows]
            summary.append(dict(criterion=criterion, target_sparsity=target,
                                seeds=len(rows), achieved_sparsity=float(np.mean(
                                    [r['achieved_sparsity'] for r in rows])),
                                test_accuracy_mean=float(np.mean(accuracies)),
                                test_accuracy_std=float(np.std(accuracies, ddof=1))
                                if len(rows) > 1 else 0.0,
                                active_macs=int(rows[0]['sparse_macs_per_sample']),
                                dense_macs=int(rows[0]['dense_macs_per_sample'])))
    return summary


def generate_claim(rows: list[dict], target: float = 0.90) -> str:
    a = {r['seed']: r for r in rows if r['criterion'] == 'taylor'
         and r['target_sparsity'] == target}
    b = {r['seed']: r for r in rows if r['criterion'] == 'magnitude'
         and r['target_sparsity'] == target}
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return 'No paired Taylor/magnitude runs available at target 90%.\n'
    at = np.array([a[s]['test_accuracy'] * 100 for s in seeds])
    bt = np.array([b[s]['test_accuracy'] * 100 for s in seeds])
    d = at - bt
    sd = lambda arr: float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    r = a[seeds[0]]
    return (f'Across {len(seeds)} paired seeds {seeds}, with a nominal '
            f'{target:.0%} target (achieved {r["achieved_sparsity"]:.3%}), '
            f'Taylor pruning achieved {at.mean():.2f} ± {sd(at):.2f} percentage '
            f'points test accuracy (mean ± seed SD) versus '
            f'{bt.mean():.2f} ± {sd(bt):.2f} for magnitude; '
            f'paired mean difference Taylor minus magnitude was '
            f'{(0.0 if abs(d.mean()) < 0.005 else d.mean()):+.2f} percentage points (paired SD {sd(d):.2f}).\n'
            f'The sparse inference path visits {r["sparse_macs_per_sample"]} '
            f'active weight MACs per example, versus '
            f'{r["dense_macs_per_sample"]} for the dense path. '
            f'These weight MAC counts do not include biases or activations, '
            f'and do not establish a latency speedup. See benchmark results '
            f'for measured wall-clock times.\n')


def main():
    parser = argparse.ArgumentParser(description='Paired sparsity x seed sweep')
    parser.add_argument('--sparsities', type=float, nargs='+',
                        default=[0, 0.50, 0.75, 0.90, 0.95])
    parser.add_argument('--criteria', choices=['taylor', 'magnitude'], nargs='+',
                        default=['taylor', 'magnitude'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--benchmark-repeats', type=int, default=30)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate random seeds supplied')
    dataset = load_data()
    rows = []
    for target in args.sparsities:
        for seed in args.seeds:
            for method in args.criteria:
                print(f'Running criterion={method} target={target:.0%} seed={seed}', flush=True)
                model, sparse, curve, record, _ = train_model(
                    seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                    target=target, criterion=method, lr=args.lr, dataset=dataset)
                record.update(benchmark(model, sparse, dataset,
                                        repeats=args.benchmark_repeats))
                rows.append(record)
                print(f'  test={record["test_accuracy"]:.3f} '
                      f'active={record["active_weights"]}/{record["total_weights"]}',
                      flush=True)
                if target == 0.90 and seed == args.seeds[0] and method == 'taylor':
                    save_records(ROOT / 'artifacts/raw/pruned_learning_curve.csv', curve)
                    plot_learning(curve, ROOT / 'artifacts/plots/pruned_learning_curve.png',
                                  f'{method} pruning, target={target:.0%}, seed={seed}')
                    sparse.save(str(ROOT / 'artifacts/sparse_model.npz'))
                if target == 0 and seed == args.seeds[0] and method == 'taylor':
                    save_records(ROOT / 'artifacts/raw/dense_learning_curve.csv', curve)
                    plot_learning(curve, ROOT / 'artifacts/plots/dense_learning_curve.png',
                                  f'Dense MLP, seed={seed}')
    base = ROOT / 'artifacts'
    save_records(base / 'raw/sweep_results.csv', rows)
    save_records(base / 'raw/sweep_summary.csv', summarize(rows))
    plot_pareto(rows, base / 'plots/pareto_accuracy.png')
    (base / 'raw/falsifiable_claim.txt').write_text(generate_claim(rows), encoding='utf8')
    (base / 'raw/experiment_config.json').write_text(json.dumps({
        'seeds': args.seeds, 'sparsities': args.sparsities,
        'criteria': args.criteria, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'lr': args.lr,
        'split_seed': 2026, 'benchmark_repeats': args.benchmark_repeats,
        'min_incoming': 1, 'min_outgoing': 1, 'regrow_swaps': 0,
        'environment': environment(),
        'mac_definition': 'One weight multiply-accumulate per surviving connection per example.',
        'flop_definition': 'Approx. 2 FLOPs per weight MAC; excludes bias and activation ops.',
    }, indent=2) + '\n', encoding='utf8')
    print('\n' + generate_claim(rows))


if __name__ == '__main__':
    main()
