"""Part 3: python -m train.sparse --criterion taylor --sparsity 0.90"""
import argparse
import json
from train.runner import ROOT, train_model, benchmark, save_records, plot_learning


def main():
    parser = argparse.ArgumentParser(description='Train with progressive self-pruning')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--sparsity', type=float, default=0.90)
    parser.add_argument('--criterion', choices=['taylor', 'magnitude'], default='taylor')
    parser.add_argument('--regrow-swaps', type=int, default=0,
                        help='Optional gradient-guided budget-preserving edge swaps per epoch')
    args = parser.parse_args()
    model, sparse, curve, result, dataset = train_model(
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, target=args.sparsity, criterion=args.criterion,
        regrow_swaps=args.regrow_swaps, verbose=True)
    result.update(benchmark(model, sparse, dataset))
    path = ROOT / 'artifacts'
    save_records(path / 'raw' / 'pruned_learning_curve.csv', curve)
    plot_learning(curve, path / 'plots' / 'pruned_learning_curve.png',
                  f'{args.criterion} pruning, target={args.sparsity:.0%}, seed={args.seed}')
    sparse.save(str(path / 'sparse_model.npz'))
    (path / 'raw' / 'pruned_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
