"""Part 2: python -m train.dense --seed 0"""
import argparse
import json
from train.runner import ROOT, train_model, save_records, plot_learning


def main():
    parser = argparse.ArgumentParser(description='Train dense MLP using custom autodiff')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    args = parser.parse_args()
    _, _, curve, result, _ = train_model(seed=args.seed, epochs=args.epochs,
                                         batch_size=args.batch_size, lr=args.lr,
                                         target=0.0, verbose=True)
    result_dir = ROOT / 'artifacts'
    save_records(result_dir / 'raw' / 'dense_learning_curve.csv', curve)
    plot_learning(curve, result_dir / 'plots' / 'dense_learning_curve.png',
                  f'Dense MLP, seed={args.seed}')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
