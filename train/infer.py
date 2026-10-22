"""Load the committed sparse-only model; no autodiff/optimizer or dense zeros."""
import argparse
import json
import numpy as np

from data import load_data
from prune import SparseMLP
from train.runner import ROOT


def main():
    parser = argparse.ArgumentParser(description='Inference using only surviving edges')
    parser.add_argument('--model', default=str(ROOT / 'artifacts' / 'sparse_model.npz'))
    parser.add_argument('--index', type=int, default=0,
                        help='Index into fixed held-out test split')
    args = parser.parse_args()
    x_test, y_test = load_data()[2]
    if not 0 <= args.index < len(y_test):
        parser.error(f'index must be 0..{len(y_test)-1}')
    sparse = SparseMLP.load(args.model)
    prediction = int(np.argmax(sparse.predict_logits(x_test[args.index:args.index+1])))
    print(json.dumps({'index': args.index, 'prediction': prediction,
                      'label': int(y_test[args.index]),
                      'active_weight_connections': sparse.nnz()}, indent=2))


if __name__ == '__main__':
    main()
