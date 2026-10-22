"""Demonstrate genuine stop-and-resume, with full optimizer and RNG state.

Pause:  python -m train.checkpoint --epochs 16 --stop-after 8
Resume: python -m train.checkpoint --resume --checkpoint artifacts/training_state.npz
"""
import argparse
import json
from train.runner import ROOT, TrainingSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(ROOT / 'artifacts' / 'training_state.npz'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--stop-after', type=int, default=None)
    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--regrow-swaps', type=int, default=3)
    args = parser.parse_args()
    session = (TrainingSession.load_checkpoint(args.checkpoint) if args.resume else
               TrainingSession(seed=args.seed, epochs=args.epochs,
                               target=args.sparsity, regrow_swaps=args.regrow_swaps))
    end = session.epochs if args.stop_after is None else args.stop_after
    if not session.epoch <= end <= session.epochs:
        parser.error('stop-after must be between completed and planned epochs')
    while session.epoch < end:
        session.run_epoch(verbose=True)
    session.save_checkpoint(args.checkpoint)
    print(f'Full training checkpoint saved: {args.checkpoint} '
          f'(epoch {session.epoch}/{session.epochs})')
    if session.epoch == session.epochs:
        print(json.dumps(session.finish()[3], indent=2))


if __name__ == '__main__':
    main()
