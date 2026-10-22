# AQUA defensive review — fixes, evidence, and remaining limits

This is an engineering audit **derived from the posted assignment**, not an
assertion of access to an undisclosed interviewer's hidden tests.

| Risk | Change in this version | How to verify |
|---|---|---|
| Mask changed between forward and backward | Pending graph lease; mask/weight/bias snapshot preflight; reject optimizer/pruner mutation | `test_direct_mask_edit_before_backward_is_detected_without_grads`, `test_illegal_prune_after_forward_refused_and_backward_recovers` |
| Pruned weight leaks gradient to earlier layer | Use `W * M` for *both* derivatives | `test_masked_input_gradient_and_counterfactual_gradient`, masked finite differences |
| Stale Adam first/second moments and bias age | Reset moments and per-coordinate active age on prune/revive; preflight all gradient values before mutating any layer | `test_poisoned_moments_never_move_frozen_weight`, `test_adam_bad_gradient_does_not_half_update_model` |
| Dormant gradient confused with trained parameter gradient | Copy intermediate effective weight gradient separately, preserve exact zero `W.grad` | `test_counterfactual_regrowth_keeps_hard_budget_and_clears_adam` |
| Revived connections violate target budget | Swap `k` inactive-to-active and `k` active-to-inactive using the two distinct importance scores; reset all affected Adam state | Same regrowth test; `artifacts/raw/regrowth_demo.json` |
| Global pruning kills a neuron | Configurable minimum incoming per output and outgoing per hidden unit; connectivity report | `test_dead_neuron_protection_and_atomic_infeasible_budget`, sweep result fields |
| Requested sparsity infeasible under constraints | Plan exact-k selection without mutations; raise descriptive error | Atomic infeasibility test |
| Taylor saliency asserted as exact damage | Compare signed prediction with actual validation loss following individual edge ablation | `artifacts/raw/saliency_audit.csv`, `artifacts/plots/saliency_vs_loss.png` |
| Checkpoint cannot safely resume training | Save model, binary masks, Adam moments/ages, both importance EMAs, seeded RNG, curve, epoch, dataset fingerprint | `test_full_optimizer_rng_mask_and_regrowth_resume_exact`; `python -m train.checkpoint` |
| Raw numbers and plot/claim not reproducible | Five paired seeds, raw CSV, summary, derived falsifiable claim; test artifact consistency | `test_paired_sweep_csv_summary_claim_checkpoint_agree` |
| Fake sparse speedup from dense zeros | Sparse-only index traversal path and separately measured latency | `tests/test_sparse_and_training.py`, raw per-run timing columns |

## Explicitly not claimed as solved

- Python sparse indexing is **slower** than dense BLAS for this small model;
  true deployment latency gains require a target-hardware compiled/structured
  sparse kernel and evidence on real traffic. We do not claim one.
- The optional growth policy is a minimal *gradient-guided fixed-budget swap*,
  not a full implementation of published RigL training.
- A single Linear instance has one pending differentiable pass at a time, and
  the tiny tensor engine deliberately implements matrix multiplication for
  2-D arrays only. The assignment's MLP needs neither multi-graph sharing nor
  batched higher-dimensional matmul.
- Full training resume is supported at **epoch boundaries** and requires the
  same planned total number of epochs and identical dataset/split; arbitrary
  minibatch interruption is not supported.
- Passing **all included tests** is not a guarantee against undisclosed tests.

## Reproducible commands

Run from repository root after installing requirements:

```bash
python -m unittest discover -s tests -v
python -m train.infer --index 0
python -m train.sweep --sparsities 0 .5 .75 .9 .95 --criteria taylor magnitude --seeds 0 1 2 3 4 --epochs 32 --benchmark-repeats 16
python -m train.audit --seed 0 --epochs 32 --sparsity .9 --samples 80
python -m train.sparse --seed 0 --epochs 32 --sparsity .9 --regrow-swaps 4
python -m train.checkpoint --epochs 16 --stop-after 8
python -m train.checkpoint --resume
```

**Warning:** the standalone `train.sparse` command writes demonstration
`artifacts/plots/pruned_learning_curve.png` and `artifacts/sparse_model.npz`.
To restore the committed baseline sweep's seed-0 model/curve, rerun the full
sweep last (the demonstration's `regrowth_demo.json` remains separate).
