# Correctness traps — what a skeptical reviewer could test

These are *plausible engineering checks* derived from the submitted exercise,
not secret tests known to us. Reproduce each failure locally before changing
any code. Relevant tests are in `tests/`.

| # | Bug / interviewer prompt | Why it breaks | Our prevention / test |
|---|---|---|---|
| 1 | Forward `X + bias` but return `(N,D)` bias grad | Bias was broadcast across N rows | `sum_to_shape`; `test_add_broadcast` |
| 2 | Only reverse one of multiple parents | DAG may reuse values | `test_graph_accumulation_diamond` |
| 3 | Assign `parent.grad = contribution` | Overwrites previous path's contribution | Accumulate with `+=`; diamond test |
| 4 | Traverse graph in creation order | Parent processed before all child gradients arrive | Reverse topological order |
| 5 | Reuse the *same* graph twice without clearing intermediates | Stale intermediate gradients multiply contributions | Intermediate reset; repeated backward test |
| 6 | Clear all leaves inside `backward()` | Breaks intentional gradient accumulation | Leaf grads cleared explicitly by training loop |
| 7 | Derivative of `a/b` to b has positive sign | Must be `-ga/b²` | finite-difference division check |
| 8 | Matmul with mismatched transposes | Square-only tests can hide it | Non-square `(3,5)@(5,4)` gradcheck |
| 9 | Divide `mean(axis=1)` by entire tensor size | Wrong axis-specific normalization | Mean axis gradcheck |
| 10 | Finite-difference ReLU at exact zero | No unique derivative at kink | Check at ±nonzero samples |
| 11 | Compute raw `exp(1000)` in CE | Overflow and NaNs | Stable max-shift fused CE test |
| 12 | Forget division by batch size in CE grad | Gradient scales with minibatch length | CE gradcheck |
| 13 | Accept negative class labels (`-1` indexing!) | Could silently select last class | Input validation test |
| 14 | Only multiply `W.data *= mask` once | Adam can regrow and compute masked gradient | Graph-level `W*mask` and optimizer freeze |
| 15 | Mask forward but omit mask in backward | `W.grad` not zero at deleted edges | `test_masked_forward_and_gradient` |
| 16 | Multiply gradients by mask after computing no-mask forward | Forward was not actually sparse/zero | Explicit masked tensor operation |
| 17 | Change mask between forward and backward | Backward may see inconsistent graph | Graph lease + pre-backward mutation check; adversarial test |
| 18 | Zero pruned gradient but keep old Adam momentum | Adam `m` moves dead weight | Poisoned-state test and reset |
| 19 | Reset m/v, reuse large global Adam age on revival | Bias correction no longer treats edge as fresh | Per-coordinate age + hand-calculated first step |
| 20 | Revive only mask, retain old weight and momentum | Stale history silently changes training | `GlobalPruner.revive`; zero weight and optimizer state |
| 21 | Claim `masked_weight.grad` nonzero because it might regrow | Conflates `dL/dW` and `dL/dW_eff` | Chain-rule derivation in DESIGN.md |
| 22 | Score saliency *after* pruning sets w=0 | Every deleted score becomes zero; wrong comparisons | Observe `abs(w*g)` before optimizer/pruning |
| 23 | Average signed gradient then apply abs | Opposite signs cancel | EMA of `abs(w*g)` per mini-batch |
| 24 | Call first-order score exact loss damage | Neglects curvature/interacting edges | Explicit approximation and empirical comparison |
| 25 | `np.percentile` with equal scores | Overshoots requested active count on ties | Exact-k lexsorted global indices; tied-score test |
| 26 | Compute sparsity including biases in one place | Numerator/denominator contradiction | Central `MLP.total_weights()` only Linear weights |
| 27 | Ignore layer collapse | Extreme global pruning can kill a small layer | Connectivity report and enforced minimum degrees; atomic infeasible-budget test |
| 28 | One-shot remove 90% after final epoch | No opportunity for recovery | Warmup, cubic ramp, recovery |
| 29 | Validation/test leakage | Unreliable generalization estimate | Fixed split before training; fixed pixel scale `/16` |
| 30 | Compare Taylor seed A to magnitude seed B | Initialization confounds result | Matched seed/batch/split per method |
| 31 | Compare methods with different training lengths | More optimization may explain advantage | Identical runner hyperparameters |
| 32 | Run one seed and call a 0.4-pt win definitive | Could be one changed prediction | Five paired seeds and seed SD/raw rows |
| 33 | Plot numbers typed in manually | Plot and CSV may disagree | Plot generated directly from raw experiment rows |
| 34 | Benchmark `X @ (W*mask)` and declare 10× speedup | Dense BLAS still handles all matrix entries | Index-only exported sparse inference path |
| 35 | Assume 90% fewer weights means 90% faster | Sparse indexing/loops dominate tiny model | Report real latency and MACs separately |
| 36 | Mix multiply-add count with 2-FLOP convention | Claims ambiguous 2× | Explicit MAC and approx FLOP definitions |
| 37 | Include model export time in only one path | Biased timing comparison | Export once before bench, warm up both |
| 38 | Use pickle checkpoints from untrusted repos | Deserialization can execute code | `.npz` numerical arrays, `allow_pickle=False` |
| 39 | Update model in place during live concurrent inference | Some requests see partially pruned state | Immutable versioned snapshots + atomic routing |
| 40 | Present unimplemented full RigL as completed | Fixed-budget gradient swaps are supported; not complete RigL | `regrow_swap` test + separate `regrowth_demo.json`; explicit scope |

## Four very small examples to memorize

### A. Broadcasted bias

`X` shape `(2,3)`, `b` shape `(3,)`, `Y=X+b`. If `dL/dY` is an all-ones
matrix, the gradient of `b` is `[2,2,2]`, **not** a `(2,3)` matrix.

### B. Shared-parent graph

`y=x*x+x` has `dy/dx = 2*x+1`, not `x+1` or `2*x`.

### C. Masked weight

If `W=[2,4]`, `M=[1,0]`, and `dL/dE=[3,5]`, the actual stored
parameter gradient is `[3,0]`; hypothetical effective-edge gradient is
`[3,5]`. This is the exact distinction an interviewer might probe.

### D. Stale Adam first moment

If the old first moment is `m_old=10`, and you now set `g=0`, an
unmasked Adam update still has `m_new = beta1*10`, which is nonzero.
**Zero gradient is not zero optimizer update**. We erase the state and
explicitly freeze the coordinate.

## How to demonstrate the project caught a real bug

The test `test_adam_rejects_an_illegal_masked_gradient` passes a deliberately
**incorrect nonzero masked gradient** to Adam. The optimizer checks and
raises an assertion instead of silently zeroing it. This is intentional:
otherwise an incorrect custom backward implementation could appear to work
because the optimizer covered up the evidence. We also poison optimizer state
with large values before pruning and verify every masked coordinate stays
exactly zero for multiple steps.

Run only these tests:

```bash
python -m unittest discover -s tests -p 'test_masks_and_adam.py' -v
```

## Additional caught issues addressed by hardening

| # | Issue | Defense and specific evidence |
|---|---|---|
| 41 | Mid-step `optimizer.step` silently mutates weights | Pending-graph guard in Adam; `test_illegal_prune_after_forward_refused_and_backward_recovers` |
| 42 | Direct mask edit invalidates a snapshot | Backward preflight guard before gradients; `test_direct_mask_edit_before_backward_is_detected_without_grads` |
| 43 | Direct stored-weight edit invalidates graph | Snapshot verification; `test_direct_weight_mutation_before_backward_is_detected` |
| 44 | Ignore masked upstream `dL/dX` | Both analytical and finite-difference masked-input tests |
| 45 | Gradient of absent weight conflated with growth score | Save `latest_effective_gradient`; regrowth uses dormant EMA; `test_counterfactual_regrowth_keeps_hard_budget_and_clears_adam` |
| 46 | Regrowth violates hard sparsity cap | Select equal numbers of live removals and dormant insertions; assert unchanged count |
| 47 | Infeasible degree safeguards half-prune the network | Plan exact selection before mutations, raise `ValueError` atomically |
| 48 | Pretend Taylor always approximates actual damage | 80-edge empirical validation audit, scatterplot and quadratic zero-gradient counterexample |
| 49 | Resume forgets Adam age/pruning state/RNG | Full `.npz` training checkpoint; compare array-exact resumed/uninterrupted training |
| 50 | Optimizer updates half the model before finding an invalid gradient | Validate all gradients prior to any updates; atomic preflight test |

### Adversarial tests and demonstrations

```bash
python -m unittest discover -s tests -v
python -m train.audit --seed 0 --epochs 32 --sparsity .90 --samples 80
python -m train.sparse --seed 0 --sparsity .90 --regrow-swaps 4
python -m train.checkpoint --epochs 16 --stop-after 8
python -m train.checkpoint --resume
```

**Known trade-off:** The graph lease is intentionally strict: use the explicit
NumPy evaluation path when you do not intend to call backward on a tensor.
The static mask is treated as a constant during a graph. Advanced multi-graph
weight sharing is beyond the tiny MLP and not claimed here.
