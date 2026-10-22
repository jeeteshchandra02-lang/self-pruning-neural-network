# 45-minute live technical defense — whiteboard and coding preparation

Start from first principles. The assignment permits AI assistance but explicitly
says candidates must defend every line. Be prepared to type an operation and
explain the gradient rather than reciting a project description.

## 0–5 minutes: the architectural walkthrough

**Say:** “I built a tiny reverse-mode autodiff engine that trains a digits MLP.
Every Linear weight has a fixed-for-the-step binary mask. The mask is part of
the differentiable forward graph, so deleted weights have zero parameter
gradient. Adam resets optimizer state on deletion or revival. A cubic schedule
removes the least important active edges using a smoothed Taylor score. I then
export a separate sparse inference model, verify its predictions match the
masked-dense model, and compare the arithmetic and wall-clock measurements.
Across paired seeds the Taylor method was not consistently better than the
magnitude baseline.”

## 5–15 minutes: five whiteboard derivations

1. **Broadcasted addition:** If `Y=X+b`, `X` is N×D and b is D,
   `∂L/∂b_j = Σ_n ∂L/∂Y_{n,j}`. Explain that broadcasting duplicates `b[j]` N times.
2. **Matrix multiplication:** `Y=XW` gives `∂L/∂X=G Wᵀ` and
   `∂L/∂W=XᵀG`; verify shapes N×I, I×O, N×O.
3. **Stable CE:** `L_i = -z_iy + log Σ_j exp(z_ij)` after max shift;
   average gradient `(softmax(z)-one_hot(y))/N`.
4. **Mask:** `E=M⊙W`, `Y=XE`, so `dL/dE=XᵀG` and `dL/dW=M⊙XᵀG`.
   Explain that absent-edge *candidate gradient* is different from stored
   parameter gradient.
5. **Taylor:** Removal `Δw=-w`; `ΔL≈gΔw=-gw`; use `|gw|` to
   rank approximate **absolute** damage. Explain omitted `0.5 ΔwᵀHΔw`.

## 15–25 minutes: code/optimizer defense

### Expected live coding task A: implement `square`

Derive `y=x²`, `dy/dx=2x`. Use a forward-time copy of `x` (to avoid mutation
between forward and backward) and accumulate parent gradients with `+=`:

```python
# Example code to write LIVE in engine/tensor.py as a new method:
def square(self):
    saved = self.data.copy()
    out = Tensor._result(saved * saved, (self,))
    def backward():
        if self.requires_grad:
            self.grad += out.grad * 2.0 * saved
    out._backward = backward
    return out
```

Test against central differences on random `x`. The important engineering
answer is that a new op needs: forward, VJP, accumulation, tests.

### Expected live coding task B: add `leaky_relu(alpha)`

Forward `max(x, alpha*x)`; derivative 1 for x>0 and alpha for x<0. At x=0
choose/document a subgradient and don't numerical-gradcheck at that kink.
Start with a cached copy of `x`, then implement the VJP and a finite-difference
test with positive and negative samples away from zero.

### Expected live coding task C: add another importance criterion

Use **movement-style** score that accumulates the signed first-order
optimisation tendency `-w*g` before the optimizer update, e.g.
`movement_ema = 0.9*movement_ema + 0.1*(-w*g)`. Large positive tendency means
the gradient step is moving the current weight away from zero; negative values
mean it is tending toward zero. Document precisely whether ranks select low
signed score or absolute score, and compare fairly. It is an ablation, not a
claim of optimality.

### Adam proof that users often miss

Imagine the old Adam moment on a deleted connection is `m=2`. Its current
parameter gradient is zero. Then `m_new=0.9*2=1.8` and Adam **can still move it**.
Our `reset_coordinates` zeros weight, first/second moments and active age.
Even after 100 frozen steps, the value remains exactly zero. Revival begins
from zero state; per-coordinate age starts from one on its first active step.

## 25–35 minutes: evidence, cost, falsification

**Say:** “At nominal 90% sparsity the actual mask contains 1,123 of 11,232
weight connections. An example must traverse only 1,123 active weight edges in
our exported sparse forward path, compared to 11,232 dense weight MACs. The
`numpy` sparse code is intentionally transparent, not optimized, and on this
small problem Python indexing dominates the clock time. Therefore I claim
reduced weight arithmetic, not reduced server cost. I show raw per-seed rows
and measured batch-1 and batch-32 latency separately.”

Use the final `artifacts/raw/falsifiable_claim.txt` for the actual five-seed
accuracy numbers, **not a memorized optimistic estimate**. The result may show
no Taylor advantage at 90%. Show the 75% result if asked whether Taylor always
wins; it does not. Do not call a small test-set difference statistically
significant unless tested.

## 35–45 minutes: production architecture at scale

**Say:** “Never deploy the Python graph/optimizer in a high-throughput serving
process. Export versioned immutable sparse indices and weights, load only
validated numeric files, and benchmark a compiled kernel on the actual CPU/GPU.
Authenticate tenant identities, enforce per-tenant admission/rate/batch limits,
route by model version, dynamic-batch compatible requests, prewarm a replacement
model, and atomically promote or roll back the snapshot. Track P50/P95/P99,
throughput, queueing time, per-tenant resource usage, and failure rates. Training
and pruning run separately from inference.”

## 45 high-probability cross-questions — concise spoken answers

1. **Why do this without PyTorch?** Assignment requires proving the underlying autodiff and optimizer behavior ourselves.
2. **Why NumPy?** Efficient array computation but no framework-provided neural gradients.
3. **What is reverse-mode autodiff?** Backpropagate gradients from one scalar loss through a topologically ordered computation graph.
4. **Why a topological sort?** Every downstream gradient contribution must arrive before an operation backpropagates.
5. **Why `+=`?** A value can influence loss by multiple paths, so gradients add.
6. **Why `sum_to_shape`?** Broadcasting duplicates values; backward sums across all duplicated locations.
7. **Can you support arbitrary `matmul` shapes?** The tiny engine intentionally supports only 2-D matmul, which is sufficient here and tested.
8. **Derivative of `A@B`?** `dA=G@B.T`; `dB=A.T@G`.
9. **Derivative of `a/b` to `b`?** `-g*a/b²`, reduced over broadcast axes.
10. **Why is ReLU not checked at zero?** It has no unique derivative there.
11. **Derivative of tanh?** `1-tanh(x)²`.
12. **Why stable softmax CE?** Max shift prevents numerical overflow; fused CE gives `(p-onehot)/N`.
13. **What if logits contain infinity?** Reject input rather than silently returning invalid gradients.
14. **Why use finite differences?** Independent numerical comparison against analytic backward code.
15. **What epsilon?** `1e-5`, in float64, with absolute and relative tolerances.
16. **Why does a masked edge have zero gradient?** Because `E=M*W` and `dL/dW=M*dL/dE` for fixed M.
17. **Is `dL/dE` also zero at a masked edge?** Not necessarily; it measures a hypothetical effective edge independently of W's mask.
18. **How do you choose regrowth candidates?** Compute a separate dense `dL/dE` periodically; *not* using masked `W.grad`.
19. **Do you implement automatic regrowth?** No. Manual revival with clean Adam state is implemented/tested, which is within mandatory scope.
20. **Can you prune in the middle of backward?** No. It breaks forward/backward consistency; topology transitions are between steps.
21. **Why keep an explicit binary mask?** It defines exact zero connections and distinguishes active from inactive weights.
22. **If the raw deleted weight is nonzero by accident?** The forward still multiplies by the zero mask, while optimizer/pruner enforce stored zero as a separate invariant.
23. **Why isn't zero gradient enough with Adam?** Nonzero previous first moment continues producing parameter updates.
24. **Which state do you reset?** Stored weight, first moment, second moment and active-update age; also stale gradient.
25. **Why per-coordinate age?** Correct fresh first-step bias correction when a connection is revived after many global updates.
26. **Why reset both moments?** Second-moment scale is also stale after the topology changes.
27. **Why He initialization?** Hidden layers use ReLU; scale variance by fan-in to maintain activations/gradients.
28. **Why smaller output-layer variance?** Final logits aren't followed by ReLU; use lower variance for stability.
29. **Why use Adam?** Adaptive per-parameter steps, and it stresses correctness of pruning optimizer state.
30. **What is your Taylor derivation?** Removal `Δw=-w`, first-order `ΔL≈-wg`, importance `abs(wg)`.
31. **Does Taylor guarantee improvement over magnitude?** No. Its assumptions fail near stationary points and our measured baseline can be equally good or better.
32. **Why EMA of `abs(wg)`?** Smooth stochastic minibatch sensitivity without cancelling opposite signed gradients.
33. **Why a cubic schedule?** Gradually changes topology and tapers pruning as the model approaches the target; recovery gives optimization time.
34. **Why not prune at the end?** No training steps remain to adapt the surviving weights.
35. **How exactly hit 90%?** Round the target integer prune count and deterministically remove precisely those active indices.
36. **What is in the sparsity denominator?** Only weight-matrix entries, never biases.
37. **What if a whole layer disappears?** Global pruning can do that; monitor per-layer occupancy and use minimum floors if needed, documenting changed budget logic.
38. **Why 5 seeds?** Estimate variability; pair methods under identical initialization/batch randomness.
39. **What is a falsifiable claim?** A concrete mean accuracy difference and active MAC count that raw CSV can verify or contradict.
40. **Can you claim a significant improvement?** Not without justified statistical evidence and a meaningful effect across paired seeds.
41. **What is an active MAC?** One live connection's multiply-accumulate for one example; approximate 2 FLOPs under our convention.
42. **Why slower sparse inference at 90%?** Indexing/gather and Python-loop overhead beat arithmetic savings on these small matrices.
43. **Would a sparse GPU kernel definitely be faster?** No. It depends on sparsity pattern, batch size, hardware, and kernel implementation.
44. **Where does dynamic autodiff spend time?** Python object/closure creation, memory allocations, and callback dispatch; matrix multiplication itself uses native NumPy.
45. **How do you serve thousands of RPS across tenants?** Immutable model versions, compiled kernels, compatible dynamic batching, quotas, isolation, metrics, canaries and rollback.

## Self-check before submission

- Type and derive `sum_to_shape` from memory; show a 2×3 + 3 bias example.
- Type and derive `square` and `leaky_relu` without this file open.
- Explain a pruned coordinate's *zero stored gradient* versus its *effective edge sensitivity* in one minute.
- Predict why naïvely zeroing gradient still allows Adam to update a dead edge.
- Reproduce the five-seed accuracy figure **from CSV**, not from a picture.
- Point at the real sparse loop and identify exactly where deleted edges are skipped.
- Admit if Taylor loses or sparse latency is worse; explain why and how to improve.


## New hardening defense drills

**Q: Why does backward reject mask edits if the engine captured a mask copy?**
A: Snapshotting can produce internally consistent math but permit an optimizer
to update against a stale network. We enforce a stricter lifecycle: changes to
weights or masks before backward are programmer errors. The preflight catches
unmanaged NumPy in-place writes and official topology operations fail fast.

**Q: How can a dormant edge have a nonzero growth score when `W.grad=0`?**
A: `W.grad = M * dL/dE` is zero on dormant edges. The intermediate effective
weight `E=W*M` has its own `dL/dE = X.T @ dL/dY` which is generally nonzero.
After backward we copy the intermediate gradient, accumulate a dormant-only
EMA and use it to identify revival candidates. We do not train masked weights.

**Q: Does revival destroy your 90% budget?**
A: Manual `revive` may change the budget, but production-style
`regrow_swap(k)` replaces k low-score active connections with k high-gradient
dormant connections. It resets weights and Adam moments/ages for both sets
and asserts that the active-edge count is unchanged.

**Q: What happens when exact-k and connectivity are inconsistent?**
A: We ensure at least one incoming weight per output and at least one outgoing
weight per hidden neuron. Ranking skips deletions that break those invariants.
If a requested budget cannot be reached, the selection fails *before*
mutating any masks, and the caller must relax the target/constraints or alter
the architecture.

**Q: Can you pause training and exactly resume?**
A: Yes, at epoch boundaries for an unchanged total schedule and dataset. The
checkpoint contains weights, masks, optimizer moments/ages, both saliency EMAs,
RNG state and curves; the resumed run matches an uninterrupted run array-for-
array under the same environment. Sparse inference `.npz` alone is not enough.

**Q: Have you proven that Taylor beats magnitude?**
A: No. The committed five-seed raw comparison gives only a small average
difference, with substantial paired-seed variation. The empirical ablation
audit also shows that the Taylor estimate differs from the actual loss change.
Both limitations are explicitly documented.
