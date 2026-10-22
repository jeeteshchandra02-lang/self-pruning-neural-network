# DESIGN: The Self-Pruning Network

## 1. Problem framing and choices

**Source requirements:** NumPy-only core, reverse-mode autodiff, broadcast-aware
backprop, finite-difference checks, masked gradients, an optimizer that survives
prune/revive, MLP training, derived importance, gradual schedule, a hard budget,
fair magnitude comparison, real sparse computation evidence, and a falsifiable
claim. Regrowth is optional. The exercise does **not** prescribe architecture,
dataset, accuracy threshold, mask definition, or experimental seed count.

**Implementation choices:** handwritten 70/15/15 stratified split of sklearn's
standard digits **loader only**; MLP `64→96→48→10`; Adam; global unstructured
weight sparsity excluding biases; smoothed first-order Taylor saliency; cubic
schedule with warmup and recovery; paired five-seed experiments; separate sparse
inference format with measured latency and actual NNZ MACs. No neural network,
pruning, or autodiff dependency is used.

## 2. Reverse-mode differentiation

For an operation `y = f(a,b)`, store the two parents and a callback that computes
vector–Jacobian products (VJPs). The graph is a DAG because a node may be reused,
e.g. `z = x*x + x`. Build a topological ordering with a visited set, seed the
scalar loss with gradient 1, then process the reversed order, accumulating with
`+=`. Clearing **intermediate** gradients between repeated backwards avoids
stale DAG contributions; persistent **leaf** gradients accumulate until the
training loop explicitly resets them with `zero_grad()`.

### Broadcast reversal

Forward: `A` has shape `(N,D)` and `b` has shape `(D,)`. The bias component `b[j]`
contributes to `N` outputs. Consequently its derivative is a sum, not an `(N,D)`
array:

\[
 \frac{\partial L}{\partial b_j}
 = \sum_{n=1}^N \frac{\partial L}{\partial y_{n,j}}.
\]

`sum_to_shape` first removes extra leading dimensions then sums along originally
unit-size axes (`keepdims=True`). The same logic is applied to both parents of
addition, subtraction, multiplication and division.

| Forward operation | Backward to first operand | Backward to second operand |
|---|---|---|
| `a + b` | `g` | `g` |
| `a - b` | `g` | `-g` |
| `a * b` | `g*b` | `g*a` |
| `a / b` | `g/b` | `-g*a/b²` |
| `A @ B` (2-D) | `g @ B.T` | `A.T @ g` |

Element-wise VJPs are **summed to each original shape** after calculation.
For `sum(axis=...)`, reshape and broadcast the upstream gradient across reduced
axes. For `mean`, divide by the exact **number of elements reduced**, not always
the full array size. ReLU's gradient is the indicator of `x>0` (defined zero at
zero); tanh's is `1-tanh(x)^2`.

### Stable combined classification loss

Given batch logits `Z` (N,C), integer class labels `y`, and
`a_ij = z_ij - max_k z_ik`, the average loss is:

\[
 L=\frac{1}{N}\sum_i
 \left[\log\left(\sum_j e^{a_{ij}}\right)-a_{i,y_i}\right].
\]

The derivative with respect to one logit is:

\[
 \frac{\partial L}{\partial z_{ij}} =
 \frac{p_{ij} - \mathbf 1[j=y_i]}{N},\qquad
 p_{ij}=e^{a_{ij}}/\sum_k e^{a_{ik}}.
\]

The max subtraction cancels algebraically but prevents overflowing exponentials.
The loss is **fused**, instead of using `np.log(np.exp(z)/sum(exp(z)))`. Invalid
shapes, labels and nonfinite logits fail early.

### Why gradient checks

For differentiable coordinate `x_i`, compare reverse-mode results with central
finite differences:

\[
 g_i^{FD}=\frac{f(x+\epsilon e_i)-f(x-\epsilon e_i)}{2\epsilon}.
\]

Checks use float64, `epsilon=1e-5` and both relative/absolute tolerances, and
exercise broadcasted addition/multiplication/division, non-square matmul,
scalar and axis reductions, ReLU away from its kink, tanh, exp and fused CE.
ReLU at zero has no unique mathematical derivative, so checking central
finite differences at exactly zero would be an invalid test.

## 3. Masked gradients: the intentionally tricky part

Let a linear layer use:

\[
 Y = X(M\odot W)+b,\qquad M_{ij}\in\{0,1\}.
\]

The layer's effective weight is `E=M*W`. By the chain rule:

\[
 \frac{\partial L}{\partial W}
 = M \odot \frac{\partial L}{\partial E}.
\]

For a standard 2-D linear layer, `dL/dE = X.T @ dL/dY`, but **a masked parameter's
actual gradient is zero**. Example: if `dL/dE = [[2,3],[4,5]]` and
`M=[[1,0],[1,1]]`, the gradient of stored `W` is `[[2,0],[4,5]]`.

This does **not** imply that a currently absent edge is useless. A regrowth
algorithm can calculate the *counterfactual* sensitivity `dL/dE` separately,
then use `abs(dL/dE)` as a dormant-edge score. The stored `W.grad` remains zero
at inactive coordinates because the fixed mask is part of the forward function.
`dL/dM = W * dL/dE` is a different gate derivative, and at `W=0` it is also
zero. A zero-initialized reactivated edge can still have nonzero `dL/dE` and a
nonzero first training gradient once its mask becomes 1.

**Lifecycle invariant:** topology changes happen *between* completed steps.
Each differentiable Linear forward registers a pending-graph lease with saved
copies of its mask and parameters. A backward preflight rejects any in-place
mask/parameter edit *before* accumulating gradients. After successful backward
all counterfactual effective-gradient evidence is captured and the lease is
released. `prune_to`, `regrow_swap`, `revive`, checkpoint loading, Adam reset and
optimizer step reject in-flight graphs. For evaluation that will not call
backward, use the separate NumPy `predict_logits`/`dense_logits` methods rather
than abandoning a differentiable Tensor graph. This tiny engine deliberately
allows only one in-flight differentiable pass per Linear; shared Tensor DAG
expressions within a single graph remain valid.

The constant `Tensor(saved_mask.astype(float64))` keeps the forward algebra
consistent. Both the stored-weight gradient `M * dL/dE` and the upstream input
gradient `dL/dX = dL/dY @ (W*M).T` are checked numerically. The engine also
captures `dL/dE` on dormant coordinates to guide optional regrowth while
maintaining exactly zero `dL/dW` on those coordinates.

## 4. Adam and stale state after pruning or revival

Adam keeps first and second gradient moments:

\[
 m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,
 \quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2.
\]

After bias correction, `w -= lr * m_hat / (sqrt(v_hat)+eps)`.
Setting `g=0` for a pruned coordinate **does not zero old `m`**, and Adam can
move a weight even when its current gradient is zero. This is a high-risk hidden
bug. We therefore reset all of these to zero on every prune and manual revive:

```
weight = 0; first_moment = 0; second_moment = 0; active_age = 0
```

The optimizer also freezes absent coordinates, validates zero masked gradient
**before** cleanup, and remasks the stored weight after the update. For a
revived coordinate, `active_age` starts at zero and increases only on actual
active optimizer updates. Bias correction `1-beta**active_age` therefore treats
its first renewed step like a fresh parameter, not like an old coordinate at a
large global step. This per-coordinate age is an intentionally documented
**extension** to ordinary globally-counted Adam. See the poisoned-moment and
revival tests in `tests/test_masks_and_adam.py`.

## 5. Importance: derive what is estimated

Removing an active scalar connection `w` means a finite perturbation
`delta_w=-w`. The first-order Taylor expansion:

\[
 L(w+\delta_w) - L(w) \approx g\,\delta_w=-gw.
\]

Therefore the ranking score is:

\[
 S_{Taylor}=|w g|.
\]

We accumulate an EMA of **per-minibatch absolute scores**, not the absolute of
an average signed gradient (which would cancel opposite signs):

\[
 S_t=0.9 S_{t-1}+0.1|w_t g_t|.
\]

The alternate magnitude baseline is `S_mag=|w|`. Both are ranked across all
active weight entries under **identical** seeds, architecture, initialization,
mini-batch order, optimizer, training schedule and total number of updates.
The tie breaker is stable original flat coordinate index. No percentile
threshold can accidentally overshoot the budget when scores are tied. Our
optional **connectivity safeguard** forbids removal of a hidden neuron's last
outgoing or any neuron's last incoming edge. If the requested exact budget is
infeasible under those constraints, selection fails atomically with a clear
exception. The constraints are enabled in the committed sweep.

**Limitation:** `abs(w*g)` is an approximate *absolute* first-order loss change,
not necessarily the *positive* loss increase. Near a stationary optimum
`g≈0`, the neglected Hessian curvature term `0.5 * delta_w.T H delta_w` may
dominate. Removing many interacting weights also makes independent first-order
scores approximate. **We empirically test, not assume, Taylor > magnitude.**

## 6. Gradual pruning and strict count

Sparsity follows a cubic ramp after 20% warmup and before 20% recovery:

\[
 s(t) = s_f [1-(1-u)^3],\qquad
 u=\operatorname{clip}\left(\frac{t/T-0.2}{0.6},0,1\right).
\]

The requested integer number of pruned weights at an event is
`round(s(t) * total_prunable)`. We rank **only currently active** connections
and prune precisely the extra amount needed. No regrowth is performed in the
experiment sweep, so active count is nonincreasing. After the last topology
event, the remaining epochs allow other parameters to recover. In a separate
optional experiment, dormant edges are ranked by an EMA of `|dL/dE|` and
`regrow_swap(k)` removes the same number of low-score live edges, preserving
both **exact sparsity budget** and minimum connectivity. Regrown weights and
per-coordinate Adam state start at zero; on the next batch they may learn from
new gradients. This small fixed-budget exchange is *inspired by*, but is not
a complete reproduction of, RigL.

**Sparsity definition:** `1 - active_weight_entries / total_weight_entries`
across all `Linear.weight` matrices; biases excluded. At a nominal 90% target,
rounding is discrete, giving 1,123 survivors out of 11,232 and roughly 90.002%
achieved sparsity, not exactly 90.000%. Both target and achieved fractions are
reported.

## 7. Evidence methodology and honest cost accounting

Use the same fixed stratified split for all methods and sparsity levels. Seeds
0–4 control the initial weights and their corresponding mini-batch orders;
the pairing is the **same** for Taylor and magnitude. Data is loaded using
sklearn **only to obtain samples and labels**; all other logic is custom.

The experiment matrix is 5 sparsities × 2 methods × 5 seeds, each 32 epochs,
batch size 64 and Adam learning rate 0.003. We report all per-seed test scores,
cross-seed sample standard deviations and per-seed paired differences. The
279-example test set makes a single misclassified example worth about 0.358
percentage points; small differences should not be declared significant.

For real computation evidence, we export each matrix in an output-column
compressed representation of `(indptr, indices, values)` with **only active
edges stored**. For output column `j`, inference uses:

```
y[:,j] += x[:, input_indices_of_live_edges_j] @ live_weights_j
```

No deleted edge appears in that vector dot product. `SparseMLP` and the regular
masked-dense MLP must agree to `1e-9` in float64 before a result is accepted.
Per example the **weight MAC** count is the number of live connections. We
multiply by two to provide approximate weight FLOPs and **exclude bias,
activation, memory accesses, index selection and Python overhead**. Report
observed median and p95 latency separately for batch sizes 1 and 32 after
warmup and without including export time. Benchmark comparisons come from a
single machine/process. The sparse Python version **may be slower** than dense
BLAS even when live MACs are ten times fewer; the data file reveals this.

These results do NOT prove reduced real production latency, energy use, or
GPU cost. Such claims require a production-grade sparse kernel and a benchmark
on the deployment hardware with the real traffic mix.

Read `artifacts/raw/falsifiable_claim.txt` for the generated quantitative
claim, and `artifacts/raw/sweep_results.csv` to audit every supporting row.

## 8. Where the engine bottlenecks and what to optimize

- Dynamic Python object construction for every mini-batch operation and its
  Python `backward()` closure costs allocations and dispatch time.
- Graph recursion/visited bookkeeping is fine for the tiny MLP but not a general
  scalable tape; iterative traversal and fused operations would help.
- The fused CE removes naive softmax graph nodes, already an example of fusion.
- Dense NumPy matmul is efficient for small matrices because native BLAS hides
  its inner loops; Python sparse loops pay per-column indexing and allocations.
- A production path would compile sparse kernels and choose CSR/CSC, block,
  N:M, or structured sparsity based on deployment hardware. Benchmark the
  total deployment path before claiming latency wins.
- Inference needs **no graph, optimizer, gradient or mask updates**.

## 9. Multi-tenant production deployment at thousands of RPS

```
client → tenant identity/auth → per-tenant quotas and admission control
       → model/version router → compatible-batch queue / dynamic batcher
       → preloaded immutable sparse inference snapshots → response
                                      ↑
    versioned model registry → verify hash + format + index bounds
                            → warm candidate → atomic version promotion
```

The model produced by training is **exported once** to a versioned, immutable
inference-only artifact. `SparseMLP.load` uses `np.load(..., allow_pickle=False)`
and validates pointer monotonicity, weight shapes and index bounds. For a real
service also place explicit model-size caps before allocation, verify hashes,
access-control artifacts by tenant, isolate tenants' compute/batch budgets,
enforce request shape limits and deadlines, and record tenant/model/version
P50/P95/P99 throughput and error metrics. Load and warm a new model before an
atomic pointer swap; don't allow inference requests to observe mutable weights
or a partly changed pruning mask. Deploy canary/rollback. Dynamic batching must
never mix incompatible model versions or leak one tenant's data. Separate the
learning system and serving cluster to avoid contention and for security.

With real model sizes, choose between tenant-specific pools or shared guarded
pools according to access and isolation requirements. Per-tenant concurrency
limits prevent a noisy tenant exhausting all workers. A compiled sparse kernel
and deployment-specific A/B benchmarks are needed before setting an SLA.

## 10. Validating Taylor *against actual deletion damage*

The first-order estimate is mathematically imperfect. To measure its error,
`python -m train.audit` trains a model, takes a fixed validation subset,
computes the full unmasked effective gradient, and independently deletes
sampled **one** active edge at a time only during inference. It reports the
signed prediction `-w*g`, the actual held-out-validation `L(delete)-L(base)`,
and absolute approximation error for each tested connection. The diagnostic
writes raw rows to `artifacts/raw/saliency_audit.csv` and an observed-vs-
predicted scatterplot. It does not use test labels, does not rank hyperparameters
using the test set, and does not imply one-at-a-time damage predicts the joint
damage of deleting thousands of interacting edges.

## 11. Complete checkpoint and restart semantics

`TrainingSession.save_checkpoint` records all model parameters and binary
masks, Adam's first/second moments and each coordinate's bias-correction age,
Taylor and dormant-edge gradient EMAs, epoch number, full training curve,
random-number generator state and SHA-256 of every dataset split. Loading is
NumPy `allow_pickle=False` and checks shapes, dtype, finiteness and stale Adam
state on masked coordinates; a different dataset/split is rejected. This
format is separate from the inference-only `SparseMLP.save`. Unit tests compare
all arrays and final metrics between an uninterrupted run and a save/resume run
with regrowth enabled **at exactly an epoch boundary**. Changing planned total
epochs or the environment is outside that bit-exact guarantee.

## 12. Validation boundaries and scientific integrity

No post-hoc fabricated result is allowed. A negative Taylor-versus-magnitude
result is still a scientific result; the strongest submission is **correct and
reproducible**. This package includes automatic fixed-budget regrowth swaps, but makes no
claim of sparse latency improvement. Tests cover mathematical gradients and state invariants;
like any software, they cannot prove that all conceivable edge cases are absent.

### Reading and provenance

- Kingma & Ba, *Adam: A Method for Stochastic Optimization*:
  https://arxiv.org/abs/1412.6980
- He et al., *Delving Deep into Rectifiers*:
  https://arxiv.org/abs/1502.01852
- Molchanov et al., first-order Taylor pruning:
  https://arxiv.org/abs/1611.06440
- Zhu & Gupta, gradual pruning:
  https://arxiv.org/abs/1710.01878
- Evci et al., gradient-based sparse topology and regrowth:
  https://proceedings.mlr.press/v119/evci20a.html
- NumPy broadcasting:
  https://numpy.org/doc/stable/user/basics.broadcasting.html

These are reference ideas and standard mathematics, not dependencies used by
our core implementation.
