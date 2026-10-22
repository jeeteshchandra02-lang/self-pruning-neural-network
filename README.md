# AQUA — Self-Pruning Network

[Source code](engine/tensor.py) · [Design and derivations](DESIGN.md) · [Bug traps](docs/BUG_TRAPS.md) · [Live defense prep](docs/DEFENSE.md)

A **from-scratch Python + NumPy** reverse-mode autodiff engine, MLP classifier,
mask-aware Adam optimizer, gradual importance-based pruning, optional
budget-preserving regrowth, epoch-resumable training, and an inference
implementation that visits only nonzero connections. This repository is a
runnable, studyable implementation of the supplied **AQUA selection exercise**.
It includes real measured results—not fabricated target numbers.

**Important:** The assignment explicitly permits AI assistance, but applicants
must understand, verify, and be able to defend every statement and line before
submitting. This is an implementation package, not evidence that the candidate
personally coded every part. All claims are tied to reproducible experiment files.

## Requirements and dependency boundaries

- Python **3.10+** (tested with Python 3.13.5)
- **NumPy** for all tensor mathematics, autodiff, optimizers, training, pruning,
  and sparse inference
- **scikit-learn is used exclusively for `sklearn.datasets.load_digits`**, to
  load the standard digits dataset. It is **not** used for train/test splitting,
  preprocessing, models, gradients, optimization, training, metrics, or pruning.
- Matplotlib is used **only to generate plots**.
- No PyTorch, TensorFlow, JAX, Keras, Hugging Face, external autodiff,
  neural-network, optimizer, or pruning libraries.
- The tests use Python's built-in `unittest`, so there is **no pytest dependency**.

Install from the **repository root**:

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternative `uv` install: `uv venv && uv pip install -r requirements.txt`.

## Commands to reproduce all four parts

```bash
# Part 1: every gradient check, masked gradient, Adam revival, and integration test
python -m unittest discover -s tests -v

# Part 2: dense model; saves learning curves and plot
python -m train.dense --seed 0 --epochs 32

# Part 3: progressive Taylor-pruned model; exports a sparse .npz checkpoint
python -m train.sparse --criterion taylor --sparsity 0.90 --seed 0 --epochs 32

# Part 4: Pareto sweep, both pruning methods, five PAIRED seeds
python -m train.sweep --sparsities 0 0.50 0.75 0.90 0.95 --criteria taylor magnitude --seeds 0 1 2 3 4 --epochs 32 --benchmark-repeats 16

# Inference using the exported sparse-only model (no training graph)
python -m train.infer --index 0

# Bonus: demonstrate 4 evidence-guided, fixed-budget edge exchanges per epoch
python -m train.sparse --sparsity 0.90 --regrow-swaps 4 --seed 0 --epochs 32

# Bonus: measure REAL loss changes after deleting 80 sampled live edges
python -m train.audit --seed 0 --epochs 32 --sparsity 0.90 --samples 80

# Bonus: save a COMPLETE epoch-boundary training checkpoint and resume
python -m train.checkpoint --epochs 16 --stop-after 8 --checkpoint artifacts/training_state.npz
python -m train.checkpoint --resume --checkpoint artifacts/training_state.npz
```

On Linux/macOS, to more closely reproduce the supplied single-threaded CPU
experiment, prefix the commands with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
On Windows PowerShell, set `$env:OPENBLAS_NUM_THREADS="1"` and
`$env:OMP_NUM_THREADS="1"` in the shell before running. BLAS versions, thread
counts, hardware and operating systems influence timings (and may affect
small floating-point differences).

### What the commands write

| File | Purpose |
|---|---|
| `artifacts/raw/dense_learning_curve.csv` | Epoch loss and accuracy, seed 0 |
| `artifacts/raw/pruned_learning_curve.csv` | Epoch loss, accuracy and sparsity, 90% Taylor, seed 0 |
| `artifacts/raw/sweep_results.csv` | Every combination of method, sparsity, and seed; includes actual test accuracy and timing |
| `artifacts/raw/sweep_summary.csv` | Cross-seed mean and standard deviation |
| `artifacts/raw/experiment_config.json` | Seeds, optimizer configuration, environment and operation definitions |
| `artifacts/raw/falsifiable_claim.txt` | Quantitative claim generated from raw rows |
| `artifacts/plots/dense_learning_curve.png` | Dense learning plot |
| `artifacts/plots/pruned_learning_curve.png` | Progressive-pruning learning plot |
| `artifacts/plots/pareto_accuracy.png` | Accuracy against achieved sparsity with error bars |
| `artifacts/sparse_model.npz` | Exported **sparse-only inference** checkpoint |
| `artifacts/raw/saliency_audit.csv` | Single-edge measured validation-loss change versus signed Taylor estimate |
| `artifacts/plots/saliency_vs_loss.png` | Taylor prediction vs actual loss change scatterplot |
| `artifacts/raw/regrowth_demo.json` | Separate 90% sparsity fixed-budget regrowth demonstration |
| `artifacts/training_state.npz` (generated; not committed) | Full weights/masks/optimizer/saliency/RNG/curve checkpoint for pausing and resuming |

Running the single-model CLIs overwrites their corresponding demonstration
learning-curve outputs; rerun the full sweep to restore the committed sweep
figures and seed-0 curves. Do not mix artifacts from different experiment runs.

## What the system does, end to end

1. `data/digits.py` loads the 1,797-row / 64-feature / 10-class digits dataset.
   All train/validation/test indexes are assigned by our own seeded stratified
   split (70%/15%/15% approximately per class). Pixel values are divided by 16,
   a known fixed value; no test-set statistics are used for normalization.
2. `nn/layers.py` builds a `64 → 96 → 48 → 10` MLP with ReLU hidden layers and
   He-style normal initialization (smaller final-layer variance).
3. `engine/tensor.py` creates a fresh computation DAG per mini-batch; its own
   reverse-mode `backward()` computes parameter gradients, including summed
   gradients across broadcast dimensions.
4. `engine.cross_entropy` computes stable log-sum-exp cross entropy and its
   analytically derived gradient, without overflow-prone naive softmax/log.
5. `optim/adam.py` updates active weights with separate first/second moments;
   **masked entries are exactly zero**, and their optimizer history is reset.
6. `prune/pruner.py` accumulates minibatch EMA Taylor importance `abs(weight *
   gradient)`, or computes the comparison score `abs(weight)`; an exact-k global
   pruning event applies the **cubic sparsity schedule** after each epoch.
   It keeps at least one input edge per output neuron and at least one outgoing
   edge per hidden neuron when those constraints are feasible. Optional
   **regrow_swap** selects dormant connections using `|dL/d(W_effective)|`,
   removes equally many live low-importance edges and initializes revived
   coordinates/Adam state cleanly. This never increases live-edge count.
7. Inference exports to `prune/sparse.py`: for every output neuron, only
   surviving input indices and their values are stored. This sparse-only
   representation is verified against the masked-dense model before timing.
8. `train/sweep.py` runs matched random seeds, compares both methods at the
   **same** budget and training settings, saves the complete numeric evidence,
   and draws the Pareto chart.

## Empirical outcomes

See the **committed CSVs**, charts and generated claim for exact numbers. A
nominal 90% mask target prunes **10,109 out of 11,232** weight connections,
leaving **1,123** active weight MACs per example (versus 11,232 dense). This
counts **weight-multiplication work only**, not biases, activation costs, index
handling or total system FLOPs. Biases are **never pruned** and are excluded
from the sparsity denominator.

**Critically, this simple Python/NumPy sparse path can be slower in elapsed
milliseconds than the dense BLAS-backed implementation**, despite performing
far fewer arithmetic weight operations. The actual latency values are measured
and committed, including batch sizes 1 and 32. Do **not** claim a production
latency speedup from this microbenchmark or generalize to GPUs.

Taylor importance **is not guaranteed** to outperform magnitude importance.
Treat the paired-seed comparison as a falsifiable hypothesis, not a sales claim.

## Exact gradients and masks (interview-critical)

Every masked layer implements `W_eff = mask * W`, with a copied constant mask
Tensor used in that forward graph. Therefore:

```
dL/dW = mask * (dL/dW_eff)
```

A removed weight **must have zero parameter gradient**. Its optimizer first
moment, second moment, and active-update age are reset to zero on pruning and
on manual revival. The optimizer checks that masked gradients are zero **before**
performing any protective state reset. Changing masks or stored parameters between forward and backward is
**actively rejected**. The model prevents pruning, restoring checkpoints or
optimizer steps while a forward graph is in flight; backward checks for direct
out-of-band mask/weight mutations *before* touching gradients. Differentiable
`Linear` calls require a matching backward; for evaluation without backward,
use `model.predict_logits()` / `layer.dense_logits()`.

`GlobalPruner.regrow_swap(k)` implements **optional gradient-guided regrowth**:
we record counterfactual effective-weight gradients separately from actual
masked parameter gradients; swap exactly `k` or fewer available dormant edges
with live edges without exceeding the hard budget. `--regrow-swaps 4` provides
a reproducible demonstration (separate from the Part 4 baseline sweep).
`GlobalPruner.revive()` is a manual API that may change sparsity: **do not use
it as a budget-preserving scheduler**.

## Repository layout

```text
engine/            Tensor, fused stable loss, gradient propagation
nn/                Linear, ReLU, MLP, binary masks
optim/             Adam and topology-safe per-coordinate state
prune/             exact-k pruning, Taylor score, cubic schedule, sparse model
data/              permitted digits loading and handwritten split
train/             dense, sparse, sweep commands + deterministic runner
tests/             numerical gradient checks and adversarial invariants
artifacts/raw/     actual experimental data (CSV / JSON / measured latency)
artifacts/plots/   generated curves and Pareto chart
artifacts/         exported sparse inference-only model
DESIGN.md          mathematical derivations and deployment discussion
docs/BUG_TRAPS.md  specific failure modes and unit tests
docs/DEFENSE.md    likely live cross-questions, math and coding exercises
```

## How to push to GitHub

This ZIP includes a `.git` folder with incremental construction commits made
by the **project-building assistant**. Those commits are project provenance,
not evidence of personal human authorship. Keep the provided history (the
exercise specifically says **do not squash commits**), review and test the code,
and add **your own** commits for any modifications before submitting.

```bash
# Unzip the archive, enter the aqua_self_pruning_network directory
# Review actual history:
git log --oneline

# Configure YOUR identity for FUTURE commits:
git config user.name "YOUR NAME"
git config user.email "YOUR_EMAIL@example.com"

# Create an empty repository on GitHub; substitute its URL below.
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

GitHub credentials or collaboration access must be handled on your account;
this package does not push to an external repository.

## Hardening changes in this version

The **38 passing tests** include input-gradient finite differences for masked
weights, rejected mid-graph topology/parameter mutation, atomic Adam gradient
preflight, disconnected-neuron prevention plus infeasible-budget rejection,
a counterfactual-gradient regrowth exchange, and **bit-exact full checkpoint
resume** with Adam, pruning EMA, RNG, mask and curves. The original tests remain.
This package does not assert access to any secret AQUA tests: these are our
adversarial deductions from the assignment.

`artifacts/raw/falsifiable_claim.txt` is regenerated from the modified
five-seed sweep; older values from the previous ZIP should **not** be quoted.
The optional regrowth demonstration is not included in the fair Taylor-versus-
magnitude Pareto comparison (both compared methods have regrowth disabled).

## Known boundaries

- Implemented `matmul` is **strictly 2-D**, which suffices for the MLP.
- All gradients and weights use float64 to simplify validation.
- The micro autodiff engine is for educational/small workloads, not a general
  automatic differentiation framework or a high-throughput service.
- Unstructured pruning reduces **active operation count**, not necessarily
  memory footprint or wall-clock latency on conventional hardware.
- Automatic budget-preserving gradient-guided edge swaps are implemented and
  separately tested, but are not claimed to be the full RigL training algorithm.
- The Python sparse inference path genuinely visits surviving weights but is
  **not** a hardware-optimized sparse inference kernel. No production latency
  speedup is claimed. Structured sparse formats/compiled kernels would require
  additional target-hardware work.
- Checkpoint resume is guaranteed/tested at **epoch boundaries** with an
  unchanged planned total epoch count, fixed dataset and identical environment;
  not at an arbitrary interrupted minibatch.
- Results on digits are small-task evidence; performance may differ on real
  large language models or on other accelerators.

The project intentionally prioritizes verifiable correctness and clear
limitations over an unsubstantiated benchmark claim.
