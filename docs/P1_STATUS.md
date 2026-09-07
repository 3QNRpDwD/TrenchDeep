# P1 implementation and comparison status

P1 is not yet declared complete. The implementation now has replaceable storage,
autograd and operation contracts, common training steps, real U-Net/DDPM, explicit
capture, independent RNG streams and retained comparisons. Full legacy equivalence
has an identified mathematical discrepancy and the comparison coverage below has
deliberate limits. No legacy source is deleted or numerically modified.

## Verified behavior

- Independent slot storage, vector tape and reference operations run the same
  public linear model/optimizer/trainer individually and together. The integration
  test also runs with all default implementations absent from the build.
- Missing storage, autograd, operations and capture return structured dependency
  errors; unsupported operations return capability errors. No-grad inference works
  without autograd. Detach aliasing and context lifetime are tested.
- Shared parameters are registered and clipped once. Initialization, model noise,
  data ordering and RL action streams are separate. Trainer seeding preserves weights.
- Loader setup/forward errors clean the training scope. A primary error and an
  independent cleanup error are both returned. Missing dataset samples return errors.
- Captures contain intermediate gradients before backward cleanup, remain owned
  after context destruction, and do not use a global capture session.
- Training publishes snapshots before batch completion, after graph/gradient
  cleanup. A failing metric hook suppresses both success events and cleans gradients.
- U-Net time embedding, residuals, attention, down/up paths, linear/cosine schedules,
  explicit-noise training/sampling, and checkpoint round-trip are exercised.
- RL uses shared step processing and no-grad policy forward. Interrupt checkpoints
  are saved after completed episode cleanup and retain the existing metadata format.
  Complete resume and optimizer snapshots explicitly remain unsupported in P1.

## Retained comparisons

| Fixture | Verified comparison | Limits |
|---|---|---|
| 16×16 matrix multiplication | Forward values, release timings | One shape; not broad operator coverage |
| Two-stage attention U-Net, `[1,1,4,4]` | Forward, MSE, every parameter gradient, actual SGD updates, release timings | Small CPU fixture, not a large-image performance claim |
| Linear Trainer, eight samples, batches of two | Loader-to-optimizer epoch loss and updated parameters, release timings | Fixed order, no I/O in timed region |
| Bigram | Last-step logits, shift loss, gradients, SGD update | Single sequence fixture; batched shift has separate production tests |
| Pi-model | Output, loss, gradients, SGD update with zero and nonzero noise | Nonzero views are read from the original graph and supplied unchanged to production |
| RL policy | Fixed-action loss, gradients, SGD update | Separate from rollout |
| RL rollout | Three actions, loss, Trainer update | One episode; action RNG seeds explicitly aligned across implementations |
| Legacy MLP/Sigmoid | Discrepancy reproduced and preserved | Performance comparison blocked |

### Legacy Sigmoid discrepancy

`legacy/src/nn/activation/sigmoid.rs` computes `1/(1+exp(x))` in forward. Its
backward uses the positive derivative of `1/(1+exp(-x))`. At `x=1`, baseline
forward is approximately `0.2689414`; production is approximately `0.7310586`.
This is not a tolerance issue. Production retains the mathematically correct
sigmoid, including finite-difference regression coverage. The original baseline
is untouched. `legacy_sigmoid_discrepancy_blocks_mlp_performance_comparison`
asserts the discrepancy rather than ignoring a failed test or widening tolerance.
MLP must not be labeled numerically equivalent or assigned parity-qualified timings.

The new MLP trains on logits with fused softmax cross entropy, while legacy MLP
returns probabilities. Prediction comparisons must first use the same representation;
doing so still exposes the sigmoid discrepancy above.

## Reproduction

From the repository root:

```powershell
python scripts/verify_legacy.py
cargo test --all-features --lib --tests
cargo test --all-features --doc
cargo test --no-default-features --test providers --test metrics --test cleanup
cargo check --no-default-features --features enableBackward
cargo check --no-default-features --features enableVisualization
cargo test --manifest-path legacy/Cargo.toml --lib --features enableBackward
cargo bench --bench p1_compare --features legacyBenchmark
```

The original baseline library remains **309 passing, zero failing, three original
ignored tests**. `legacy/BASELINE.json` verifies all 142 retained source files.
The current all-feature library/integration suite has 119 passing tests with no
failures or ignores in the final full run. Documentation tests pass 4/4;
the no-default provider/metrics/cleanup suite passes 13/13. Backward-only and
visualization-only no-default builds pass.
The additive comparison wrapper changes module visibility and macro provenance
only; details are in `legacy/README.md`. No deletion is part of P1 or a follow-up.

Benchmarks compare values before publishing timings. U-Net and Trainer cases
recheck parity throughout repeated updates; failures abort reporting. Tolerance is
absolute **or** relative `1e-3`, with no numerical exception currently permitted.
Initialization is outside timing, with combined fixture setup and separate model construction costs.
One warmup iteration is discarded. Reports contain median, nearest-rank p95,
throughput, all samples, and both implementations' storage/graph handle counts. U-Net timing
includes forward, backward and SGD; Trainer timing includes loader, validation,
backward, optimizer and cleanup. Correctness checks are outside the timed regions.
Counts are live handles, not allocator peak memory.

Results and environment/source metadata are kept under `docs/benchmarks`. These small fixtures show a performance
regression in the new path; they do not establish a general speedup. Owned snapshots
at provider boundaries currently copy inputs; this is an identifiable source of
overhead, not a measured attribution of the entire regression.

## Remaining acceptance work

- Resolve the MLP numerical-equivalence criterion without silently changing the
  preserved baseline or corrupting production sigmoid semantics.
- Broaden operation/model/Trainer benchmark coverage beyond the fixtures above.
- Complete the broader failure/observer-order matrix. Existing tests establish the listed
  cases, not every combination of user-supplied providers and callbacks.

These remaining items are visible acceptance gaps; passing the current test suite
does not by itself mean every original P1 acceptance condition is complete.
