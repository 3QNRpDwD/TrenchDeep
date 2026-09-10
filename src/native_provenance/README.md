# Native source integration provenance

The historical package has been removed. Native execution compiles directly from
root `src/` under the `crate::legacy` namespace; P1 remains the default route.

- BASELINE.json: original source hashes and commit.
- CORRECTIONS.json: authorized numerical fixes before integration.
- INTEGRATED.json: source paths/hashes after namespace integration, with explicit
  removal entries for superseded pre-P1 Context implementations and per-file
  change notes for subsequent native additions (Abs/Log/Sqrt backward).

Run `python scripts/verify_legacy.py` from the repository root. Changes to mapped
sources require reviewing and updating their integration hashes; historical hashes
must not be rewritten to pretend that integrated code is byte-identical.

Integration changes include crate namespace/module paths, direct read-only named
parameter accessors, shared reference-model modules, and shared identical helpers.
TensorHandle Drop no longer logs during thread-local teardown. The native DOT
regression fixture now distinguishes vector inputs from hidden scalar inputs.

Native model implementations now live under `src/nn/native_models` and are
exported by `legacy::nn::models`; the old test-model path only re-exports them.
Model/layer and trainer unit tests live under `src/tests/nn` and
`src/tests/trainer`, connected by their original owning modules for private access.
Convergence is shared with the Context trainer, including a single test suite.
`INTEGRATED.json` records these relocations; historical baseline hashes remain intact.
