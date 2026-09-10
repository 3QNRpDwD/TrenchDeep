# Internal and native regression tests

- `nn/`: extracted layer/model unit tests. Native reference model implementations
  live in `src/nn/native_models/`, exposed as `legacy::nn::models`.
- `trainer/`: extracted trainer unit tests, including private state and native
  API regression checks.
- `common/`: test data/logging helpers. `common/model/mod.rs` only re-exports the
  native models for compatibility; it contains no model implementation.

Internal tests are connected with `cfg(test)` from their owning modules so they
retain private access without making implementation details public. They must not
also be registered here, which would execute the same tests twice.

Public Context model/trainer E2E tests live in `tests/models/`, registered once by
`tests/models.rs`. Route parity and other public API contracts remain in `tests/`.
