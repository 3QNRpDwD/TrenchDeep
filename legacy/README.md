# Preserved comparison engine

`src/` is the byte-for-byte source snapshot of commit `65b4d40` (142 files).
This source is retained for numerical and performance comparisons; deletion belongs
to the user. `BASELINE.json` records its original hashes. Run
`python scripts/verify_legacy.py` from the repository root to verify preservation.

The package name differs from production. `benchmark_lib.rs` includes the unchanged
baseline and adds `comparison.rs`, a visibility bridge for the original models.
The wrapper allows Rust's include-generated macro-path lint because `include!`
changes macro provenance. No calculation, initializer, backward rule, or trainer
loop in the baseline is edited. Unit tests still compile the original test module.
Production loads this package only through the optional `legacyBenchmark` feature.

Known discrepancy: baseline Sigmoid forward computes `1/(1+exp(x))`, whereas its
backward uses a positive sigmoid derivative. MLP performance comparison is blocked;
the baseline is preserved, and production retains the correct sigmoid definition.
See `docs/P1_STATUS.md` for coverage and benchmark limits.
# Reference DDPM bridge update

The user subsequently authorized correcting the original Conv2D input-gradient
index directly. `BASELINE.json` remains the historical snapshot manifest;
`CORRECTIONS.json` records the corrected source hash and reason. The verification
script checks both. Reference DDPM gradient/Adam parity now passes with that fix.
Statements about byte-for-byte retention below apply except for this recorded fix.

`reference_models.rs` makes the original common model modules available at their
existing crate paths outside unit tests. No original source is edited.
`comparison::diffusion_draw` reads the original DDPM noise and signal multiplier
after forward, recovering its exact timestep without changing random generation.
The new reference comparison uncovered a Conv2D input-gradient indexing discrepancy
(see `docs/P1_STATUS.md` in the parent project). The original remains preserved.
# Named reference parameter access

The comparison build now uses `build.rs` to copy the preserved source tree into
`OUT_DIR/reference_src` and append the read-only accessors in `named_parameters/`.
No original function body is replaced. Standalone legacy unit tests still include
`src/lib.rs` directly. The generated model keeps the original crate/module paths.
Do not edit generated files; Cargo regenerates them when source/accessors change.

U-Net paths are traversed through actual fields and Sequential child layers.
Leaf parameter field names come from the original layer macro's `save_state`,
paired with that same macro's `params` order. This local leaf contract remains;
whole-model legacy/Context matching uses names, shapes and sharing identities,
not vector positions. Leaf metadata currently uses owned checkpoint snapshots;
enumeration is outside the numerical execution/benchmark region.
