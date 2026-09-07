#![allow(macro_expanded_macro_exports_accessed_by_absolute_paths)]
// include! changes macro expansion provenance, not baseline implementation semantics.
// Keep the baseline source byte-for-byte intact. Only additive benchmark access lives here.
include!("src/lib.rs");
#[cfg(not(test))]
pub mod comparison;
// Original reference models retain their crate::tests::common paths.
// Only this additive visibility shim is compiled outside baseline unit tests.
#[cfg(not(test))]
#[path = "reference_models.rs"]
pub mod tests;
