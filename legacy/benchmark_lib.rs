#![allow(macro_expanded_macro_exports_accessed_by_absolute_paths)]
// include! changes macro expansion provenance, not baseline implementation semantics.
// Keep the baseline source byte-for-byte intact. Only additive benchmark access lives here.
include!("src/lib.rs");
#[cfg(not(test))]
pub mod comparison;
