//! A module containing backends correctness tests.
//!
//! Each submodule here is expected to be activated by a given feature flag (matching the
//! `backend_*` naming), and to contain the instantiation of a generic correctness test for every
//! implemented operator.

#[cfg(all(test, feature = "backend_core"))]
pub mod core;
