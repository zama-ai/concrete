//! A module containing various backends implementing the `concrete` FHE scheme.
//!
//! This module contains all the backends implementing the concrete specification. As of now we
//! support the following backend:
//!
//! + `core` : A single threaded CPU backend geared towards x86_64 architectures.

#[cfg(feature = "backend_core")]
pub mod core;

#[cfg(feature = "backend_optalysys")]
pub mod optalysys;
