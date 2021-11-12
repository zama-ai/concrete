//! A module containing the specification for the backends of the `concrete` FHE scheme.
//!
//! A backend is expected to provide access to two different families of objects:
//!
//! + __Entities__ which are FHE objects you can manipulate with the library (the data).
//! + __Engines__ which are types you can use to operate on entities (the operators).
//!
//! The specification contains traits for both entities and engines which are then implemented in
//! the backend modules.

pub mod engines;
pub mod entities;
