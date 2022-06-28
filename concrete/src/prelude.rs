//! The purpose of this module is to make it easier to have the most commonly needed
//! traits of this crate.
//!
//! It is meant to be glob imported:
//! ```
//! use concrete::prelude::*;
//! ```
pub use crate::traits::{
    DynamicFheEncryptor, DynamicFheTrivialEncryptor, DynamicFheTryEncryptor, FheBootstrap,
    FheDecrypt, FheEncrypt, FheNumberConstant, FheTrivialEncrypt, FheTryEncrypt,
};
