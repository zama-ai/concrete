//! # Tutorial
//! We provide some tutorials in the documentation of the tutorial module.

#[macro_use]
pub mod core_api;
pub mod guide;
pub mod npe;
#[macro_use]
pub mod crypto_api;
pub mod traits;
pub mod types;

pub use types::Types;

#[macro_use]
extern crate itertools;
extern crate kolmogorov_smirnov;

// for the rust lib
pub use crypto_api::*;
