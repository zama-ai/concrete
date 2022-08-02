#![allow(unused_doc_comments)]
#![cfg_attr(fmt, rustfmt::skip)]
#![cfg_attr(doc, feature(doc_auto_cfg))]
#![cfg_attr(doc, feature(doc_cfg))]
#![allow(clippy::bool_assert_comparison)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::eq_op)]
#![allow(clippy::assign_op_pattern)]


//! Welcome to concrete documentation
//!
//! The goal of this crate is to make it as easy as possible to write FHE programs
//! by abstracting away most of the FHE details and by providing data types which are as close as
//! possible to the native ones (`bool`, `u8`, `u16`) that we are used to.
//!
//! # Cargo Features
//!
//! ## Data Types
//!
//! This crate exposes 3 kinds of data types, each kind is enabled by activating its corresponding
//! feature. Each kind may have multiple types:
//!
//! | Kind      | Cargo Feature | Type(s)                                  |
//! |-----------|---------------|------------------------------------------|
//! | Booleans  | `booleans`    | [FheBool]                                |
//! | ShortInts | `shortints`   | [FheUint2]<br>[FheUint3]<br>[FheUint4]   |
//! | Integers  | `integers`    | [FheUint8]<br>[FheUint12]<br>[FheUint16] |
//!
//!
//!
//! ## Dynamic types
//!
//! To allow further customization, it is possible to create at runtime new data types
//! which are based on a certain kind.
//!
//! For example it is possible to create your own 10 bits integer data type.
//!
//! ### Example
//!
//! Creating a 10-bits integer by combining 5 2-bits shortints
//! ```rust
//! // This requires the integers feature
//! #[cfg(feature = "integers")]
//! {
//!     use concrete::prelude::*;
//!     use concrete::{
//!         generate_keys, set_server_key, ConfigBuilder, DynIntegerParameters, FheUint2Parameters,
//!     };
//!
//!     let mut config = ConfigBuilder::all_disabled();
//!     let uint10_type = config.add_integer_type(DynIntegerParameters {
//!         block_parameters: FheUint2Parameters::default().into(),
//!         num_block: 5,
//!     });
//!
//!     let (client_key, server_key) = generate_keys(config);
//!
//!     set_server_key(server_key);
//!
//!     let a = uint10_type.encrypt(177, &client_key);
//!     let b = uint10_type.encrypt(100, &client_key);
//!
//!     let c: u64 = (a + b).decrypt(&client_key);
//!     assert_eq!(c, 277);
//! }
//! ```
//!
//! ## Serialization
//!
//! Most of the data types are `Serializable` and `Deserializable` via the serde crate
//! and its corresponding feature: `serde`.
//!
//!
//! ## Activating features
//!
//! To activate features, in your `Cargo.toml` add:
//! ```toml
//! concrete = { version = "0.2.0-beta", features = ["booleans", "serde"] }
//! ```
pub use config::{ConfigBuilder, Config};
pub use global_state::set_server_key;
pub use keys::{generate_keys, ClientKey, ServerKey};
pub use errors::OutOfRangeError;

#[cfg(feature = "serde")]
pub use keycache::KeyCacher;

#[cfg(feature = "booleans")]
pub use crate::booleans::{DynFheBool, DynFheBoolEncryptor};

#[cfg(feature = "booleans")]
pub use booleans::{if_then_else, FheBool, FheBoolParameters};

// GenericShortInt is exported only to produce better docs
#[cfg(feature = "shortints")]
pub use crate::shortints::{
    FheUint2, FheUint2Parameters, FheUint3, FheUint3Parameters, FheUint4, FheUint4Parameters,
    GenericShortInt,
};

#[cfg(feature = "integers")]
pub use crate::integers::{
    DynInteger, DynIntegerEncryptor, DynIntegerParameters,
    FheUint8, FheUint12, FheUint16, GenericInteger,
};

#[cfg(feature = "shortints")]
pub use crate::shortints::{DynShortInt, DynShortIntEncryptor, DynShortIntParameters};


#[macro_use]
mod global_state;
#[macro_use]
mod keys;
mod config;
mod traits;
/// The concrete prelude.
pub mod prelude;
pub mod errors;
#[cfg(feature = "serde")]
mod keycache;
#[cfg(feature = "booleans")]
mod booleans;
#[cfg(feature = "shortints")]
mod shortints;
#[cfg(feature = "integers")]
mod integers;
#[cfg(feature = "experimental_syntax_sugar")]
mod syntax_sugar;

pub mod parameters {
    #[cfg(all(feature = "booleans", feature = "__newer_booleans"))]
    pub use concrete_boolean::parameters::{
        DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
        StandardDev,
    };
    #[cfg(feature = "shortints")]
    pub use concrete_shortint::parameters::{CarryModulus, MessageModulus};
    #[cfg(all(feature = "shortints", not(all(feature = "booleans", feature = "__newer_booleans"))))]
    pub use concrete_shortint::parameters::{
        DecompositionBaseLog, DecompositionLevelCount, DispersionParameter, GlweDimension,
        LweDimension, PolynomialSize, StandardDev,
    };
}

#[cfg(all(
doctest,
feature = "integers",
feature = "shortints",
feature = "booleans"
))]
mod user_doc_tests;
