pub use base::{if_then_else, GenericBool};
pub use dynamic::{DynFheBool, DynFheBoolEncryptor};
pub use r#static::FheBool;
pub(super) use r#static::{FheBoolClientKey, FheBoolServerKey, StaticBoolParameters};

mod base;
pub mod dynamic;
pub mod r#static;
