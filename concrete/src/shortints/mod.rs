mod dynamic;
mod keys;
mod r#static;

pub(crate) use keys::{ShortIntClientKey, ShortIntConfig, ShortIntServerKey};

pub use dynamic::{DynShortInt, DynShortIntEncryptor, DynShortIntParameters};

pub use r#static::{
    FheUint2, FheUint2Parameters, FheUint3, FheUint3Parameters, FheUint4, FheUint4Parameters,
    GenericShortInt,
};

#[cfg(feature = "integers")]
pub(crate) use r#static::ShortIntegerParameter;
