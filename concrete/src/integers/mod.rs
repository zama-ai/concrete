mod dynamic;
mod keys;
mod r#static;

pub(crate) use keys::{IntegerClientKey, IntegerConfig, IntegerServerKey};

pub use dynamic::{DynInteger, DynIntegerEncryptor, DynIntegerParameters};
pub use r#static::{FheUint12, FheUint16, FheUint8, GenericInteger};
