pub use base::GenericInteger;
pub use dynamic::{DynInteger, DynIntegerEncryptor, DynIntegerParameters};
pub use r#static::{FheUint12, FheUint16, FheUint8};

pub(super) mod base;
pub(super) mod dynamic;
pub(super) mod r#static;
