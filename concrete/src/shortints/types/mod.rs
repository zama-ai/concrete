pub use base::GenericShortInt;
pub use dynamic::{DynShortInt, DynShortIntEncryptor, DynShortIntParameters};
pub(in crate::shortints) use r#dynamic::{
    DynShortIntClientKey, DynShortIntServerKey, ShortIntTypeId,
};
pub use r#static::{
    FheUint2, FheUint2Parameters, FheUint3, FheUint3Parameters, FheUint4, FheUint4Parameters,
};
pub(in crate::shortints) use r#static::{
    FheUint2ClientKey, FheUint2ServerKey, FheUint3ClientKey, FheUint3ServerKey, FheUint4ClientKey,
    FheUint4ServerKey,
};

use super::client_key::ShortIntegerClientKey;
use super::server_key::ShortIntegerServerKey;

mod base;
mod dynamic;
mod r#static;

pub trait ShortIntegerParameter: Copy + Into<concrete_shortint::Parameters> {
    type Id: Copy;
}

// Can it be made so that we are sure that both Id are the same
pub trait StaticShortIntegerParameter: ShortIntegerParameter {
    const MESSAGE_BITS: u8;
}

// Static types
