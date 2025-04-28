pub use cxx::{SharedPtr, UniquePtr};
pub use ffi::c_void;

mod ffi;
#[cfg(feature = "tfhe-rs")]
pub mod tfhe;

#[cfg(feature = "compiler")]
#[doc(hidden)]
pub mod compiler {
    pub use crate::ffi::{compile, CompilationOptions, Library};
}

pub mod common {
    pub use crate::ffi::{
        ClientKeyset, EncryptionCsprng, Keyset, LweBootstrapKey, LweKeyswitchKey, LweSecretKey,
        PackingKeyswitchKey, SecretCsprng, ServerKeyset, Tensor, TransportValue, Value,
    };
}

pub mod client {
    pub use crate::ffi::{ClientFunction, ClientModule};
}

pub mod server {
    pub use crate::ffi::ServerFunction;
}

pub mod protocol;

#[doc(hidden)]
pub mod utils;
