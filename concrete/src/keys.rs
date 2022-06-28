//! This module defines the two aggregate key types
//!
//! - [ServerKey] aggregates the server keys of each types. (Which in turn will contain things like
//!   bootstrapping keys)
//! - [ClientKey] aggregates the keys used to encrypt/decrypt between normal and homomorphic types.

use crate::config::Config;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "booleans")]
use crate::booleans::{BoolClientKey, BoolServerKey};
use crate::errors::{UninitializedClientKey, UnwrapResultExt};
#[cfg(feature = "integers")]
use crate::integers::{IntegerClientKey, IntegerServerKey};
#[cfg(feature = "shortints")]
use crate::shortints::{ShortIntClientKey, ShortIntServerKey};

/// Generates keys using the provided config.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "shortints")]
/// # {
/// use concrete::{generate_keys, ConfigBuilder};
///
/// let config = ConfigBuilder::all_disabled().enable_default_uint3().build();
/// let (client_key, server_key) = generate_keys(config);
/// # }
/// ```
pub fn generate_keys<C: Into<Config>>(config: C) -> (ClientKey, ServerKey) {
    let client_kc = ClientKey::generate(config);
    let server_kc = client_kc.generate_server_key();

    (client_kc, server_kc)
}

/// Key of the client
///
/// This struct contains the keys that are of interest to the user
/// as they will allow to encrypt and decrypt data.
///
/// This key **MUST NOT** be sent to the server.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct ClientKey {
    #[cfg(feature = "booleans")]
    pub(crate) bool_key: BoolClientKey,
    #[cfg(feature = "shortints")]
    pub(crate) shortint_key: ShortIntClientKey,
    #[cfg(feature = "integers")]
    pub(crate) integer_key: IntegerClientKey,
}

impl ClientKey {
    /// Generates randomly a ne keys.
    pub fn generate<C: Into<Config>>(config: C) -> ClientKey {
        #[allow(unused_variables)]
        let config: Config = config.into();
        ClientKey {
            #[cfg(feature = "booleans")]
            bool_key: BoolClientKey::from(config.bool_config),
            #[cfg(feature = "shortints")]
            shortint_key: ShortIntClientKey::from(config.shortint_config),
            #[cfg(feature = "integers")]
            integer_key: IntegerClientKey::from(config.integer_config),
        }
    }

    /// Generates a new ServerKeyChain
    ///
    /// The `ServerKeyChain` generated is meant to be used to initialize the global state
    /// using [crate::set_server_key].
    pub fn generate_server_key(&self) -> ServerKey {
        ServerKey::new(self)
    }
}

/// Trait to be implemented on the client key types that have a corresponding member
/// in the `ClientKeyChain`.
///
/// This is to allow the writing of generic functions.
pub trait RefKeyFromKeyChain {
    /// The method to implement, shall return a ref to the key or an error if
    /// the key member in the key was not initialized
    fn ref_key(keys: &ClientKey) -> Result<&Self, UninitializedClientKey>;

    /// Returns a mutable ref to the key member of the key
    ///
    /// # Panic
    ///
    /// This will panic if the key was not initialized
    #[track_caller]
    fn unwrapped_ref_key(keys: &ClientKey) -> &Self {
        Self::ref_key(keys).unwrap_display()
    }
}

/// Helper macro to help reduce boiler plate
/// needed to implement `RefKeyFromKeyChain` since for
/// our keys, the implementation is the same, only a few things change.
///
/// It expects:
/// - The  `name` of the key type for which the trait will be implemented.
/// - The identifier (or identifier chain) that points to the member in the `ClientKey` that hols
///   the key for which the trait is implemented.
/// - Type Variant used to identify the type at runtime (see `error.rs`)
#[cfg(any(feature = "shortints", feature = "integers"))]
macro_rules! impl_ref_key_from_keychain {
    (
        for $key_type:ty {
            keychain_member: $($member:ident).*,
            type_variant: $enum_variant:expr,
        }
    ) => {
        impl crate::keys::RefKeyFromKeyChain for $key_type {
            fn ref_key(keys: &ClientKey) -> Result<&Self, UninitializedClientKey> {
                keys$(.$member)*
                    .as_ref()
                    .ok_or(UninitializedClientKey($enum_variant))
            }
        }
    }
}

/// Key of the server
///
/// This key contains the different keys needed to be able to do computations for
/// each data type.
///
/// For a server to be able to do some FHE computations, the client needs to send this key
/// beforehand.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Default)]
pub struct ServerKey {
    #[cfg(feature = "booleans")]
    pub(crate) bool_key: BoolServerKey,
    #[cfg(feature = "shortints")]
    pub(crate) shortint_key: ShortIntServerKey,
    #[cfg(feature = "integers")]
    pub(crate) integer_key: IntegerServerKey,
}

impl ServerKey {
    #[allow(unused_variables)]
    pub(crate) fn new(keys: &ClientKey) -> Self {
        Self {
            #[cfg(feature = "booleans")]
            bool_key: BoolServerKey::new(&keys.bool_key),
            #[cfg(feature = "shortints")]
            shortint_key: ShortIntServerKey::new(&keys.shortint_key),
            #[cfg(feature = "integers")]
            integer_key: IntegerServerKey::new(&keys.integer_key),
        }
    }
}
