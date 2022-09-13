#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey};
use crate::global_state::WithGlobalKey;
use crate::keys::RefKeyFromKeyChain;
use crate::shortints::client_key::ShortIntegerClientKey;
use crate::shortints::server_key::ShortIntegerServerKey;
use crate::shortints::types::ShortIntegerParameter;
use crate::traits::DynamicFheTryEncryptor;
use crate::{ClientKey, GenericShortInt, OutOfRangeError};

#[cfg_attr(doc, cfg(feature = "shortints"))]
pub type DynShortIntParameters = concrete_shortint::parameters::Parameters;
/// Type dynamically defined shortint.
#[cfg_attr(doc, cfg(feature = "shortints"))]
pub type DynShortInt = GenericShortInt<DynShortIntParameters>;
#[cfg_attr(doc, cfg(feature = "shortints"))]
pub type DynShortIntServerKey = ShortIntegerServerKey<DynShortIntParameters>;
#[cfg_attr(doc, cfg(feature = "shortints"))]
pub type DynShortIntClientKey = ShortIntegerClientKey<DynShortIntParameters>;

/// The type id for dynamic shortints
///
/// The type id serves as the index in the Vec of dynamic shortint keys
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ShortIntTypeId(pub usize);

impl RefKeyFromKeyChain for ShortIntTypeId {
    type Key = DynShortIntClientKey;

    fn ref_key(self, keys: &ClientKey) -> Result<&Self::Key, UninitializedClientKey> {
        keys.shortint_key
            .dynamic_keys
            .get(self.0)
            .ok_or(UninitializedClientKey(Type::DynamicShortInt))
    }
}

impl WithGlobalKey for ShortIntTypeId {
    type Key = DynShortIntServerKey;

    fn with_global_mut<R, F>(self, func: F) -> Result<R, UninitializedServerKey>
    where
        F: FnOnce(&mut Self::Key) -> R,
    {
        crate::global_state::with_internal_keys_mut(|key| {
            key.shortint_key
                .dynamic_keys
                .get_mut(self.0)
                .map(func)
                .ok_or(UninitializedServerKey(Type::DynamicShortInt))
        })
    }
}

impl ShortIntegerParameter for DynShortIntParameters {
    type Id = ShortIntTypeId;
}

/// The encryptor for dynamic shortint
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "shortints"))]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct DynShortIntEncryptor {
    type_id: ShortIntTypeId,
}

impl From<ShortIntTypeId> for DynShortIntEncryptor {
    fn from(type_id: ShortIntTypeId) -> Self {
        Self { type_id }
    }
}

impl<T> DynamicFheTryEncryptor<T> for DynShortIntEncryptor
where
    T: TryInto<u8>,
{
    type FheType = DynShortInt;
    type Error = OutOfRangeError;

    #[track_caller]
    fn try_encrypt(&self, value: T, key: &ClientKey) -> Result<Self::FheType, Self::Error> {
        let client_key = self.type_id.unwrapped_ref_key(key);

        let value = value.try_into().map_err(|_err| OutOfRangeError)?;
        if value as usize >= client_key.key.parameters.message_modulus.0 {
            Err(OutOfRangeError)
        } else {
            let message = u64::from(value);
            let encrypted_int = client_key.key.encrypt(message);
            Ok(DynShortInt {
                ciphertext: std::cell::RefCell::new(encrypted_int),
                id: self.type_id,
            })
        }
    }
}
