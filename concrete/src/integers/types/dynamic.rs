#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey};
use crate::global_state::WithGlobalKey;
use crate::integers::client_key::GenericIntegerClientKey;
use crate::integers::parameters::IntegerParameter;
use crate::integers::server_key::GenericIntegerServerKey;
use crate::keys::RefKeyFromKeyChain;
use crate::traits::DynamicFheEncryptor;
use crate::{ClientKey, GenericInteger, IntegerParameterSet};

pub type DynIntegerParameters = IntegerParameterSet;
pub type DynInteger = GenericInteger<DynIntegerParameters>;
pub type DynIntegerServerKey = GenericIntegerServerKey<DynIntegerParameters>;
pub type DynIntegerClientKey = GenericIntegerClientKey<DynIntegerParameters>;

/// Id the allows to retrieve the key for the dynamic integer
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IntegerTypeId(pub(in crate::integers) usize);

impl RefKeyFromKeyChain for IntegerTypeId {
    type Key = DynIntegerClientKey;

    fn ref_key(self, keys: &ClientKey) -> Result<&Self::Key, UninitializedClientKey> {
        keys.integer_key
            .custom_keys
            .get(self.0)
            .ok_or(UninitializedClientKey(Type::DynamicInteger))
    }
}

impl WithGlobalKey for IntegerTypeId {
    type Key = DynIntegerServerKey;

    fn with_global_mut<R, F>(self, func: F) -> Result<R, UninitializedServerKey>
    where
        F: FnOnce(&mut Self::Key) -> R,
    {
        crate::global_state::with_internal_keys_mut(|key| {
            key.integer_key
                .custom_keys
                .get_mut(self.0)
                .map(func)
                .ok_or(UninitializedServerKey(Type::DynamicInteger))
        })
    }
}

impl IntegerParameter for DynIntegerParameters {
    type Id = IntegerTypeId;
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "integers"))]
#[derive(Clone)]
pub struct DynIntegerEncryptor {
    type_id: IntegerTypeId,
}

impl From<IntegerTypeId> for DynIntegerEncryptor {
    fn from(type_id: IntegerTypeId) -> Self {
        Self { type_id }
    }
}

impl DynamicFheEncryptor<u64> for DynIntegerEncryptor {
    type FheType = DynInteger;

    #[track_caller]
    fn encrypt(&self, value: u64, key: &ClientKey) -> Self::FheType {
        let key = self.type_id.unwrapped_ref_key(key);
        let ciphertext = key.key.encrypt(value);
        DynInteger::new(ciphertext, self.type_id)
    }
}
