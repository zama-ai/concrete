use concrete_boolean::parameters::BooleanParameters;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::booleans::client_key::GenericBoolClientKey;
use crate::booleans::parameters::BooleanParameterSet;
use crate::booleans::server_key::GenericBoolServerKey;
use crate::errors::{Type, UninitializedClientKey, UninitializedServerKey};
use crate::global_state::WithGlobalKey;
use crate::keys::RefKeyFromKeyChain;
use crate::traits::DynamicFheEncryptor;
use crate::{ClientKey, FheBoolParameters};

use super::GenericBool;

/// Parameters for dynamically defined booleans
// It is a simple wrapper around the BooleanParameters
// as we just want to tie it to an Id type.
#[cfg_attr(doc, cfg(feature = "booleans"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct DynFheBoolParameters(pub(crate) FheBoolParameters);

impl From<DynFheBoolParameters> for BooleanParameters {
    fn from(p: DynFheBoolParameters) -> Self {
        p.0.into()
    }
}

impl From<FheBoolParameters> for DynFheBoolParameters {
    fn from(p: FheBoolParameters) -> Self {
        Self(p)
    }
}

impl BooleanParameterSet for DynFheBoolParameters {
    type Id = BooleanTypeId;
}

pub type DynFheBool = GenericBool<DynFheBoolParameters>;
pub type DynFheBoolClientKey = GenericBoolClientKey<DynFheBoolParameters>;
pub type DynFheBoolServerKey = GenericBoolServerKey<DynFheBoolParameters>;

/// Type Id for dynamically defined booleans
// It acts as an index in the vec of dynamic bool keys
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BooleanTypeId(pub(in crate::booleans) usize);

impl RefKeyFromKeyChain for BooleanTypeId {
    type Key = DynFheBoolClientKey;

    fn ref_key(self, keys: &ClientKey) -> Result<&Self::Key, UninitializedClientKey> {
        keys.bool_key
            .dynamic_keys
            .get(self.0)
            .ok_or(UninitializedClientKey(Type::DynamicBool))
    }
}

impl WithGlobalKey for BooleanTypeId {
    type Key = DynFheBoolServerKey;

    fn with_global_mut<R, F>(self, func: F) -> Result<R, UninitializedServerKey>
    where
        F: FnOnce(&mut Self::Key) -> R,
    {
        crate::global_state::with_internal_keys_mut(|key| {
            key.bool_key
                .dynamic_keys
                .get_mut(self.0)
                .map(func)
                .ok_or(UninitializedServerKey(Type::DynamicBool))
        })
    }
}

/// struct to create new values of a dynamically defined type of boolean
///
/// You get one of these by using `ConfigBuilder`'s [add_bool_type]
///
/// [add_bool_type]: crate::ConfigBuilder::add_bool_type
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(doc, cfg(feature = "booleans"))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DynFheBoolEncryptor {
    type_id: BooleanTypeId,
}

impl From<BooleanTypeId> for DynFheBoolEncryptor {
    fn from(type_id: BooleanTypeId) -> Self {
        Self { type_id }
    }
}

impl DynamicFheEncryptor<bool> for DynFheBoolEncryptor {
    type FheType = DynFheBool;

    #[track_caller]
    fn encrypt(&self, value: bool, key: &ClientKey) -> Self::FheType {
        let client_key = self.type_id.unwrapped_ref_key(key);
        let ciphertext = client_key.key.encrypt(value);
        DynFheBool::new(ciphertext, self.type_id)
    }
}
