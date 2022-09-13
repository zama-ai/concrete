#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::booleans::types::dynamic::{
    BooleanTypeId, DynFheBoolClientKey, DynFheBoolEncryptor, DynFheBoolParameters,
    DynFheBoolServerKey,
};

use super::types::{FheBoolClientKey, FheBoolServerKey, StaticBoolParameters};
use super::FheBoolParameters;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct BoolConfig {
    pub(crate) parameters: Option<StaticBoolParameters>,

    pub(crate) custom_booleans: Vec<DynFheBoolParameters>,
}

impl BoolConfig {
    pub(crate) fn all_default() -> Self {
        Self {
            parameters: Some(FheBoolParameters::default().into()),

            custom_booleans: vec![],
        }
    }

    pub(crate) fn all_none() -> Self {
        Self {
            parameters: None,

            custom_booleans: vec![],
        }
    }

    pub(crate) fn add_bool_type(&mut self, parameters: FheBoolParameters) -> DynFheBoolEncryptor {
        let type_id = BooleanTypeId(self.custom_booleans.len());
        self.custom_booleans.push(parameters.into());
        <DynFheBoolEncryptor as From<_>>::from(type_id)
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct BoolClientKey {
    pub(super) key: Option<FheBoolClientKey>,

    pub(super) dynamic_keys: Vec<DynFheBoolClientKey>,
}

impl From<BoolConfig> for BoolClientKey {
    fn from(config: BoolConfig) -> Self {
        Self {
            key: config.parameters.map(FheBoolClientKey::new),

            dynamic_keys: config
                .custom_booleans
                .into_iter()
                .map(DynFheBoolClientKey::new)
                .collect(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Default)]
pub(crate) struct BoolServerKey {
    pub(super) key: Option<FheBoolServerKey>,

    pub(super) dynamic_keys: Vec<DynFheBoolServerKey>,
}

impl BoolServerKey {
    pub(crate) fn new(client_key: &BoolClientKey) -> Self {
        Self {
            key: client_key.key.as_ref().map(FheBoolServerKey::new),

            dynamic_keys: client_key
                .dynamic_keys
                .iter()
                .map(DynFheBoolServerKey::new)
                .collect(),
        }
    }
}
