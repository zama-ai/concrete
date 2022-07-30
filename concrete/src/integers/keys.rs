use super::r#static::{
    FheUint12ClientKey, FheUint12ServerKey, FheUint16ClientKey, FheUint16ServerKey,
    FheUint8ClientKey, FheUint8ServerKey,
};
use crate::FheUint2Parameters;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::dynamic::{
    DynIntegerClientKey, DynIntegerEncryptor, DynIntegerParameters, DynIntegerServerKey,
    IntegerTypeId,
};

#[derive(Clone, Debug)]
pub(crate) struct IntegerConfig {
    pub(crate) uint8_params: Option<FheUint2Parameters>,
    pub(crate) uint12_params: Option<FheUint2Parameters>,
    pub(crate) uint16_params: Option<FheUint2Parameters>,

    pub(crate) custom_params: Vec<DynIntegerParameters>,
}

impl IntegerConfig {
    pub(crate) fn all_default() -> Self {
        Self {
            uint8_params: Some(FheUint2Parameters::default()),
            uint12_params: Some(FheUint2Parameters::default()),
            uint16_params: Some(FheUint2Parameters::default()),

            custom_params: vec![],
        }
    }

    pub(crate) fn all_none() -> Self {
        Self {
            uint8_params: None,
            uint12_params: None,
            uint16_params: None,

            custom_params: vec![],
        }
    }

    pub(crate) fn add_integer_type(
        &mut self,
        parameters: DynIntegerParameters,
    ) -> DynIntegerEncryptor {
        let type_id = IntegerTypeId(self.custom_params.len());
        self.custom_params.push(parameters);
        <DynIntegerEncryptor as From<IntegerTypeId>>::from(type_id)
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct IntegerClientKey {
    pub(super) uint8_key: Option<FheUint8ClientKey>,
    pub(super) uint12_key: Option<FheUint12ClientKey>,
    pub(super) uint16_key: Option<FheUint16ClientKey>,

    pub(super) custom_keys: Vec<DynIntegerClientKey>,
}

impl From<IntegerConfig> for IntegerClientKey {
    fn from(config: IntegerConfig) -> Self {
        Self {
            uint8_key: config.uint8_params.map(FheUint8ClientKey::from),
            uint12_key: config.uint12_params.map(FheUint12ClientKey::from),
            uint16_key: config.uint16_params.map(FheUint16ClientKey::from),

            custom_keys: config
                .custom_params
                .into_iter()
                .map(DynIntegerClientKey::from)
                .collect(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default, Clone)]
pub(crate) struct IntegerServerKey {
    pub(super) uint8_key: Option<FheUint8ServerKey>,
    pub(super) uint12_key: Option<FheUint12ServerKey>,
    pub(super) uint16_key: Option<FheUint16ServerKey>,

    pub(super) custom_keys: Vec<DynIntegerServerKey>,
}

impl IntegerServerKey {
    pub(crate) fn new(client_key: &IntegerClientKey) -> Self {
        Self {
            uint8_key: client_key.uint8_key.as_ref().map(FheUint8ServerKey::new),
            uint12_key: client_key.uint12_key.as_ref().map(FheUint12ServerKey::new),
            uint16_key: client_key.uint16_key.as_ref().map(FheUint16ServerKey::new),

            custom_keys: client_key
                .custom_keys
                .iter()
                .map(DynIntegerServerKey::new)
                .collect(),
        }
    }
}
