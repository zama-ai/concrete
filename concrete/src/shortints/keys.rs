use super::r#static::{
    FheUint2ClientKey, FheUint2Parameters, FheUint2ServerKey, FheUint3ClientKey,
    FheUint3Parameters, FheUint3ServerKey, FheUint4ClientKey, FheUint4Parameters,
    FheUint4ServerKey,
};

use super::dynamic::{
    DynShortIntClientKey, DynShortIntEncryptor, DynShortIntParameters, DynShortIntServerKey,
    ShortIntTypeId,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub(crate) struct ShortIntConfig {
    pub(crate) uint2_parameters: Option<FheUint2Parameters>,
    pub(crate) uint3_parameters: Option<FheUint3Parameters>,
    pub(crate) uint4_parameters: Option<FheUint4Parameters>,

    pub(crate) custom_short_ints: Vec<DynShortIntParameters>,
}

impl ShortIntConfig {
    pub(crate) fn all_default() -> Self {
        Self {
            uint2_parameters: Some(FheUint2Parameters::default()),
            uint3_parameters: Some(FheUint3Parameters::default()),
            uint4_parameters: Some(FheUint4Parameters::default()),

            custom_short_ints: vec![],
        }
    }

    pub(crate) fn all_none() -> Self {
        Self {
            uint2_parameters: None,
            uint3_parameters: None,
            uint4_parameters: None,

            custom_short_ints: vec![],
        }
    }

    pub(crate) fn add_short_int_type(
        &mut self,
        parameters: DynShortIntParameters,
    ) -> DynShortIntEncryptor {
        let type_id = ShortIntTypeId(self.custom_short_ints.len());
        self.custom_short_ints.push(parameters);
        DynShortIntEncryptor::from(type_id)
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct ShortIntClientKey {
    pub(super) uint2_key: Option<FheUint2ClientKey>,
    pub(super) uint3_key: Option<FheUint3ClientKey>,
    pub(super) uint4_key: Option<FheUint4ClientKey>,

    pub(super) dynamic_keys: Vec<DynShortIntClientKey>,
}

impl From<ShortIntConfig> for ShortIntClientKey {
    fn from(config: ShortIntConfig) -> Self {
        Self {
            uint2_key: config.uint2_parameters.map(FheUint2ClientKey::new),
            uint3_key: config.uint3_parameters.map(FheUint3ClientKey::new),
            uint4_key: config.uint4_parameters.map(FheUint4ClientKey::new),

            dynamic_keys: config
                .custom_short_ints
                .into_iter()
                .map(DynShortIntClientKey::new)
                .collect(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Default)]
pub(crate) struct ShortIntServerKey {
    pub(super) uint2_key: Option<FheUint2ServerKey>,
    pub(super) uint3_key: Option<FheUint3ServerKey>,
    pub(super) uint4_key: Option<FheUint4ServerKey>,

    pub(super) dynamic_keys: Vec<DynShortIntServerKey>,
}

impl ShortIntServerKey {
    pub(crate) fn new(client_key: &ShortIntClientKey) -> Self {
        Self {
            uint2_key: client_key.uint2_key.as_ref().map(FheUint2ServerKey::new),
            uint3_key: client_key.uint3_key.as_ref().map(FheUint3ServerKey::new),
            uint4_key: client_key.uint4_key.as_ref().map(FheUint4ServerKey::new),

            dynamic_keys: client_key
                .dynamic_keys
                .iter()
                .map(DynShortIntServerKey::new)
                .collect(),
        }
    }
}
