use crate::backends::core::private::crypto::secret::LweSecretKey as ImpLweSecretKey;
use crate::prelude::{BinaryKeyDistribution, LweSecretKeyKind};
use crate::specification::entities::{AbstractEntity, LweSecretKeyEntity};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::LweDimension;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A structure representing an LWE secret key with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LweSecretKey32(pub(crate) ImpLweSecretKey<BinaryKeyKind, Vec<u32>>);
impl AbstractEntity for LweSecretKey32 {
    type Kind = LweSecretKeyKind;
}
impl LweSecretKeyEntity for LweSecretKey32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.key_size()
    }
}

/// A structure representing an LWE secret key with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LweSecretKey64(pub(crate) ImpLweSecretKey<BinaryKeyKind, Vec<u64>>);
impl AbstractEntity for LweSecretKey64 {
    type Kind = LweSecretKeyKind;
}
impl LweSecretKeyEntity for LweSecretKey64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.key_size()
    }
}
