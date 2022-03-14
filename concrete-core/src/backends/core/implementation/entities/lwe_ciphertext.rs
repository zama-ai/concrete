use super::super::super::private::crypto::lwe::LweCiphertext as ImplLweCiphertext;
use crate::prelude::{BinaryKeyDistribution, LweCiphertextKind};
use crate::specification::entities::{AbstractEntity, LweCiphertextEntity};
use concrete_commons::parameters::LweDimension;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A structure representing an LWE ciphertext with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweCiphertext32(pub(crate) ImplLweCiphertext<Vec<u32>>);
impl AbstractEntity for LweCiphertext32 {
    type Kind = LweCiphertextKind;
}
impl LweCiphertextEntity for LweCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }
}

/// A structure representing an LWE ciphertext with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweCiphertext64(pub(crate) ImplLweCiphertext<Vec<u64>>);
impl AbstractEntity for LweCiphertext64 {
    type Kind = LweCiphertextKind;
}
impl LweCiphertextEntity for LweCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }
}
