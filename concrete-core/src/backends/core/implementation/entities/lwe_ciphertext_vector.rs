#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::parameters::{LweCiphertextCount, LweDimension};

use crate::prelude::{BinaryKeyDistribution, LweCiphertextVectorKind};
use crate::specification::entities::{AbstractEntity, LweCiphertextVectorEntity};

use super::super::super::private::crypto::lwe::LweList as ImplLweList;

/// A structure representing a vector of LWE ciphertexts with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweCiphertextVector32(pub(crate) ImplLweList<Vec<u32>>);

impl AbstractEntity for LweCiphertextVector32 {
    type Kind = LweCiphertextVectorKind;
}

impl LweCiphertextVectorEntity for LweCiphertextVector32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }

    fn lwe_ciphertext_count(&self) -> LweCiphertextCount {
        LweCiphertextCount(self.0.count().0)
    }
}

/// A structure representing a vector of LWE ciphertexts with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweCiphertextVector64(pub(crate) ImplLweList<Vec<u64>>);

impl AbstractEntity for LweCiphertextVector64 {
    type Kind = LweCiphertextVectorKind;
}

impl LweCiphertextVectorEntity for LweCiphertextVector64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }

    fn lwe_ciphertext_count(&self) -> LweCiphertextCount {
        LweCiphertextCount(self.0.count().0)
    }
}
