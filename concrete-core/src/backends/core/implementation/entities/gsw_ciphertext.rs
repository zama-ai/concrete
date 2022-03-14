#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};

use crate::backends::core::private::crypto::gsw::GswCiphertext as ImplGswCiphertext;
use crate::prelude::{BinaryKeyDistribution, GswCiphertextKind};
use crate::specification::entities::{AbstractEntity, GswCiphertextEntity};

/// A structure representing a GSW ciphertext with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GswCiphertext32(ImplGswCiphertext<Vec<u32>, u32>);

impl AbstractEntity for GswCiphertext32 {
    type Kind = GswCiphertextKind;
}

impl GswCiphertextEntity for GswCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_level_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}

/// A structure representing a GSW ciphertext with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GswCiphertext64(ImplGswCiphertext<Vec<u64>, u64>);

impl AbstractEntity for GswCiphertext64 {
    type Kind = GswCiphertextKind;
}

impl GswCiphertextEntity for GswCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_level_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}
