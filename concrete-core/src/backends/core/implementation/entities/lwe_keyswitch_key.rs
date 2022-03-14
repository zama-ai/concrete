use crate::backends::core::private::crypto::lwe::LweKeyswitchKey as ImplLweKeyswitchKey;
use crate::prelude::{BinaryKeyDistribution, LweKeyswitchKeyKind};
use crate::specification::entities::{AbstractEntity, LweKeyswitchKeyEntity};
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A structure representing an LWE keyswitch key with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LweKeyswitchKey32(pub(crate) ImplLweKeyswitchKey<Vec<u32>>);
impl AbstractEntity for LweKeyswitchKey32 {
    type Kind = LweKeyswitchKeyKind;
}
impl LweKeyswitchKeyEntity for LweKeyswitchKey32 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;

    fn input_lwe_dimension(&self) -> LweDimension {
        self.0.before_key_size()
    }

    fn output_lwe_dimension(&self) -> LweDimension {
        self.0.after_key_size()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_levels_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}

/// A structure representing an LWE keyswitch key with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LweKeyswitchKey64(pub(crate) ImplLweKeyswitchKey<Vec<u64>>);
impl AbstractEntity for LweKeyswitchKey64 {
    type Kind = LweKeyswitchKeyKind;
}
impl LweKeyswitchKeyEntity for LweKeyswitchKey64 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;

    fn input_lwe_dimension(&self) -> LweDimension {
        self.0.before_key_size()
    }

    fn output_lwe_dimension(&self) -> LweDimension {
        self.0.after_key_size()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_levels_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}
