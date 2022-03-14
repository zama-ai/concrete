use crate::prelude::{KeyDistributionMarker, LweKeyswitchKeyKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};

/// A trait implemented by types embodying an LWE keyswitch key.
///
/// An LWE keyswitch key is associated with two [`KeyDistributionMarker`] types:
///
/// + The [`InputKeyDistribution`](`LweKeyswitchKeyEntity::InputKeyDistribution`) type conveys the
/// distribution of the input secret key.
/// + The [`OutputKeyDistribution`](`LweKeyswitchKeyEntity::OutputKeyDistribution`) type conveys the
/// distribution of the output secret key.
///
/// # Formal Definition
pub trait LweKeyswitchKeyEntity: AbstractEntity<Kind = LweKeyswitchKeyKind> {
    /// The distribution of the key the input ciphertext is encrypted with.
    type InputKeyDistribution: KeyDistributionMarker;

    /// The distribution of the key the output ciphertext is encrypted with.
    type OutputKeyDistribution: KeyDistributionMarker;

    /// Returns the input LWE dimension of the key.
    fn input_lwe_dimension(&self) -> LweDimension;

    /// Returns the output lew dimension of the key.
    fn output_lwe_dimension(&self) -> LweDimension;

    /// Returns the number of decomposition levels of the key.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;

    /// Returns the logarithm of the base used in the key.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;
}
