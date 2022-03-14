use crate::prelude::{GswCiphertextKind, KeyDistributionMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};

/// A trait implemented by types embodying a GSW ciphertext.
///
/// A GSW ciphertext is associated with a
/// [`KeyDistribution`](`GswCiphertextEntity::KeyDistribution`) type, which conveys the distribution
/// of the secret key it was encrypted with.
///
/// # Formal Definition
pub trait GswCiphertextEntity: AbstractEntity<Kind = GswCiphertextKind> {
    /// The distribution of the key the ciphertext was encrypted with.
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the LWE dimension of the ciphertext.
    fn lwe_dimension(&self) -> LweDimension;

    /// Returns the number of decomposition levels of the ciphertext.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;

    /// Returns the logarithm of the base used in the ciphertext.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;
}
