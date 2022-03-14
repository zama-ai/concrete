use crate::prelude::{GswCiphertextVectorKind, KeyDistributionMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GswCiphertextCount, LweDimension,
};

/// A trait implemented by types embodying a GSW ciphertext vector.
///
/// A GSW ciphertext vector is associated with a
/// [`KeyDistribution`](`GswCiphertextVectorEntity::KeyDistribution`) type, which conveys the
/// distribution of the secret key it was encrypted with.
///
/// # Formal Definition
pub trait GswCiphertextVectorEntity: AbstractEntity<Kind = GswCiphertextVectorKind> {
    /// The distribution of the key the ciphertext was encrypted with.
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the LWE dimension of the ciphertexts.
    fn lwe_dimension(&self) -> LweDimension;

    /// Returns the number of decomposition levels of the ciphertexts.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;

    /// Returns the logarithm of the base used in the ciphertexts.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;

    /// Returns the number of ciphertexts in the vector.
    fn gsw_ciphertext_count(&self) -> GswCiphertextCount;
}
