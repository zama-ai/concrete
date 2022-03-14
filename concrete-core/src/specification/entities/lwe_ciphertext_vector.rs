use crate::prelude::{KeyDistributionMarker, LweCiphertextVectorKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{LweCiphertextCount, LweDimension};

/// A trait implemented by types embodying an LWE ciphertext vector.
///
/// An LWE ciphertext vector is associated with a
/// [`KeyDistribution`](`LweCiphertextVectorEntity::KeyDistribution`) type, which conveys the
/// distribution of the secret key it was encrypted with.
///
/// # Formal Definition
pub trait LweCiphertextVectorEntity: AbstractEntity<Kind = LweCiphertextVectorKind> {
    /// The distribution of key the ciphertext was encrypted with.
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the LWE dimension of the ciphertexts.
    fn lwe_dimension(&self) -> LweDimension;

    /// Returns the number of ciphertexts contained in the vector.
    fn lwe_ciphertext_count(&self) -> LweCiphertextCount;
}
