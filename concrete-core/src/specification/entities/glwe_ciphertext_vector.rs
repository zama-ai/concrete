use crate::prelude::{GlweCiphertextVectorKind, KeyDistributionMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{GlweCiphertextCount, GlweDimension, PolynomialSize};

/// A trait implemented by types embodying a GLWE ciphertext vector.
///
/// A GLWE ciphertext vector is associated with a
/// [`KeyDistribution`](`GlweCiphertextVectorEntity::KeyDistribution`) type, which conveys the
/// distribution of the secret key it was encrypted with.
///
/// # Formal Definition
///
/// GLWE ciphertexts generalize LWE ciphertexts by definition, however in this library, GLWE
/// ciphertext entities do not generalize LWE ciphertexts, i.e., polynomial size cannot be 1.
pub trait GlweCiphertextVectorEntity: AbstractEntity<Kind = GlweCiphertextVectorKind> {
    /// The distribution of the key the ciphertext was encrypted with.
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the GLWE dimension of the ciphertexts.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the ciphertexts.
    fn polynomial_size(&self) -> PolynomialSize;

    /// Returns the number of ciphertexts in the vector.
    fn glwe_ciphertext_count(&self) -> GlweCiphertextCount;
}
