use crate::specification::entities::markers::{GgswCiphertextVectorKind, KeyFlavorMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GgswCiphertextCount, GlweDimension,
    PolynomialSize,
};

/// A trait implemented by types embodying a GGSW ciphertext vector.
///
/// A GGSW ciphertext vector is associated with a
/// [`KeyFlavor`](`GgswCiphertextVectorEntity::KeyFlavor`) type, which conveys the flavor of secret
/// key it was encrypted with.
pub trait GgswCiphertextVectorEntity: AbstractEntity<Kind = GgswCiphertextVectorKind> {
    /// The flavor of key the ciphertext was encrypted with.
    type KeyFlavor: KeyFlavorMarker;

    /// Returns the GLWE dimension of the ciphertexts.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the ciphertexts.
    fn polynomial_size(&self) -> PolynomialSize;

    /// Returns the number of decomposition levels of the ciphertexts.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;

    /// Returns the logarithm of the base used in the ciphertexts.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;

    /// Returns the number of ciphertexts in the vector.
    fn ggsw_ciphertext_count(&self) -> GgswCiphertextCount;
}
