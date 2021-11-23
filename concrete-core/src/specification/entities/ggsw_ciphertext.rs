use crate::specification::entities::markers::{GgswCiphertextKind, KeyFlavorMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
};

/// A trait implemented by types embodying a GGSW ciphertext.
///
/// A GGSW ciphertext is associated with a
/// [`KeyFlavor`](`GgswCiphertextEntity::KeyFlavor`) type, which conveys the flavor of secret key it
/// was encrypted with.
///
/// # Formal Definition
pub trait GgswCiphertextEntity: AbstractEntity<Kind = GgswCiphertextKind> {
    /// The flavor of key the ciphertext was encrypted with.
    type KeyFlavor: KeyFlavorMarker;

    /// Returns the GLWE dimension of the ciphertext.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the ciphertext.
    fn polynomial_size(&self) -> PolynomialSize;

    /// Returns the number of decomposition levels of the ciphertext.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;

    /// Returns the logarithm of the base used in the ciphertext.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;
}
