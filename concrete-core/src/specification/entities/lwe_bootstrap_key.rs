use crate::specification::entities::markers::{KeyFlavorMarker, LweBootstrapKeyKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};

/// A trait implemented by types embodying an LWE bootstrap key.
///
/// An LWE bootstrap key is associated with two [`KeyFlavorMarker`] types:
///
/// + The [`InputKeyFlavor`](`LweBootstrapKeyEntity::InputKeyFlavor`) type conveys the flavor of the
/// secret key encrypted inside the bootstrap key.
/// + The [`OutputKeyFlavor`](`LweBootstrapKeyEntity::OutputKeyFlavor`) type conveys the flavor of
/// the secret key used to encrypt the bootstrap key.
pub trait LweBootstrapKeyEntity: AbstractEntity<Kind = LweBootstrapKeyKind> {
    /// The flavor of key the input ciphertext is encrypted with.
    type InputKeyFlavor: KeyFlavorMarker;

    /// The flavor of the key the output ciphertext is encrypted with.
    type OutputKeyFlavor: KeyFlavorMarker;

    /// Returns the GLWE dimension of the key.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the key.
    fn polynomial_size(&self) -> PolynomialSize;

    /// Returns the input LWE dimension of the key.
    fn input_lwe_dimension(&self) -> LweDimension;

    /// Returns the output LWE dimension of the key.
    fn output_lwe_dimension(&self) -> LweDimension {
        LweDimension(self.glwe_dimension().0 * self.polynomial_size().0)
    }

    /// Returns the number of decomposition levels of the key.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;

    /// Returns the logarithm of the base used in the key.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;
}
