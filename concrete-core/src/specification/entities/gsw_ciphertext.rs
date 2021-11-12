use crate::specification::entities::markers::{GswCiphertextKind, KeyFlavorMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};

/// A trait implemented by types embodying a GSW ciphertext.
///
/// A GSW ciphertext is associated with a
/// [`KeyFlavor`](`GswCiphertextEntity::KeyFlavor`) type, which conveys the flavor of secret
/// key it was encrypted with.
pub trait GswCiphertextEntity: AbstractEntity<Kind = GswCiphertextKind> {
    /// The flavor of key the ciphertext was encrypted with.
    type KeyFlavor: KeyFlavorMarker;

    /// Returns the LWE dimension of the ciphertext.
    fn lwe_dimension(&self) -> LweDimension;

    /// Returns the number of decomposition levels of the ciphertext.
    fn decomposition_level_count(&self) -> DecompositionLevelCount;

    /// Returns the logarithm of the base used in the ciphertext.
    fn decomposition_base_log(&self) -> DecompositionBaseLog;
}
