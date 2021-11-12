use crate::specification::entities::markers::{GlweSecretKeyKind, KeyFlavorMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};

/// A trait implemented by types embodying a GLWE secret key.
///
/// A GLWE secret key is associated with a
/// [`KeyFlavor`](`GlweSecretKeyEntity::KeyFlavor`) type, which conveys its flavor.
pub trait GlweSecretKeyEntity: AbstractEntity<Kind = GlweSecretKeyKind> {
    /// The flavor of this key.
    type KeyFlavor: KeyFlavorMarker;

    /// Returns the GLWE dimension of the key.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the key.
    fn polynomial_size(&self) -> PolynomialSize;
}
