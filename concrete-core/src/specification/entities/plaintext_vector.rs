use crate::specification::entities::markers::PlaintextVectorKind;
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::PlaintextCount;

/// A trait implemented by types embodying a plaintext vector.
pub trait PlaintextVectorEntity: AbstractEntity<Kind = PlaintextVectorKind> {
    /// Returns the number of plaintext contained in the vector.
    fn plaintext_count(&self) -> PlaintextCount;
}
