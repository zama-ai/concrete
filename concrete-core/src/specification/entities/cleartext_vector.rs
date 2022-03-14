use crate::prelude::CleartextVectorKind;
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::CleartextCount;

/// A trait implemented by types embodying a cleartext vector entity.
///
/// # Formal Definition
pub trait CleartextVectorEntity: AbstractEntity<Kind = CleartextVectorKind> {
    /// Returns the number of cleartext contained in the vector.
    fn cleartext_count(&self) -> CleartextCount;
}
