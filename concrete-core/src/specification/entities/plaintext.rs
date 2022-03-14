use crate::prelude::PlaintextKind;
use crate::specification::entities::AbstractEntity;

/// A trait implemented by types embodying a plaintext.
///
/// # Formal Definition
pub trait PlaintextEntity: AbstractEntity<Kind = PlaintextKind> {}
