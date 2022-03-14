use crate::prelude::CleartextKind;
use crate::specification::entities::AbstractEntity;

/// A trait implemented by types embodying a cleartext entity.
///
/// # Formal Definition
pub trait CleartextEntity: AbstractEntity<Kind = CleartextKind> {}
