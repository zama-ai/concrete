use crate::specification::entities::markers::CleartextKind;
use crate::specification::entities::AbstractEntity;

/// A trait implemented by types embodying a cleartext entity.
pub trait CleartextEntity: AbstractEntity<Kind = CleartextKind> {}
