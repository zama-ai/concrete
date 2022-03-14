use crate::prelude::EncoderKind;
use crate::specification::entities::AbstractEntity;

/// A trait implemented by types embodying an encoder entity.
///
/// # Formal Definition
pub trait EncoderEntity: AbstractEntity<Kind = EncoderKind> {}
