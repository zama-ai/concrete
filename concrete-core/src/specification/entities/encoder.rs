use crate::specification::entities::markers::EncoderKind;
use crate::specification::entities::AbstractEntity;

/// A trait implemented by types embodying an encoder entity.
pub trait EncoderEntity: AbstractEntity<Kind = EncoderKind> {}
