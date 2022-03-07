use crate::backends::core::private::crypto::encoding::Cleartext as ImplCleartext;
use crate::specification::entities::markers::CleartextKind;
use crate::specification::entities::{AbstractEntity, CleartextEntity};

/// A structure representing a cleartext with 32 bits of precision.
#[derive(Debug, Clone, PartialEq)]
pub struct Cleartext32(pub(crate) ImplCleartext<u32>);
impl AbstractEntity for Cleartext32 {
    type Kind = CleartextKind;
}
impl CleartextEntity for Cleartext32 {}

/// A structure representing a cleartext with 64 bits of precision.
#[derive(Debug, Clone, PartialEq)]
pub struct Cleartext64(pub(crate) ImplCleartext<u64>);
impl AbstractEntity for Cleartext64 {
    type Kind = CleartextKind;
}
impl CleartextEntity for Cleartext64 {}

/// A structure representing a cleartext wrapping a 64-bit float.
#[derive(Debug, Clone, PartialEq)]
pub struct CleartextF64(pub(crate) ImplCleartext<f64>);
impl AbstractEntity for CleartextF64 {
    type Kind = CleartextKind;
}
impl CleartextEntity for CleartextF64 {}
