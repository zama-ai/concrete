use super::super::super::private::crypto::encoding::PlaintextList as CorePlaintextList;
use crate::prelude::PlaintextVectorKind;
use crate::specification::entities::{AbstractEntity, PlaintextVectorEntity};
use concrete_commons::parameters::PlaintextCount;

/// A structure representing a vector of plaintexts with 32 bits of precision.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaintextVector32(pub(crate) CorePlaintextList<Vec<u32>>);
impl AbstractEntity for PlaintextVector32 {
    type Kind = PlaintextVectorKind;
}
impl PlaintextVectorEntity for PlaintextVector32 {
    fn plaintext_count(&self) -> PlaintextCount {
        self.0.count()
    }
}

/// A structure representing a vector of plaintexts with 64 bits of precision.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaintextVector64(pub(crate) CorePlaintextList<Vec<u64>>);
impl AbstractEntity for PlaintextVector64 {
    type Kind = PlaintextVectorKind;
}
impl PlaintextVectorEntity for PlaintextVector64 {
    fn plaintext_count(&self) -> PlaintextCount {
        self.0.count()
    }
}
