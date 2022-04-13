use crate::generation::prototypes::{ContainerPrototype, ProtoVec32, ProtoVec64};
// , ProtoMutSlice32, ProtoMutSlice64, ProtoSlice32, ProtoSlice64
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};

/// A trait allowing to manipulate container prototypes.
pub trait PrototypesContainer<Precision: IntegerPrecision> {
    type ContainerProto: ContainerPrototype<Precision = Precision>;
    fn transform_raw_vec_to_container(&mut self, raw: &[Precision::Raw]) -> Self::ContainerProto;
    fn transform_container_to_raw_vec(
        &mut self,
        container: &Self::ContainerProto,
    ) -> Vec<Precision::Raw>;
}

impl PrototypesContainer<Precision32> for Maker {
    type ContainerProto = ProtoVec32;

    fn transform_raw_vec_to_container(&mut self, raw: &[u32]) -> Self::ContainerProto {
        ProtoVec32(raw.to_owned())
    }

    fn transform_container_to_raw_vec(&mut self, cleartext: &Self::ContainerProto) -> Vec<u32> {
        cleartext.0.to_owned()
    }
}

impl PrototypesContainer<Precision64> for Maker {
    type ContainerProto = ProtoVec64;

    fn transform_raw_vec_to_container(&mut self, raw: &[u64]) -> Self::ContainerProto {
        ProtoVec64(raw.to_owned())
    }

    fn transform_container_to_raw_vec(&mut self, cleartext: &Self::ContainerProto) -> Vec<u64> {
        cleartext.0.to_owned()
    }
}
