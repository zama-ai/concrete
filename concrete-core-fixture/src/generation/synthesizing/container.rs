use crate::generation::prototyping::PrototypesContainer;
use crate::generation::IntegerPrecision;

/// A trait allowing to synthesize an actual containre from a prototype.
pub trait SynthesizesContainer<Precision: IntegerPrecision, Container>:
    PrototypesContainer<Precision>
where
    Container: Sized,
{
    fn synthesize_container(&mut self, prototype: &Self::ContainerProto) -> Container;
    fn unsynthesize_container(&mut self, container: Container) -> Self::ContainerProto;
    fn destroy_container(&mut self, container: Container);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoVec32, ProtoVec64};
    use crate::generation::synthesizing::SynthesizesContainer;
    use crate::generation::{Maker, Precision32, Precision64};

    impl SynthesizesContainer<Precision32, Vec<u32>> for Maker {
        fn synthesize_container(&mut self, prototype: &Self::ContainerProto) -> Vec<u32> {
            prototype.0.to_owned()
        }

        fn unsynthesize_container(&mut self, container: Vec<u32>) -> Self::ContainerProto {
            ProtoVec32(container)
        }

        fn destroy_container(&mut self, _container: Vec<u32>) {}
    }

    impl SynthesizesContainer<Precision64, Vec<u64>> for Maker {
        fn synthesize_container(&mut self, prototype: &Self::ContainerProto) -> Vec<u64> {
            prototype.0.to_owned()
        }

        fn unsynthesize_container(&mut self, container: Vec<u64>) -> Self::ContainerProto {
            ProtoVec64(container)
        }

        fn destroy_container(&mut self, _container: Vec<u64>) {}
    }

    impl<'a> SynthesizesContainer<Precision32, &'a [u32]> for Maker {
        fn synthesize_container(&mut self, prototype: &Self::ContainerProto) -> &'a [u32] {
            prototype.0.to_owned().leak() as &[u32]
        }

        fn unsynthesize_container(&mut self, container: &[u32]) -> Self::ContainerProto {
            let reconstructed_vec = unsafe {
                Vec::from_raw_parts(
                    container.as_ptr() as *mut u32,
                    container.len(),
                    container.len(),
                )
            };
            ProtoVec32(reconstructed_vec)
        }

        fn destroy_container(&mut self, container: &[u32]) {
            unsafe {
                Vec::from_raw_parts(
                    container.as_ptr() as *mut u32,
                    container.len(),
                    container.len(),
                )
            };
        }
    }

    impl<'a> SynthesizesContainer<Precision64, &'a [u64]> for Maker {
        fn synthesize_container(&mut self, prototype: &Self::ContainerProto) -> &'a [u64] {
            prototype.0.to_owned().leak() as &[u64]
        }

        fn unsynthesize_container(&mut self, container: &[u64]) -> Self::ContainerProto {
            let reconstructed_vec = unsafe {
                Vec::from_raw_parts(
                    container.as_ptr() as *mut u64,
                    container.len(),
                    container.len(),
                )
            };
            ProtoVec64(reconstructed_vec)
        }

        fn destroy_container(&mut self, container: &[u64]) {
            unsafe {
                Vec::from_raw_parts(
                    container.as_ptr() as *mut u64,
                    container.len(),
                    container.len(),
                )
            };
        }
    }

    impl<'a> SynthesizesContainer<Precision32, &'a mut [u32]> for Maker {
        fn synthesize_container(&mut self, prototype: &Self::ContainerProto) -> &'a mut [u32] {
            prototype.0.to_owned().leak()
        }

        fn unsynthesize_container(&mut self, container: &mut [u32]) -> Self::ContainerProto {
            let reconstructed_vec = unsafe {
                Vec::from_raw_parts(container.as_mut_ptr(), container.len(), container.len())
            };
            ProtoVec32(reconstructed_vec)
        }

        fn destroy_container(&mut self, container: &mut [u32]) {
            unsafe {
                Vec::from_raw_parts(container.as_mut_ptr(), container.len(), container.len())
            };
        }
    }

    impl<'a> SynthesizesContainer<Precision64, &'a mut [u64]> for Maker {
        fn synthesize_container(&mut self, prototype: &Self::ContainerProto) -> &'a mut [u64] {
            prototype.0.to_owned().leak()
        }

        fn unsynthesize_container(&mut self, container: &mut [u64]) -> Self::ContainerProto {
            let reconstructed_vec = unsafe {
                Vec::from_raw_parts(container.as_mut_ptr(), container.len(), container.len())
            };
            ProtoVec64(reconstructed_vec)
        }

        fn destroy_container(&mut self, container: &mut [u64]) {
            unsafe {
                Vec::from_raw_parts(container.as_mut_ptr(), container.len(), container.len())
            };
        }
    }
}
