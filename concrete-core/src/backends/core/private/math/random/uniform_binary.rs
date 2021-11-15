use super::*;

/// A distribution type representing uniform sampling for binary type.
#[derive(Clone, Copy)]
pub struct UniformBinary;

macro_rules! implement_uniform_binary {
    ($T:ty) => {
        impl RandomGenerable<UniformBinary> for $T {
            #[allow(unused)]
            fn generate_one(generator: &mut RandomGenerator, distribution: UniformBinary) -> Self {
                if generator.generate_next() & 1 == 1 {
                    1
                } else {
                    0
                }
            }
        }
    };
}

implement_uniform_binary!(u8);
implement_uniform_binary!(u16);
implement_uniform_binary!(u32);
implement_uniform_binary!(u64);
