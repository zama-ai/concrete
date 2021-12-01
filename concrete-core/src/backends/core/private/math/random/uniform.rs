use super::*;

/// A distribution type representing uniform sampling for unsigned integer types. The value is
/// uniformly sampled in `[0, 2^n[` where `n` is the size of the integer type.
#[derive(Copy, Clone)]
pub struct Uniform;

macro_rules! implement_uniform {
    ($T:ty, $bytes:literal) => {
        impl RandomGenerable<Uniform> for $T {
            #[allow(unused)]
            fn generate_one(generator: &mut RandomGenerator, distribution: Uniform) -> Self {
                let mut buf = [0; $bytes];
                buf.iter_mut().for_each(|a| *a = generator.generate_next());
                unsafe { *(buf.as_ptr() as *const $T) }
            }
        }
    };
}

implement_uniform!(u8, 1);
implement_uniform!(u16, 2);
implement_uniform!(u32, 4);
implement_uniform!(u64, 8);
implement_uniform!(u128, 16);
implement_uniform!(i8, 1);
implement_uniform!(i16, 1);
implement_uniform!(i32, 1);
implement_uniform!(i64, 1);
implement_uniform!(i128, 1);
