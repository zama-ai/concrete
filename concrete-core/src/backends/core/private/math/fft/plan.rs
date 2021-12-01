use crate::backends::core::private::math::fft::Complex64;
use concrete_commons::parameters::PolynomialSize;
use concrete_fftw::plan::{C2CPlan, C2CPlan64};
use concrete_fftw::types::{Flag, Sign};
use lazy_static::lazy_static;
use std::fmt;
use std::fmt::{Debug, Formatter};

/// A set of forward/backward plans to perform the
#[derive(Clone)]
pub struct Plans {
    forward: &'static C2CPlan64,
    backward: &'static C2CPlan64,
    size: PolynomialSize,
}

impl Debug for Plans {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Plans {{ size: {:?} }}", self.size)
    }
}

impl Plans {
    /// Generates a new plan
    pub fn new(size: PolynomialSize) -> Plans {
        debug_assert!(
            [128, 256, 512, 1024, 2048, 4096, 8192, 16384].contains(&size.0),
            "The size chosen is not valid ({}). Should be 128, 256, 512, 1024, 2048, 4096, 8192 \
            or 16384",
            size.0
        );
        let (forward, backward) = match size.0 {
            128 => (&*C2C_128_64_F, &*C2C_128_64_B),
            256 => (&*C2C_256_64_F, &*C2C_256_64_B),
            512 => (&*C2C_512_64_F, &*C2C_512_64_B),
            1024 => (&*C2C_1024_64_F, &*C2C_1024_64_B),
            2048 => (&*C2C_2048_64_F, &*C2C_2048_64_B),
            4096 => (&*C2C_4096_64_F, &*C2C_4096_64_B),
            8192 => (&*C2C_8192_64_F, &*C2C_8192_64_B),
            16384 => (&*C2C_16384_64_F, &*C2C_16384_64_B),
            _ => unreachable!(),
        };
        Plans {
            forward,
            backward,
            size,
        }
    }

    /// Returns the plans polynomial sizes.
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.size
    }

    /// Performs a forward transform.
    pub fn forward(&self, input: &[Complex64], output: &mut [Complex64]) {
        self.forward
            .c2c(
                // It is valid to cast this slice into a mutable slice here, because we use the
                // flag preserveinput, which prevents fftw from writing the input.
                unsafe {
                    std::slice::from_raw_parts_mut(input.as_ptr() as *mut Complex64, input.len())
                },
                output,
            )
            .expect("forward: fft.c2c threw an error...");
    }

    /// Performs a backward transform.
    pub fn backward(&self, input: &[Complex64], output: &mut [Complex64]) {
        self.backward
            .c2c(
                // It is valid to cast this slice into a mutable slice here, because we use the
                // flag preserveinput, which prevents fftw from writing the input.
                unsafe {
                    std::slice::from_raw_parts_mut(input.as_ptr() as *mut Complex64, input.len())
                },
                output,
            )
            .expect("forward: fft.c2c threw an error...");
    }
}

lazy_static! {
    pub static ref C2C_128_64_F: C2CPlan64 =
        <C2CPlan64 as C2CPlan>::aligned(&[128], Sign::Forward, Flag::MEASURE | Flag::PRESERVEINPUT)
            .unwrap();
    pub static ref C2C_128_64_B: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[128],
        Sign::Backward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_256_64_F: C2CPlan64 =
        <C2CPlan64 as C2CPlan>::aligned(&[256], Sign::Forward, Flag::MEASURE | Flag::PRESERVEINPUT)
            .unwrap();
    pub static ref C2C_256_64_B: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[256],
        Sign::Backward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_512_64_F: C2CPlan64 =
        <C2CPlan64 as C2CPlan>::aligned(&[512], Sign::Forward, Flag::MEASURE | Flag::PRESERVEINPUT)
            .unwrap();
    pub static ref C2C_512_64_B: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[512],
        Sign::Backward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_1024_64_F: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[1024],
        Sign::Forward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_1024_64_B: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[1024],
        Sign::Backward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_2048_64_F: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[2048],
        Sign::Forward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_2048_64_B: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[2048],
        Sign::Backward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_4096_64_F: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[4096],
        Sign::Forward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_4096_64_B: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[4096],
        Sign::Backward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_8192_64_F: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[8192],
        Sign::Forward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_8192_64_B: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[8192],
        Sign::Backward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_16384_64_F: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[16384],
        Sign::Forward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
    pub static ref C2C_16384_64_B: C2CPlan64 = <C2CPlan64 as C2CPlan>::aligned(
        &[16384],
        Sign::Backward,
        Flag::MEASURE | Flag::PRESERVEINPUT
    )
    .unwrap();
}
