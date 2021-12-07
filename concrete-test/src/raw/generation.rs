//! A module containing sampling entry points for raw integers

use concrete_commons::numeric::{CastInto, UnsignedInteger};
use concrete_core::backends::core::private::math::random::RandomGenerator;

/// A trait to generate raw unsigned integer values.
pub trait RawUnsignedIntegers: UnsignedInteger + CastInto<f64> {
    fn one() -> Self;
    fn one_vec(size: usize) -> Vec<Self>;
    fn zero() -> Self;
    fn zero_vec(size: usize) -> Vec<Self>;
    fn uniform() -> Self;
    fn uniform_vec(size: usize) -> Vec<Self>;
    fn uniform_n_msb(n: usize) -> Self;
    fn uniform_n_msb_vec(n: usize, size: usize) -> Vec<Self>;
    fn uniform_weight() -> Self;
}

impl RawUnsignedIntegers for u32 {
    fn one() -> Self {
        1u32
    }
    fn one_vec(size: usize) -> Vec<Self> {
        vec![1u32; size]
    }
    fn zero() -> Self {
        0u32
    }
    fn zero_vec(size: usize) -> Vec<Self> {
        vec![0u32; size]
    }
    fn uniform() -> Self {
        let mut generator = RandomGenerator::new(None);
        generator.random_uniform()
    }
    fn uniform_vec(size: usize) -> Vec<Self> {
        let mut generator = RandomGenerator::new(None);
        generator.random_uniform_tensor(size).into_container()
    }
    fn uniform_n_msb(n: usize) -> Self {
        let mut generator = RandomGenerator::new(None);
        generator.random_uniform_n_msb(n)
    }
    fn uniform_n_msb_vec(n: usize, size: usize) -> Vec<Self> {
        let mut generator = RandomGenerator::new(None);
        generator
            .random_uniform_n_msb_tensor(size, n)
            .into_container()
    }
    fn uniform_weight() -> Self {
        let val: u32 = Self::uniform();
        let val = val % 1024u32;
        let val: i32 = val as i32;
        let val = val - 512i32;
        val as u32
    }
}

impl RawUnsignedIntegers for u64 {
    fn one() -> Self {
        1u64
    }
    fn one_vec(size: usize) -> Vec<Self> {
        vec![1u64; size]
    }
    fn zero() -> Self {
        0u64
    }
    fn zero_vec(size: usize) -> Vec<Self> {
        vec![0u64; size]
    }
    fn uniform() -> Self {
        let mut generator = RandomGenerator::new(None);
        generator.random_uniform()
    }
    fn uniform_vec(size: usize) -> Vec<Self> {
        let mut generator = RandomGenerator::new(None);
        generator.random_uniform_tensor(size).into_container()
    }
    fn uniform_n_msb(n: usize) -> Self {
        let mut generator = RandomGenerator::new(None);
        generator.random_uniform_n_msb(n)
    }
    fn uniform_n_msb_vec(n: usize, size: usize) -> Vec<Self> {
        let mut generator = RandomGenerator::new(None);
        generator
            .random_uniform_n_msb_tensor(size, n)
            .into_container()
    }
    fn uniform_weight() -> Self {
        let val: u64 = Self::uniform();
        let val = val % 1024u64;
        let val: i64 = val as i64;
        let val = val - 512i64;
        val as u64
    }
}
