//! A module containing sampling entry points for raw integers
use concrete_commons::numeric::{CastInto, UnsignedInteger};
use concrete_core::backends::core::private::math::random::RandomGenerator;
use std::fmt::Debug;
use std::ops::Range;

/// A trait to generate raw unsigned integer values.
pub trait RawUnsignedIntegers: UnsignedInteger + CastInto<f64> + Debug {
    fn one() -> Self;
    fn one_vec(size: usize) -> Vec<Self>;
    fn zero() -> Self;
    fn zero_vec(size: usize) -> Vec<Self>;
    fn uniform() -> Self;
    fn uniform_vec(size: usize) -> Vec<Self>;
    fn uniform_n_msb(n: usize) -> Self;
    fn uniform_n_msb_vec(n: usize, size: usize) -> Vec<Self>;
    fn uniform_between(range: Range<usize>) -> Self;
    fn uniform_between_vec(range: Range<usize>, size: usize) -> Vec<Self>;
    fn uniform_zero_centered(width: usize) -> Self;
    fn uniform_zero_centered_vec(width: usize, size: usize) -> Vec<Self>;
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

    fn uniform_between(range: Range<usize>) -> Self {
        let mut generator = RandomGenerator::new(None);
        let val: u32 = generator.random_uniform();
        val % ((range.end as u32) - (range.start as u32)) + (range.start as u32)
    }

    fn uniform_between_vec(range: Range<usize>, size: usize) -> Vec<Self> {
        let mut generator = RandomGenerator::new(None);
        let mut output = generator.random_uniform_tensor(size).into_container();
        output.iter_mut().for_each(|val| {
            *val %= ((range.end as u32) - (range.start as u32)) + (range.start as u32)
        });
        output
    }

    fn uniform_zero_centered(width: usize) -> Self {
        let val: u32 = Self::uniform();
        let val = val % (width as u32);
        let val: i32 = val as i32;
        let val = val - ((width / 2) as i32);
        val as u32
    }
    fn uniform_zero_centered_vec(width: usize, size: usize) -> Vec<Self> {
        let mut generator = RandomGenerator::new(None);
        let mut output = generator.random_uniform_tensor(size).into_container();
        output.iter_mut().for_each(|val| {
            let v = *val % (width as u32);
            let v: i32 = v as i32;
            let v = v - ((width / 2) as i32);
            *val = v as u32;
        });
        output
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

    fn uniform_between(range: Range<usize>) -> Self {
        let mut generator = RandomGenerator::new(None);
        let val: u64 = generator.random_uniform();
        val % ((range.end as u64) - (range.start as u64)) + (range.start as u64)
    }

    fn uniform_between_vec(range: Range<usize>, size: usize) -> Vec<Self> {
        let mut generator = RandomGenerator::new(None);
        let mut output = generator.random_uniform_tensor(size).into_container();
        output.iter_mut().for_each(|val| {
            *val %= ((range.end as u64) - (range.start as u64)) + (range.start as u64)
        });
        output
    }

    fn uniform_zero_centered(width: usize) -> Self {
        let val: u64 = Self::uniform();
        let val = val % (width as u64);
        let val: i64 = val as i64;
        let val = val - ((width / 2) as i64);
        val as u64
    }
    fn uniform_zero_centered_vec(width: usize, size: usize) -> Vec<Self> {
        let mut generator = RandomGenerator::new(None);
        let mut output = generator.random_uniform_tensor(size).into_container();
        output.iter_mut().for_each(|val| {
            let v = *val % (width as u64);
            let v: i64 = v as i64;
            let v = v - ((width / 2) as i64);
            *val = v as u64;
        });
        output
    }
}
