use crate::backends::core::private::math::decomposition::SignedDecomposer;
use crate::backends::core::private::math::random::{RandomGenerable, Uniform};
use crate::backends::core::private::math::tensor::Tensor;
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::{any_uint, any_usize, random_usize_between};
use concrete_commons::numeric::{Numeric, SignedInteger, UnsignedInteger};
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
use std::fmt::Debug;

// Returns a random decomposition valid for the size of the T type.
fn random_decomp<T: UnsignedInteger>() -> SignedDecomposer<T> {
    let mut base_log;
    let mut level_count;
    loop {
        base_log = random_usize_between(2..T::BITS);
        level_count = random_usize_between(2..T::BITS);
        if base_log * level_count < T::BITS {
            break;
        }
    }
    SignedDecomposer::new(
        DecompositionBaseLog(base_log),
        DecompositionLevelCount(level_count),
    )
}

fn test_decompose_recompose<T: UnsignedInteger + Debug + RandomGenerable<Uniform>>()
where
    <T as UnsignedInteger>::Signed: Debug + SignedInteger,
{
    // Checks that the decomposing and recomposing a value brings the closest representable
    for _ in 0..100_000 {
        let decomposer = random_decomp::<T>();
        let input = any_uint::<T>();
        for term in decomposer.decompose(input) {
            assert!(1 <= term.level().0);
            assert!(term.level().0 <= decomposer.level_count);
            let signed_term = term.value().into_signed();
            let half_basis = (T::Signed::ONE << decomposer.base_log) / T::TWO.into_signed();
            assert!(-half_basis <= signed_term);
            assert!(signed_term < half_basis);
        }
        let closest = decomposer.closest_representable(input);
        assert_eq!(
            closest,
            decomposer.recompose(decomposer.decompose(closest)).unwrap()
        );
    }
}

#[test]
fn test_decompose_recompose_u32() {
    test_decompose_recompose::<u32>()
}

#[test]
fn test_decompose_recompose_u64() {
    test_decompose_recompose::<u64>()
}

fn test_decompose_recompose_tensor<T: UnsignedInteger + Debug + RandomGenerable<Uniform>>()
where
    <T as UnsignedInteger>::Signed: Debug + SignedInteger,
{
    // Checks that the decomposing and recomposing a value brings the closest representable
    for _ in 0..100_000 {
        let decomposer = random_decomp::<T>();
        let input = Tensor::allocate(any_uint::<T>(), 1);
        let mut decomp = decomposer.decompose_tensor(&input);
        while let Some(term) = decomp.next_term() {
            assert!(1 <= term.level().0);
            assert!(term.level().0 <= decomposer.level_count);
            let signed_term = term.as_tensor().get_element(0).into_signed();
            let half_basis = (T::Signed::ONE << decomposer.base_log) / T::TWO.into_signed();
            assert!(-half_basis <= signed_term);
            assert!(signed_term < half_basis);
        }
        let mut rounded = Tensor::allocate(T::ZERO, 1);
        decomposer.fill_tensor_with_closest_representable(&mut rounded, &input);
        let mut recomposition = Tensor::allocate(T::ZERO, 1);
        let decomp_iter = decomposer.decompose_tensor(&rounded);
        decomposer.fill_tensor_with_recompose(decomp_iter, &mut recomposition);
        assert_eq!(rounded, recomposition);
    }
}

#[test]
fn test_decompose_recompose_tensor_u32() {
    test_decompose_recompose_tensor::<u32>()
}

#[test]
fn test_decompose_recompose_tensor_u64() {
    test_decompose_recompose_tensor::<u64>()
}

fn test_round_to_closest_representable<T: UnsignedTorus>() {
    for _ in 0..1000 {
        let log_b = any_usize();
        let level_max = any_usize();
        let val = any_uint::<T>();
        let delta = any_uint::<T>();
        let bits = T::BITS;
        let log_b = (log_b % ((bits / 4) - 1)) + 1;
        let level_max = (level_max % 4) + 1;
        let bit: usize = log_b * level_max;

        let val = val << (bits - bit);
        let delta = delta >> (bits - (bits - bit - 1));

        let decomposer = SignedDecomposer::new(
            DecompositionBaseLog(log_b),
            DecompositionLevelCount(level_max),
        );

        assert_eq!(
            val,
            decomposer.closest_representable(val.wrapping_add(delta))
        );
        assert_eq!(
            val,
            decomposer.closest_representable(val.wrapping_sub(delta))
        );
    }
}

#[test]
fn test_round_to_closest_representable_u32() {
    test_round_to_closest_representable::<u32>();
}

#[test]
fn test_round_to_closest_representable_u64() {
    test_round_to_closest_representable::<u64>();
}

fn test_round_to_closest_twice<T: UnsignedTorus + Debug>() {
    for _ in 0..1000 {
        let decomp = random_decomp();
        let input: T = any_uint();

        let rounded_once = decomp.closest_representable(input);
        let rounded_twice = decomp.closest_representable(rounded_once);
        assert_eq!(rounded_once, rounded_twice);
    }
}

#[test]
fn test_round_to_closest_twice_u32() {
    test_round_to_closest_twice::<u32>();
}

#[test]
fn test_round_to_closest_twice_u64() {
    test_round_to_closest_twice::<u64>();
}

fn test_round_tensor_to_closest_twice<T: UnsignedTorus + Debug>() {
    for _ in 0..1000 {
        let decomp = random_decomp();
        let input: T = any_uint();
        let input = Tensor::from_container(vec![input]);
        let mut rounded_once = Tensor::from_container(vec![T::ZERO]);
        let mut rounded_twice = Tensor::from_container(vec![T::ZERO]);

        decomp.fill_tensor_with_closest_representable(&mut rounded_once, &input);
        decomp.fill_tensor_with_closest_representable(&mut rounded_twice, &rounded_once);
        assert_eq!(rounded_once, rounded_twice);
    }
}

#[test]
fn test_round_tensor_to_closest_twice_u32() {
    test_round_tensor_to_closest_twice::<u32>();
}

#[test]
fn test_round_tensor_to_closest_twice_u64() {
    test_round_tensor_to_closest_twice::<u64>();
}
