use std::convert::TryInto;
use std::fmt::Binary;

use crate::crypto::UnsignedTorus;
use crate::numeric::CastInto;
use crate::test_tools::{any_usize, any_utorus};

use super::*;

fn test_round_to_closest_multiple<T: UnsignedTorus>() {
    let log_b = any_usize();
    let level_max = any_usize();
    let val = any_utorus::<T>();
    let delta = any_utorus::<T>();
    let bits = T::BITS;
    let log_b = (log_b % ((bits / 4) - 1)) + 1;
    let log_b: usize = log_b.try_into().unwrap();
    let level_max = (level_max % 4) + 1;
    let level_max: usize = level_max.try_into().unwrap();
    let bit: usize = log_b * level_max;

    let val = val << (bits - bit);
    let delta = delta >> (bits - (bits - bit - 1));

    for _ in 0..1000 {
        assert_eq!(
            val,
            val.wrapping_add(delta).round_to_closest_multiple(
                DecompositionBaseLog(log_b),
                DecompositionLevelCount(level_max)
            )
        );
        assert_eq!(
            val,
            val.wrapping_sub(delta).round_to_closest_multiple(
                DecompositionBaseLog(log_b),
                DecompositionLevelCount(level_max)
            )
        );
    }
}

#[test]
fn test_round_to_closest_multiple_u32() {
    test_round_to_closest_multiple::<u32>();
}

#[test]
fn test_round_to_closest_multiple_u64() {
    test_round_to_closest_multiple::<u64>();
}

#[allow(unused)]
fn test_panic_round_to_closest_multiple<T: UnsignedTorus>() {
    //! test that it panics when log_b * level_max==TORUS_BIT
    let log_b: usize = T::BITS / 4;
    let level_max: usize = 4;

    let value = T::ONE;

    value.round_to_closest_multiple(
        DecompositionBaseLog(log_b),
        DecompositionLevelCount(level_max),
    );
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_panic_round_to_closest_multiple_u32() {
    test_panic_round_to_closest_multiple::<u32>();
}

#[test]
#[should_panic]
#[cfg(debug_assertions)]
fn test_panic_round_to_closest_multiple_u64() {
    test_panic_round_to_closest_multiple::<u64>();
}

fn test_signed_decompose_one_level<T: UnsignedTorus + Debug + Binary>() {
    // This test picks a random Torus value,
    // rounds them according to the decomposition precision (base_log*level_max) which is randomly picked each time,
    // decomposes them with the signed_decompose_one_level() function,
    // recomposes Torus elements,
    // and finally makes sure that they are equal to the rounded values.

    let log_b = any_usize();
    let level_max = any_usize();
    let x = any_utorus::<T>();

    let log_b = (log_b % ((T::BITS / 4) - 1)) + 1;
    let log_b: usize = log_b.try_into().unwrap();
    let level_max = (level_max % 4) + 1;
    let level_max: usize = level_max.try_into().unwrap();
    println!("logB:{}, levelMax:{}", log_b, level_max);

    // round the value
    let x = x.round_to_closest_multiple(
        DecompositionBaseLog(log_b),
        DecompositionLevelCount(level_max),
    );
    println!("x:{:?} -> {:b}", x, x);

    // decompose the rounded value
    let mut decomp_x: Vec<T> = vec![T::ZERO; level_max];
    let mut carries: Vec<T> = vec![T::ZERO; level_max];
    for i in (0..level_max).rev() {
        let pair = x.signed_decompose_one_level(
            carries[i],
            DecompositionBaseLog(log_b),
            DecompositionLevel(i),
        );
        decomp_x[i] = pair.0;
        if i > 0 {
            carries[i - 1] = pair.1;
        }
        println!("XXdecomp_{} -> {:?}", i, decomp_x[i]);
    }

    // recompose the Primitive element
    let mut recomp_x = T::ZERO;
    for (i, di) in decomp_x.iter().enumerate() {
        println!("decomp_{} -> {:?}", i, di);
        let right: f64 = di.into_signed().cast_into();
        let left: f64 = T::ONE
            .set_val_at_level(DecompositionBaseLog(log_b), DecompositionLevel(i))
            .cast_into();
        let mut tmp = left * right;
        if tmp < 0. {
            tmp = -tmp;
            recomp_x = recomp_x.wrapping_sub(tmp.cast_into());
        } else {
            recomp_x = recomp_x.wrapping_add(tmp.cast_into());
        }
    }

    println!("recomp x:{:?} -> {:b}", recomp_x, recomp_x);

    println!("recomp:{:?} -> x:{:?}", recomp_x, x);

    // test
    assert_eq!(recomp_x, x);
}

#[test]
fn test_signed_decompose_one_level_u32() {
    test_signed_decompose_one_level::<u32>();
}

#[test]
fn test_signed_decompose_one_level_u64() {
    test_signed_decompose_one_level::<u64>();
}
