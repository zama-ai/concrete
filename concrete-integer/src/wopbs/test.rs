//!
//! # WARNING: this module is experimental.

#![allow(unused)]

use crate::gen_keys;
use crate::parameters::*;
use crate::wopbs::{encode_radix, WopbsKey};
use concrete_shortint::parameters::parameters_wopbs::*;
use concrete_shortint::parameters::parameters_wopbs_message_carry::*;
use concrete_shortint::parameters::{Parameters, *};
use rand::Rng;
use std::cmp::max;

use crate::keycache::{KEY_CACHE, KEY_CACHE_WOPBS};
use paste::paste;

const NB_TEST: usize = 10;

macro_rules! create_parametrized_test{
    ($name:ident { $( ($sks_param:ident, $wopbs_param:ident) ),* }) => {
        paste! {
            $(
            #[test]
            fn [<test_ $name _ $wopbs_param:lower>]() {
                $name(($sks_param, $wopbs_param))
            }
            )*
        }
    };
     ($name:ident)=> {
        create_parametrized_test!($name
        {
            (PARAM_MESSAGE_2_CARRY_2, WOPBS_PARAM_MESSAGE_2_CARRY_2),
            (PARAM_MESSAGE_2_CARRY_2, WOPBS_PARAM_MESSAGE_3_CARRY_3),
            (PARAM_MESSAGE_2_CARRY_2, WOPBS_PARAM_MESSAGE_4_CARRY_4)
        });
    };
}

//create_parametrized_test!(wopbs_v_init);
//create_parametrized_test!(wopbs_v0_16_bits);
// create_parametrized_test!(wopbs_v1);
create_parametrized_test!(wopbs_16_lut_test);
create_parametrized_test!(wopbs_crt_without_padding);
create_parametrized_test!(wopbs_crt_fake_crt);
create_parametrized_test!(wopbs_bivariate_radix);
create_parametrized_test!(wopbs_bivariate_fake_crt);
create_parametrized_test!(wopbs_crt_without_padding_bivariate);
create_parametrized_test!(wopbs_radix);

pub fn wopbs(params: (Parameters, Parameters)) {
    let (cks, mut sks) = gen_keys(&params.0);
    let wopbs_key = KEY_CACHE_WOPBS.get_from_params(params);

    let mut rng = rand::thread_rng();
    let mut cpt = 0;
    let nb_test = 1;

    for _ in 0..nb_test {
        println!("-------------------------------------");

        let clear = rng.gen::<usize>() % 8;
        let mut ct = cks.encrypt_radix(clear as u64, 8);
        let delta =
            63 - f64::log2((params.0.message_modulus.0 * params.0.carry_modulus.0) as f64) as u64;

        let nb_bit_to_extract =
            f64::log2((params.0.message_modulus.0 * params.0.carry_modulus.0) as f64) as usize * 2;

        let mut lut_size = params.0.polynomial_size.0;
        if (1 << nb_bit_to_extract) > params.0.polynomial_size.0 {
            lut_size = 1 << nb_bit_to_extract;
        }

        let mut lut_1: Vec<u64> = vec![];
        for _ in 0..lut_size {
            lut_1.push(1 << delta);
        }
        let big_lut = vec![lut_1; 8];

        let ct_res = wopbs_key.wopbs(&ct, &big_lut);

        let res = cks.decrypt_radix(&ct_res);
        if res != 1 {
            cpt += 1;
        }
        println!("res = {}, decoded={}", res, 1);
        //assert_eq!(res, decoded);
        println!("-------------------------------------");
    }
    println!("Failure Rate = {}/{}", cpt, nb_test);
    panic!();
}

pub fn wopbs_16_bits(params: (Parameters, Parameters)) {
    let nb_block = 3;
    let (cks, mut sks) = gen_keys(&params.0);
    let wopbs_key = KEY_CACHE_WOPBS.get_from_params(params);

    let mut rng = rand::thread_rng();
    let mut cpt = 0;
    let nb_test = 100;

    for _ in 0..nb_test {
        println!("-------------------------------------");

        let clear = rng.gen::<usize>() % 8;
        let mut ct = cks.encrypt_radix(clear as u64, 8);
        let delta =
            63 - f64::log2((params.0.message_modulus.0 * params.0.carry_modulus.0) as f64) as u64;

        let nb_bit_to_extract =
            f64::log2((params.0.message_modulus.0 * params.0.carry_modulus.0) as f64) as usize
                * nb_block;

        let mut lut_size = params.0.polynomial_size.0;
        if (1 << nb_bit_to_extract) > params.0.polynomial_size.0 {
            lut_size = 1 << nb_bit_to_extract;
        }

        let mut lut_1: Vec<u64> = vec![];
        let mut lut_2: Vec<u64> = vec![];
        for _ in 0..lut_size {
            lut_1.push(
                (rng.gen::<u64>() % (params.0.message_modulus.0 * params.0.carry_modulus.0) as u64)
                    << delta,
            );
            lut_2.push(
                (rng.gen::<u64>() % (params.0.message_modulus.0 * params.0.carry_modulus.0) as u64)
                    << delta,
            );
        }

        let lut_res_1 = lut_1.clone();
        let lut_res_2 = lut_2.clone();

        let ct_res = wopbs_key.wopbs(&ct, &[lut_1, lut_2]);

        let shift_clear = ((clear & 100) << 2) + ((clear & 10) << 1) + (clear & 1);
        //println!("nulber of block of the outputi ciphertext = {}", ct_res.ct_vec.len());

        let res = cks.decrypt_radix(&ct_res);
        let decoded_1 =
            (lut_res_1[shift_clear] + 2 * (lut_res_1[shift_clear] & (1 << (delta - 1)))) >> delta;
        let decoded_2 =
            (lut_res_2[shift_clear] + 2 * (lut_res_2[shift_clear] & (1 << (delta - 1)))) >> delta;
        let decoded = ((decoded_2 << 1) + decoded_1) % 8;
        //Deciphering each block separately
        for (i, block) in ct_res.blocks.iter().enumerate() {
            println!(
                "block numero {} = {}",
                i,
                cks.key.decrypt_message_and_carry(block)
            );
        }
        println!("decoded_1 : {}", decoded_1);
        println!("decoded_2 : {}", decoded_2);

        if res != decoded {
            cpt += 1;
        }
        println!("res = {}, decoded={}", res, decoded);
        //assert_eq!(res, decoded);
        println!("-------------------------------------");
    }
    println!("Failure Rate = {}/{}", cpt, nb_test);
    panic!();
}

pub fn wopbs_16_lut_test(params: (Parameters, Parameters)) {
    let nb_block = 3;
    let (cks, mut sks) = gen_keys(&params.0);
    let wopbs_key = KEY_CACHE_WOPBS.get_from_params(params);
    let mut rng = rand::thread_rng();
    let mut cpt = 0;
    let nb_test = 10;

    for _ in 0..nb_test {
        println!("-------------------------------------");

        let clear = rng.gen::<usize>() % 128;
        let mut ct = cks.encrypt_radix(clear as u64, nb_block);
        let lut = wopbs_key.generate_lut_radix(&ct, |x| x);
        //println!("lut : {:?}", lut[0]);
        let ct_res = wopbs_key.wopbs(&ct, &lut);

        let res = cks.decrypt_radix(&ct_res);

        if res != clear as u64 {
            cpt += 1;
        }
    }
    println!("failure rate : {:?} / {:?}", cpt, nb_test);
    assert_eq!(cpt, 0);
}

pub fn wopbs_crt_without_padding(params: (Parameters, Parameters)) {
    let mut rng = rand::thread_rng();
    println!("param : {:?}", params);

    let basis: Vec<u64> = vec![7, 8, 9, 11, 13];

    let nb_block = basis.len();

    let (cks, mut sks) = gen_keys(&params.0);
    let wopbs_key = KEY_CACHE_WOPBS.get_from_params(params);

    let mut msg_space = 1;
    for modulus in basis.iter() {
        msg_space *= modulus;
    }

    let nb_test = 10;

    for _ in 0..nb_test {
        let clear1 = rng.gen::<u64>() % msg_space; // Encrypt the integers
        let mut ct1 = cks.encrypt_crt_not_power_of_two(clear1, basis.clone());

        let lut = wopbs_key.generate_lut_native_crt(&ct1, |x| x);

        let ct_res = wopbs_key.wopbs_not_power_of_two(&ct1, &lut);
        let res = cks.decrypt_crt_not_power_of_two(&ct_res);

        assert_eq!(res, clear1);
    }
}

pub fn wopbs_crt_without_padding_bivariate(params: (Parameters, Parameters)) {
    let mut rng = rand::thread_rng();
    println!("param : {:?}", params);

    let basis: Vec<u64> = vec![9, 11];

    let nb_block = basis.len();

    let (cks, mut sks) = gen_keys(&params.0);
    let wopbs_key = KEY_CACHE_WOPBS.get_from_params(params);

    let mut msg_space = 1;
    for modulus in basis.iter() {
        msg_space *= modulus;
    }

    let nb_test = 10;
    let mut tmp = 0;
    for _ in 0..nb_test {
        let clear1 = rng.gen::<u64>() % msg_space; // Encrypt the integers
        let clear2 = rng.gen::<u64>() % msg_space; // Encrypt the integers
        let mut ct1 = cks.encrypt_crt_not_power_of_two(clear1, basis.clone());
        let mut ct2 = cks.encrypt_crt_not_power_of_two(clear2, basis.clone());

        let lut = wopbs_key.generate_lut_bivariate_native_crt(&ct1, |x, y| x * y);
        let ct_res = wopbs_key.bivariate_wopbs_native_crt(&ct1, &ct2, &lut);
        let res = cks.decrypt_crt_not_power_of_two(&ct_res);

        if (clear1 * clear2) % msg_space != res {
            tmp += 1;
            println!(
                "clear1 {:?}, clear2 {:?}, add {:?}, res {:?}",
                clear1,
                clear2,
                (clear1 * clear2) % msg_space,
                res
            );
        }
    }
    assert_eq!(tmp, 0);
}

// test wopbs fake crt with different degree for each Ct
pub fn wopbs_crt_fake_crt(params: (Parameters, Parameters)) {
    let mut rng = rand::thread_rng();
    println!("param : {:?}", params);

    let basis: Vec<u64> = vec![4, 3];

    let nb_block = basis.len();

    let (cks, mut sks) = gen_keys(&params.0);
    let wopbs_key = KEY_CACHE_WOPBS.get_from_params(params);

    let mut msg_space = 1;
    for modulus in basis.iter() {
        msg_space *= modulus;
    }

    let nb_test = 10;
    let mut tmp = 0;
    for _ in 0..nb_test {
        let clear1 = rng.gen::<u64>() % msg_space;
        let mut ct1 = cks.encrypt_crt(clear1, basis.clone());
        //artificially modify the degree
        for ct in ct1.blocks.iter_mut() {
            let degree = params.0.message_modulus.0
                * ((rng.gen::<usize>() % (params.0.carry_modulus.0 - 1)) + 1);
            ct.degree.0 = degree;
        }
        let lut = wopbs_key.generate_lut_crt(&ct1, |x| (x * x) + x);
        let res = cks.decrypt_crt(&ct1);
        //println!("LUT = {:?}", lut);
        let ct_res = wopbs_key.wopbs(&ct1, &lut);
        let res_wop = cks.decrypt_crt(&ct_res);
        if ((res * res) + res) % msg_space != res_wop {
            tmp += 1;
        }
    }
    if tmp != 0 {
        println!("failure rate {:?}/{:?} ", tmp, nb_test);
        panic!()
    }
}

// test wopbs fake crt with different degree for each Ct
pub fn wopbs_radix(params: (Parameters, Parameters)) {
    let mut rng = rand::thread_rng();
    println!("param : {:?}", params);

    let nb_block = 2;

    let (cks, mut sks) = gen_keys(&params.0);
    let wopbs_key = KEY_CACHE_WOPBS.get_from_params(params);

    let mut msg_space: u64 = params.0.message_modulus.0 as u64;
    for modulus in 1..nb_block {
        msg_space *= params.0.message_modulus.0 as u64;
    }

    let nb_test = 10;
    let mut tmp = 0;
    for _ in 0..nb_test {
        let clear1 = rng.gen::<u64>() % msg_space as u64;
        let mut ct1 = cks.encrypt_radix(clear1, nb_block);

        // //artificially modify the degree
        // let scalar = rng.gen::<u64>() % msg_space as u64;
        // sks.smart_scalar_add_assign(&mut ct1, scalar);

        let lut = wopbs_key.generate_lut_radix(&ct1, |x| x);
        let res = cks.decrypt_radix(&ct1);
        //println!("LUT = {:?}", lut);
        let ct_res = wopbs_key.wopbs(&ct1, &lut);
        let res_wop = cks.decrypt_radix(&ct_res);
        if (res) % msg_space as u64 != res_wop {
            tmp += 1;
        }
    }
    if tmp != 0 {
        println!("failure rate {:?}/{:?} ", tmp, nb_test);
        panic!()
    }
}

// test wopbs radix with different degree for each Ct
pub fn wopbs_bivariate_radix(params: (Parameters, Parameters)) {
    let mut rng = rand::thread_rng();
    println!("param : {:?}", params);

    let nb_block = 2;

    let (cks, mut sks) = gen_keys(&params.0);
    let wopbs_key = KEY_CACHE_WOPBS.get_from_params(params);

    let mut msg_space: u64 = params.0.message_modulus.0 as u64;
    for modulus in 1..nb_block {
        msg_space *= params.0.message_modulus.0 as u64;
    }

    let nb_test = 10;

    for _ in 0..nb_test {
        let mut clear1 = rng.gen::<u64>() % msg_space;
        let mut clear2 = rng.gen::<u64>() % msg_space;

        let mut ct1 = cks.encrypt_radix(clear1, nb_block);
        let scalar = rng.gen::<u64>() % msg_space as u64;
        sks.smart_scalar_add_assign(&mut ct1, scalar);
        let dec1 = cks.decrypt_radix(&ct1);

        let mut ct2 = cks.encrypt_radix(clear2, nb_block);
        let scalar = rng.gen::<u64>() % msg_space as u64;
        sks.smart_scalar_add_assign(&mut ct2, scalar);
        let dec2 = cks.decrypt_radix(&ct2);

        let lut = wopbs_key.generate_lut_bivariate_radix(&ct1, &ct2, |x, y| x + y * x);
        let ct_res = wopbs_key.bivariate_wopbs_with_degree(&ct1, &ct2, &lut);
        let res = cks.decrypt_radix(&ct_res);
        assert_eq!(res, (dec1 + dec2 * dec1) % msg_space);
    }
}

// test wopbs bivariate fake crt with different degree for each Ct
pub fn wopbs_bivariate_fake_crt(params: (Parameters, Parameters)) {
    let mut rng = rand::thread_rng();
    println!("param : {:?}", params);

    let basis = vec![3, 7];

    let (cks, mut sks) = gen_keys(&params.0);
    let wopbs_key = KEY_CACHE_WOPBS.get_from_params(params);

    let mut msg_space: u64 = 1;
    for modulus in basis.iter() {
        msg_space *= modulus;
    }

    let nb_test = 10;

    for _ in 0..nb_test {
        let clear1 = rng.gen::<u64>() % msg_space;
        let clear2 = rng.gen::<u64>() % msg_space;
        let mut ct1 = cks.encrypt_crt(clear1, basis.clone());
        let mut ct2 = cks.encrypt_crt(clear2, basis.clone());
        //artificially modify the degree
        for (ct_1, ct_2) in ct1.blocks.iter_mut().zip(ct2.blocks.iter_mut()) {
            let degree = params.0.message_modulus.0
                * ((rng.gen::<usize>() % (params.0.carry_modulus.0 - 1)) + 1);
            ct_1.degree.0 = degree;
            let degree = params.0.message_modulus.0
                * ((rng.gen::<usize>() % (params.0.carry_modulus.0 - 1)) + 1);
            ct_2.degree.0 = degree;
        }
        let lut = wopbs_key.generate_lut_bivariate_crt(&ct1, &ct2, |x, y| (x * y) + y);

        let ct_res = wopbs_key.bivariate_wopbs_with_degree(&ct1, &ct2, &lut);
        let res = cks.decrypt_crt(&ct_res);
        assert_eq!(res, ((clear1 * clear2) + clear2) % msg_space);
    }
}
