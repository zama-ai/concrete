//!
//! # WARNING: this module is experimental.

#![allow(unused)]
use crate::gen_keys;
use crate::parameters::*;
use crate::wopbs::WopbsKey;
use concrete_shortint::parameters::parameters_wopbs::*;
use concrete_shortint::parameters::parameters_wopbs_message_carry::*;
use concrete_shortint::parameters::{Parameters, *};
use rand::Rng;

use paste::paste;
use crate::keycache::KEY_CACHE;

const NB_TEST: usize = 10;

macro_rules! create_parametrized_test{
    ($name:ident { $($param:ident),* }) => {
        paste! {
            $(
            #[test]
            fn [<test_ $name _ $param:lower>]() {
                $name($param)
            }
            )*
        }
    };
     ($name:ident)=> {
        create_parametrized_test!($name
        {
            WOPBS_PARAM_MESSAGE_1_NORM2_2,
            WOPBS_PARAM_MESSAGE_1_NORM2_4,
            WOPBS_PARAM_MESSAGE_1_NORM2_6,
            WOPBS_PARAM_MESSAGE_1_NORM2_8,
            WOPBS_PARAM_MESSAGE_2_NORM2_2,
            WOPBS_PARAM_MESSAGE_2_NORM2_4,
            WOPBS_PARAM_MESSAGE_2_NORM2_6,
            WOPBS_PARAM_MESSAGE_2_NORM2_8,
            WOPBS_PARAM_MESSAGE_3_NORM2_2,
            WOPBS_PARAM_MESSAGE_3_NORM2_4,
            WOPBS_PARAM_MESSAGE_3_NORM2_6,
            WOPBS_PARAM_MESSAGE_3_NORM2_8,
            WOPBS_PARAM_MESSAGE_4_NORM2_2,
            WOPBS_PARAM_MESSAGE_4_NORM2_4,
            WOPBS_PARAM_MESSAGE_4_NORM2_6,
            WOPBS_PARAM_MESSAGE_4_NORM2_8,
            WOPBS_PARAM_MESSAGE_5_NORM2_2,
            WOPBS_PARAM_MESSAGE_5_NORM2_4,
            WOPBS_PARAM_MESSAGE_5_NORM2_6,
            WOPBS_PARAM_MESSAGE_5_NORM2_8,
            WOPBS_PARAM_MESSAGE_6_NORM2_2,
            WOPBS_PARAM_MESSAGE_6_NORM2_4,
            WOPBS_PARAM_MESSAGE_6_NORM2_6,
            WOPBS_PARAM_MESSAGE_6_NORM2_8,
            WOPBS_PARAM_MESSAGE_7_NORM2_2,
            WOPBS_PARAM_MESSAGE_7_NORM2_4,
            WOPBS_PARAM_MESSAGE_7_NORM2_6,
            WOPBS_PARAM_MESSAGE_7_NORM2_8,
            WOPBS_PARAM_MESSAGE_8_NORM2_2,
            WOPBS_PARAM_MESSAGE_8_NORM2_4,
            WOPBS_PARAM_MESSAGE_8_NORM2_6,
            WOPBS_PARAM_MESSAGE_1_CARRY_0,
            WOPBS_PARAM_MESSAGE_1_CARRY_1,
            WOPBS_PARAM_MESSAGE_1_CARRY_2,
            WOPBS_PARAM_MESSAGE_1_CARRY_3,
            WOPBS_PARAM_MESSAGE_1_CARRY_4,
            WOPBS_PARAM_MESSAGE_1_CARRY_5,
            WOPBS_PARAM_MESSAGE_1_CARRY_6,
            WOPBS_PARAM_MESSAGE_1_CARRY_7,
            WOPBS_PARAM_MESSAGE_2_CARRY_0,
            WOPBS_PARAM_MESSAGE_2_CARRY_1,
            WOPBS_PARAM_MESSAGE_2_CARRY_2,
            WOPBS_PARAM_MESSAGE_2_CARRY_3,
            WOPBS_PARAM_MESSAGE_2_CARRY_4,
            WOPBS_PARAM_MESSAGE_2_CARRY_5,
            WOPBS_PARAM_MESSAGE_2_CARRY_6,
            WOPBS_PARAM_MESSAGE_3_CARRY_0,
            WOPBS_PARAM_MESSAGE_3_CARRY_1,
            WOPBS_PARAM_MESSAGE_3_CARRY_2,
            WOPBS_PARAM_MESSAGE_3_CARRY_3,
            WOPBS_PARAM_MESSAGE_3_CARRY_4,
            WOPBS_PARAM_MESSAGE_3_CARRY_5,
            WOPBS_PARAM_MESSAGE_4_CARRY_0,
            WOPBS_PARAM_MESSAGE_4_CARRY_1,
            WOPBS_PARAM_MESSAGE_4_CARRY_2,
            WOPBS_PARAM_MESSAGE_4_CARRY_3,
            WOPBS_PARAM_MESSAGE_4_CARRY_4,
            WOPBS_PARAM_MESSAGE_5_CARRY_0,
            WOPBS_PARAM_MESSAGE_5_CARRY_1,
            WOPBS_PARAM_MESSAGE_5_CARRY_2,
            WOPBS_PARAM_MESSAGE_5_CARRY_3,
            WOPBS_PARAM_MESSAGE_6_CARRY_0,
            WOPBS_PARAM_MESSAGE_6_CARRY_1,
            WOPBS_PARAM_MESSAGE_6_CARRY_2,
            WOPBS_PARAM_MESSAGE_7_CARRY_0,
            WOPBS_PARAM_MESSAGE_7_CARRY_1,
            WOPBS_PARAM_MESSAGE_8_CARRY_0,
            PARAM_MESSAGE_4_CARRY_4_16_BITS,
            PARAM_MESSAGE_2_CARRY_2_16_BITS,
            PARAM_4_BITS_5_BLOCKS
        });
    };
}
//create_parametrized_test!(wopbs_v_init);
//create_parametrized_test!(wopbs_v0_16_bits);
// create_parametrized_test!(wopbs_v1);
create_parametrized_test!(wopbs_16_lut_test);
create_parametrized_test!(wopbs_crt_without_padding);

pub fn wopbs(param: Parameters) {
    //Generate the client key and the server key:
    let (cks, mut sks) = gen_keys(&param);

    // //Generate wopbs_v0 key
    let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    let mut rng = rand::thread_rng();
    let mut cpt = 0;
    let nb_test = 1;

    for _ in 0..nb_test {
        println!("-------------------------------------");

        let clear = rng.gen::<usize>() % 8;
        let mut ct = cks.encrypt_radix(clear as u64, 8);
        let delta = 63 - f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as u64;

        let nb_bit_to_extract =
            f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as usize * 2;

        let mut lut_size = param.polynomial_size.0;
        if (1 << nb_bit_to_extract) > param.polynomial_size.0 {
            lut_size = 1 << nb_bit_to_extract;
        }

        let mut lut_1: Vec<u64> = vec![];
        for _ in 0..lut_size {
            lut_1.push(1 << delta);
        }
        let big_lut = vec![lut_1; 8];

        let ct_res = wopbs_key.wopbs(&sks, &mut ct, &big_lut);

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

pub fn wopbs_16_bits(param: Parameters) {
    let nb_block = 8;

    //Generate the client key and the server key:
    let (cks, sks) = gen_keys(&param);
    //
    // //Generate wopbs_v0 key
    let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    let mut rng = rand::thread_rng();
    let mut cpt = 0;
    let nb_test = 100;

    for _ in 0..nb_test {
        println!("-------------------------------------");

        let clear = rng.gen::<usize>() % 8;
        let mut ct = cks.encrypt_radix(clear as u64, 8);
        let delta = 63 - f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as u64;

        let nb_bit_to_extract =
            f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as usize * nb_block;

        let mut lut_size = param.polynomial_size.0;
        if (1 << nb_bit_to_extract) > param.polynomial_size.0 {
            lut_size = 1 << nb_bit_to_extract;
        }

        let mut lut_1: Vec<u64> = vec![];
        let mut lut_2: Vec<u64> = vec![];
        for _ in 0..lut_size {
            lut_1.push(
                (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64)
                    << delta,
            );
            lut_2.push(
                (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64)
                    << delta,
            );
        }

        let lut_res_1 = lut_1.clone();
        let lut_res_2 = lut_2.clone();

        let ct_res =
            wopbs_key.wopbs(&sks, &mut ct, &[lut_1, lut_2]);

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

pub fn wopbs_16_lut_test(param: Parameters) {
    let nb_block = 2;
    //Generate the client key and the server key:
    let (cks, sks) = gen_keys(&param);
    //
    // //Generate wopbs_v0 key
    let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    let mut rng = rand::thread_rng();
    let mut cpt = 0;
    let nb_test = 10;

    for _ in 0..nb_test {
        println!("-------------------------------------");

        let clear = rng.gen::<usize>() % 128;
        let mut ct = cks.encrypt_radix(clear as u64, nb_block);
        let lut = wopbs_key.generate_lut(&ct, |x|x);
        //println!("lut : {:?}", lut[0]);
        let ct_res = wopbs_key.wopbs(&sks, &mut ct, &lut);

        let res = cks.decrypt_radix(&ct_res);

        if res != clear as u64 {
            println!("clear {:?}",clear);
            println!("res {:?}\n", res);
            cpt += 1;
        }
    }
    println!("failure rate : {:?} / {:?}", cpt , nb_test);
    panic!();
}

pub fn wopbs_crt_without_padding(param: Parameters){

    let mut rng = rand::thread_rng();
    println!("param : {:?}", param);

    let basis : Vec<u64> = vec![9,11,13];

    let nb_block = basis.len();

    //Generate the client key and the server key:
    let (mut cks, sks) = gen_keys(&param);
    // let (mut cks, mut sks) = KEY_CACHE.get_from_params(param, VecLength(nb_block));
    let wopbs_key =  WopbsKey::new_wopbs_key(&cks, &sks);

    let mut msg_space = 1;
    for modulus in basis.iter() {
        msg_space *= modulus;
    }

    let nb_test = 10;

    for _ in 0..nb_test {
        let clear1 = rng.gen::<u64>() % msg_space;    // Encrypt the integers
        let mut ct1 = cks.encrypt_crt_not_power_of_two(clear1, basis.clone());

        let lut = wopbs_key.generate_lut_without_padding(&ct1, |x| x);

        let ct_res = wopbs_key.circuit_bootstrap_vertical_packing_v0_without_padding(&sks, &mut ct1, &lut);
        let res = cks.decrypt_crt_not_power_of_two(&ct_res);
        assert_eq!(res, clear1);
    }
}