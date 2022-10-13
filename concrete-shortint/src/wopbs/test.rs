use crate::keycache::KEY_CACHE_WOPBS;
use crate::parameters::parameters_wopbs::*;
use crate::parameters::parameters_wopbs_message_carry::*;
use crate::parameters::parameters_wopbs_prime_moduli::*;
use crate::parameters::{MessageModulus, PARAM_MESSAGE_2_CARRY_2};
use crate::wopbs::WopbsKey;
use crate::{gen_keys, Parameters};
use concrete_core::prelude::{LweBootstrapKeyEntity, LweCiphertextEntity, LweKeyswitchKeyEntity};
use paste::paste;
use rand::Rng;

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
create_parametrized_test!(wopbs_v0);
create_parametrized_test!(wopbs_v0_norm2);
create_parametrized_test!(generate_lut);
create_parametrized_test!(generate_lut_modulus);
create_parametrized_test!(generate_lut_modulus_not_power_of_two);

fn wopbs_v0(params: (Parameters, Parameters)) {
    let keys = KEY_CACHE_WOPBS.get_from_param(params);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());

    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        let clear = rng.gen::<usize>() % params.0.message_modulus.0;
        let ct = cks.unchecked_encrypt(clear as u64);
        assert_eq!(clear as u64, cks.decrypt_message_and_carry(&ct));

        let mut lut: Vec<u64> = vec![];
        let delta =
            63 - f64::log2((params.0.message_modulus.0 * params.0.carry_modulus.0) as f64) as u64;

        for _ in 0..wopbs_key.param.polynomial_size.0 {
            lut.push(
                (rng.gen::<u64>() % (params.0.message_modulus.0 * params.0.carry_modulus.0) as u64)
                    << delta,
            );
        }
        let lut_res = lut.clone();

        let ct_res = wopbs_key.programmable_bootstrapping(&ct, &lut);
        let res = cks.decrypt_message_and_carry(&ct_res);
        assert_eq!(res, lut_res[clear] / (1 << delta));
    }
}

fn wopbs_v0_norm2(params: (Parameters, Parameters)) {
    let keys = KEY_CACHE_WOPBS.get_from_param(params);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());

    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        let clear = rng.gen::<usize>() % params.0.message_modulus.0;
        let mut ct = cks.encrypt_without_padding(clear as u64);

        let mut lut: Vec<u64> = vec![];
        let delta =
            64 - f64::log2((params.0.message_modulus.0 * params.0.carry_modulus.0) as f64) as u64;

        for _ in 0..wopbs_key.param.polynomial_size.0 {
            lut.push(
                (rng.gen::<u64>() % (params.0.message_modulus.0 * params.0.carry_modulus.0) as u64)
                    << delta,
            );
        }

        let lut_res = lut.clone();
        let vec_lut = lut;
        let ct_res = wopbs_key.programmable_bootstrapping_native_crt(&mut ct, &vec_lut);
        let res = cks.decrypt_message_and_carry_without_padding(&ct_res);
        assert_eq!(res, lut_res[clear] / (1 << delta));
    }
}

fn generate_lut(params: (Parameters, Parameters)) {
    let keys = KEY_CACHE_WOPBS.get_from_param(params);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());
    let mut rng = rand::thread_rng();

    println!(
        "lwe out wop {:?}",
        wopbs_key
            .wopbs_server_key
            .bootstrapping_key
            .output_lwe_dimension()
    );
    println!(
        "lwe in wop  {:?}",
        wopbs_key
            .wopbs_server_key
            .bootstrapping_key
            .input_lwe_dimension()
    );
    println!(
        "lwe out pbs {:?}",
        wopbs_key
            .pbs_server_key
            .bootstrapping_key
            .output_lwe_dimension()
    );
    println!(
        "lwe in pbs  {:?}",
        wopbs_key
            .pbs_server_key
            .bootstrapping_key
            .input_lwe_dimension()
    );
    println!("-----");
    println!(
        "lwe out wop ks {:?}",
        wopbs_key
            .wopbs_server_key
            .key_switching_key
            .output_lwe_dimension()
    );
    println!(
        "lwe in wop  ks {:?}",
        wopbs_key
            .wopbs_server_key
            .key_switching_key
            .input_lwe_dimension()
    );
    println!(
        "lwe out pbs ks {:?}",
        wopbs_key
            .pbs_server_key
            .key_switching_key
            .output_lwe_dimension()
    );
    println!(
        "lwe in pbs  ks {:?}",
        wopbs_key
            .pbs_server_key
            .key_switching_key
            .input_lwe_dimension()
    );
    println!(
        "lwe out ks     {:?}",
        wopbs_key.ksk_pbs_to_wopbs.output_lwe_dimension()
    );
    println!(
        "lwe in  ks     {:?}",
        wopbs_key.ksk_pbs_to_wopbs.input_lwe_dimension()
    );
    println!("______");

    let mut tmp = 0;
    for _ in 0..NB_TEST {
        let message_modulus = params.0.message_modulus.0;
        let m = rng.gen::<usize>() % message_modulus;
        println!("m:  {:?}", m);

        let ct = cks.encrypt(m as u64);
        println!("ct lwe dim {:?}", ct.ct.lwe_dimension());
        let lut = wopbs_key.generate_lut(&ct, |x| x % message_modulus as u64);
        let ct_res = wopbs_key.programmable_bootstrapping(&ct, &lut);

        // let mut res = cks.decrypt(&ct_res);
        // assert_eq!(res, ((m * m) % message_modulus) as u64);

        // println!("JUST BEFORE THE PBS");
        // println!("ct in dim = {}", ct_res.ct.lwe_dimension().0);
        // println!("KSK in:  = {}, KSK out:  = {}", wopbs_key.pbs_server_key.key_switching_key
        //     .input_lwe_dimension().0, wopbs_key.pbs_server_key.key_switching_key
        //     .output_lwe_dimension().0);
        //  println!("BSK in:  = {:?}, BSK out:  = {:?}",
        //      wopbs_key.pbs_server_key.bootstrapping_key.input_lwe_dimension(),
        //      wopbs_key.pbs_server_key.bootstrapping_key.output_lwe_dimension());

        // let acc = wopbs_key.pbs_server_key.generate_accumulator(|x| x);
        //
        // let ct_res = wopbs_key.pbs_server_key.keyswitch_programmable_bootstrap(&ct_res, &acc);

        let mut res = cks.decrypt(&ct_res);
        if res != (m % message_modulus) as u64 {
            tmp += 1;
        }
    }
    //assert!(false);
    if 0 != tmp {
        println!("______");
        println!("failure rate {:?}/{:?}", tmp, NB_TEST);
        println!("______");
    }
    assert_eq!(0, tmp);
}

fn generate_lut_modulus(params: (Parameters, Parameters)) {
    let keys = KEY_CACHE_WOPBS.get_from_param(params);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());
    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        let message_modulus = MessageModulus(params.0.message_modulus.0 - 1);
        let m = rng.gen::<usize>() % message_modulus.0;

        let ct = cks.encrypt_with_message_modulus(m as u64, message_modulus);

        let lut = wopbs_key.generate_lut(&ct, |x| (x * x) % message_modulus.0 as u64);
        let ct_res = wopbs_key.programmable_bootstrapping(&ct, &lut);

        let res = cks.decrypt(&ct_res);
        assert_eq!(res as usize, (m * m) % message_modulus.0);
    }
}

fn generate_lut_modulus_not_power_of_two(params: (Parameters, Parameters)) {
    let keys = KEY_CACHE_WOPBS.get_from_param(params);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());
    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        // let mut message_modulus = MessageModulus(rng.gen::<usize>() %
        // params.0.message_modulus.0); while(message_modulus.0 == 0) || (message_modulus.0
        // == 1) {     message_modulus = MessageModulus(rng.gen::<usize>() %
        // params.0.message_modulus.0); }
        let message_modulus = MessageModulus(params.0.message_modulus.0 - 1);

        let m = rng.gen::<usize>() % message_modulus.0;
        let mut ct =
            cks.encrypt_with_message_modulus_not_power_of_two(m as u64, message_modulus.0 as u8);
        let lut = wopbs_key.generate_lut_native_crt(&ct, |x| (x * x) % message_modulus.0 as u64);

        let ct_res = wopbs_key.programmable_bootstrapping_native_crt(&mut ct, &lut);
        let res = cks.decrypt_message_and_carry_not_power_of_two(&ct_res, message_modulus.0 as u8);
        println!("m = {}, mod = {}, lut = {:?}", m, message_modulus.0, lut);
        assert_eq!(res as usize, (m * m) % message_modulus.0);
    }
}
