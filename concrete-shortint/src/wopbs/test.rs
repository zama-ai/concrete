use crate::keycache::KEY_CACHE_WOPBS;
use crate::parameters::parameters_wopbs::*;
use crate::parameters::parameters_wopbs_message_carry::*;
use crate::parameters::parameters_wopbs_prime_moduli::*;
use crate::parameters::MessageModulus;
use crate::Parameters;
use paste::paste;
use rand::Rng;

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
            WOPBS_PARAM_MESSAGE_8_NORM2_5,
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
            WOPBS_PRIME_PARAM_MESSAGE_2_NORM2_2,
            WOPBS_PRIME_PARAM_MESSAGE_2_NORM2_3,
            WOPBS_PRIME_PARAM_MESSAGE_2_NORM2_4,
            WOPBS_PRIME_PARAM_MESSAGE_2_NORM2_5,
            WOPBS_PRIME_PARAM_MESSAGE_2_NORM2_6,
            WOPBS_PRIME_PARAM_MESSAGE_2_NORM2_7,
            WOPBS_PRIME_PARAM_MESSAGE_2_NORM2_8,
            WOPBS_PRIME_PARAM_MESSAGE_3_NORM2_2,
            WOPBS_PRIME_PARAM_MESSAGE_3_NORM2_3,
            WOPBS_PRIME_PARAM_MESSAGE_3_NORM2_4,
            WOPBS_PRIME_PARAM_MESSAGE_3_NORM2_5,
            WOPBS_PRIME_PARAM_MESSAGE_3_NORM2_6,
            WOPBS_PRIME_PARAM_MESSAGE_3_NORM2_7,
            WOPBS_PRIME_PARAM_MESSAGE_3_NORM2_8,
            WOPBS_PRIME_PARAM_MESSAGE_4_NORM2_2,
            WOPBS_PRIME_PARAM_MESSAGE_4_NORM2_3,
            WOPBS_PRIME_PARAM_MESSAGE_4_NORM2_4,
            WOPBS_PRIME_PARAM_MESSAGE_4_NORM2_5,
            WOPBS_PRIME_PARAM_MESSAGE_4_NORM2_6,
            WOPBS_PRIME_PARAM_MESSAGE_4_NORM2_7,
            WOPBS_PRIME_PARAM_MESSAGE_4_NORM2_8,
            WOPBS_PRIME_PARAM_MESSAGE_5_NORM2_2,
            WOPBS_PRIME_PARAM_MESSAGE_5_NORM2_3,
            WOPBS_PRIME_PARAM_MESSAGE_5_NORM2_4,
            WOPBS_PRIME_PARAM_MESSAGE_5_NORM2_5,
            WOPBS_PRIME_PARAM_MESSAGE_5_NORM2_6,
            WOPBS_PRIME_PARAM_MESSAGE_5_NORM2_7,
            WOPBS_PRIME_PARAM_MESSAGE_5_NORM2_8,
            WOPBS_PRIME_PARAM_MESSAGE_6_NORM2_2,
            WOPBS_PRIME_PARAM_MESSAGE_6_NORM2_3,
            WOPBS_PRIME_PARAM_MESSAGE_6_NORM2_4,
            WOPBS_PRIME_PARAM_MESSAGE_6_NORM2_5,
            WOPBS_PRIME_PARAM_MESSAGE_6_NORM2_6,
            WOPBS_PRIME_PARAM_MESSAGE_6_NORM2_7,
            WOPBS_PRIME_PARAM_MESSAGE_6_NORM2_8,
            WOPBS_PRIME_PARAM_MESSAGE_7_NORM2_2,
            WOPBS_PRIME_PARAM_MESSAGE_7_NORM2_3,
            WOPBS_PRIME_PARAM_MESSAGE_7_NORM2_4,
            WOPBS_PRIME_PARAM_MESSAGE_7_NORM2_5,
            WOPBS_PRIME_PARAM_MESSAGE_7_NORM2_6,
            WOPBS_PRIME_PARAM_MESSAGE_7_NORM2_7,
            WOPBS_PRIME_PARAM_MESSAGE_7_NORM2_8,
            WOPBS_PRIME_PARAM_MESSAGE_8_NORM2_2,
            WOPBS_PRIME_PARAM_MESSAGE_8_NORM2_3,
            WOPBS_PRIME_PARAM_MESSAGE_8_NORM2_4,
            WOPBS_PRIME_PARAM_MESSAGE_8_NORM2_5,
            WOPBS_PRIME_PARAM_MESSAGE_8_NORM2_6,
            WOPBS_PRIME_PARAM_MESSAGE_8_NORM2_7,
            WOPBS_PRIME_PARAM_MESSAGE_8_NORM2_8,
            //TODO: REMOVE NAWAK
            PARAM_NAWAK
        });
    };
}
create_parametrized_test!(wopbs_v0);
create_parametrized_test!(wopbs_v0_norm2);
create_parametrized_test!(generate_lut);
create_parametrized_test!(generate_lut_modulus);
create_parametrized_test!(generate_lut_modulus_not_power_of_two);

fn wopbs_v0(param: Parameters) {
    let keys = KEY_CACHE_WOPBS.get_from_param(param);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());

    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        let clear = rng.gen::<usize>() % param.message_modulus.0;
        let mut ct = cks.unchecked_encrypt(clear as u64);
        assert_eq!(clear as u64, cks.decrypt_message_and_carry(&ct));

        let mut lut: Vec<u64> = vec![];
        let delta = 63 - f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as u64;

        for _ in 0..wopbs_key.param.polynomial_size.0 {
            lut.push(
                (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64)
                    << delta,
            );
        }
        let lut_res = lut.clone();

        let ct_res = wopbs_key.programmable_bootstrapping(&sks, &mut ct, &lut);
        let res = cks.decrypt_message_and_carry(&ct_res);
        assert_eq!(res, lut_res[clear] / (1 << delta));
    }
}

fn wopbs_v0_norm2(param: Parameters) {
    let keys = KEY_CACHE_WOPBS.get_from_param(param);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());

    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        let clear = rng.gen::<usize>() % param.message_modulus.0;
        let mut ct = cks.encrypt_without_padding(clear as u64);

        let mut lut: Vec<u64> = vec![];
        let delta = 64 - f64::log2((param.message_modulus.0 * param.carry_modulus.0) as f64) as u64;

        for _ in 0..wopbs_key.param.polynomial_size.0 {
            lut.push(
                (rng.gen::<u64>() % (param.message_modulus.0 * param.carry_modulus.0) as u64)
                    << delta,
            );
        }

        let lut_res = lut.clone();
        let vec_lut = lut;
        let ct_res = wopbs_key.programmable_bootstrapping_native_crt(&sks, &mut ct, &vec_lut);
        let res = cks.decrypt_message_and_carry_without_padding(&ct_res);
        assert_eq!(res, lut_res[clear] / (1 << delta));
    }
}

fn generate_lut(param: Parameters) {
    let keys = KEY_CACHE_WOPBS.get_from_param(param);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());
    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        let message_modulus = param.message_modulus.0;
        let m = rng.gen::<usize>() % message_modulus;

        let mut ct = cks.encrypt(m as u64);

        let lut = wopbs_key.generate_lut(&ct, |x| (x * x) % message_modulus as u64);
        let ct_res = wopbs_key.programmable_bootstrapping(&sks, &mut ct, &lut);

        let res = cks.decrypt(&ct_res);
        assert_eq!(res, ((m * m) % message_modulus) as u64);
    }
}

fn generate_lut_modulus(param: Parameters) {
    let keys = KEY_CACHE_WOPBS.get_from_param(param);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());
    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        let message_modulus = MessageModulus(param.message_modulus.0 - 1);
        let m = rng.gen::<usize>() % message_modulus.0;

        let mut ct = cks.encrypt_with_message_modulus(m as u64, message_modulus);

        let lut = wopbs_key.generate_lut(&ct, |x| (x * x) % message_modulus.0 as u64);
        let ct_res = wopbs_key.programmable_bootstrapping(&sks, &mut ct, &lut);

        let res = cks.decrypt(&ct_res);
        assert_eq!(res as usize, (m * m) % message_modulus.0);
    }
}

fn generate_lut_modulus_not_power_of_two(param: Parameters) {
    let keys = KEY_CACHE_WOPBS.get_from_param(param);
    let (cks, sks, wopbs_key) = (keys.client_key(), keys.server_key(), keys.wopbs_key());
    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        // let mut message_modulus = MessageModulus(rng.gen::<usize>() % param.message_modulus.0);
        // while(message_modulus.0 == 0) || (message_modulus.0 == 1) {
        //     message_modulus = MessageModulus(rng.gen::<usize>() % param.message_modulus.0);
        // }
        let message_modulus = MessageModulus(param.message_modulus.0 - 1);

        let m = rng.gen::<usize>() % message_modulus.0;
        let mut ct =
            cks.encrypt_with_message_modulus_not_power_of_two(m as u64, message_modulus.0 as u8);
        let lut = wopbs_key.generate_lut_native_crt(&ct, |x| (x * x) % message_modulus.0 as u64);

        let ct_res = wopbs_key.programmable_bootstrapping_native_crt(&sks, &mut ct, &lut);
        let res = cks.decrypt_message_and_carry_not_power_of_two(&ct_res, message_modulus.0 as u8);
        println!("m = {}, mod = {}, lut = {:?}", m, message_modulus.0, lut);
        assert_eq!(res as usize, (m * m) % message_modulus.0);
    }
}
