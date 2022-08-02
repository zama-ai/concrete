use crate::keycache::KEY_CACHE_WOPBS;
use crate::parameters::parameters_wopbs::*;
use crate::parameters::parameters_wopbs_message_carry::*;
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
            WOPBS_PARAM_MESSAGE_8_CARRY_0
        });
    };
}
create_parametrized_test!(wopbs_v0);
create_parametrized_test!(wopbs_v0_norm2);

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
        let ct_res = wopbs_key.circuit_bootstrap_vertical_packing(sks, &mut ct, &[lut; 1]);
        let res = cks.decrypt_message_and_carry(&ct_res[0]);
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
        let vec_lut = vec![lut; 1];
        let ct_res =
            wopbs_key.circuit_bootstrap_vertical_packing_without_padding(sks, &mut ct, &vec_lut);
        let res = cks.decrypt_message_and_carry_without_padding(&ct_res[0]);
        assert_eq!(res, lut_res[clear] / (1 << delta));
    }
}
