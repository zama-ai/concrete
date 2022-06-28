use crate::keycache::KEY_CACHE;
use crate::parameters::*;
use crate::treepbs::TreepbsKey;
use paste::paste;
use rand::Rng;

/// Number of assert in randomized tests
const NB_TEST: usize = 30;

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
            PARAM_MESSAGE_1_CARRY_1,
            PARAM_MESSAGE_1_CARRY_2,
            PARAM_MESSAGE_1_CARRY_3,
            PARAM_MESSAGE_1_CARRY_4,
            PARAM_MESSAGE_1_CARRY_5,
            PARAM_MESSAGE_1_CARRY_6,
            PARAM_MESSAGE_1_CARRY_7,
            PARAM_MESSAGE_2_CARRY_1,
            PARAM_MESSAGE_2_CARRY_2,
            PARAM_MESSAGE_2_CARRY_3,
            PARAM_MESSAGE_2_CARRY_4,
            PARAM_MESSAGE_2_CARRY_5,
            PARAM_MESSAGE_2_CARRY_6,
            PARAM_MESSAGE_3_CARRY_1,
            PARAM_MESSAGE_3_CARRY_2,
            PARAM_MESSAGE_3_CARRY_3,
            PARAM_MESSAGE_3_CARRY_4,
            PARAM_MESSAGE_3_CARRY_5,
            PARAM_MESSAGE_4_CARRY_1,
            PARAM_MESSAGE_4_CARRY_2,
            PARAM_MESSAGE_4_CARRY_3,
            PARAM_MESSAGE_4_CARRY_4,
            PARAM_MESSAGE_5_CARRY_1,
            PARAM_MESSAGE_5_CARRY_2,
            PARAM_MESSAGE_5_CARRY_3,
            PARAM_MESSAGE_6_CARRY_1,
            PARAM_MESSAGE_6_CARRY_2,
            PARAM_MESSAGE_7_CARRY_1
        });
    };
}

create_parametrized_test!(shortint_mul_treepbs);
create_parametrized_test!(shortint_message_and_carry_extract);

fn shortint_mul_treepbs(param: Parameters) {
    let (cks, sks) = KEY_CACHE.get_from_param(param);
    let mut treepbs_key = TreepbsKey::new_tree_key(&cks);

    //RNG
    let mut rng = rand::thread_rng();

    let base = cks.parameters.message_modulus.0 as u64;
    let modulus = (sks.message_modulus.0 * sks.carry_modulus.0) as u64;

    for _ in 0..NB_TEST {
        let clear_0 = rng.gen::<u64>() % base;

        let clear_1 = rng.gen::<u64>() % base;

        // encryption of an integer
        let ctxt_zero = cks.encrypt(clear_0);

        // encryption of an integer
        let ctxt_one = cks.encrypt(clear_1);

        // multiply together the two ciphertexts
        let ct_res = treepbs_key.mul_treepbs_with_multivalue(&sks, &ctxt_zero, &ctxt_one);

        // decryption of ct_res
        let dec_res = cks.decrypt_message_and_carry(&ct_res);

        // assert
        assert_eq!((clear_0 * clear_1) % modulus, dec_res);
    }
}

fn shortint_message_and_carry_extract(param: Parameters) {
    let (cks, sks) = KEY_CACHE.get_from_param(param);
    let mut treepbs_key = TreepbsKey::new_tree_key(&cks);
    //RNG
    let mut rng = rand::thread_rng();

    let base = cks.parameters.message_modulus.0 as u64;

    for _ in 0..NB_TEST {
        let clear_0 = rng.gen::<u64>() % base;

        // encryption of an integer
        let ctxt_zero = cks.encrypt(clear_0);

        // multiply together the two ciphertexts
        let vec_res = treepbs_key.message_and_carry_extract(&sks, &ctxt_zero);

        // decryption
        let res_1 = cks.decrypt(&vec_res[0]);

        // assert
        assert_eq!(clear_0 % base, res_1);

        // decryption
        let res_2 = cks.decrypt(&vec_res[1]);

        // assert
        assert_eq!(clear_0 / base, res_2);
    }
}
