use crate::keycache::{KEY_CACHE, KEY_CACHE_TREEPBS};
use concrete_shortint::parameters::*;
use concrete_shortint::Parameters;

use rand::Rng;

create_parametrized_test!(integer_two_block_pbs);
create_parametrized_test!(integer_two_block_pbs_base);
create_parametrized_test!(integer_three_block_pbs);
create_parametrized_test!(integer_three_block_pbs_base);

/// Smaller number of loop iteration within randomized test,
/// meant for test where the function tested is more expensive
const NB_TEST_SMALLER: usize = 10;
const NB_CTXT: usize = 4;

fn integer_two_block_pbs(param: Parameters) {
    let (cks, sks) = KEY_CACHE.get_from_params(param);

    // RNG
    let mut rng = rand::thread_rng();

    // message_modulus^vec_length
    let modulus = param.message_modulus.0.pow(2) as u64;
    // println!("modulus = {}", modulus);

    let treepbs_key = KEY_CACHE_TREEPBS.get_from_params(param);

    for _ in 0..NB_TEST_SMALLER {
        let clear_0 = rng.gen::<u64>() % modulus;

        // encryption of an integer
        let ctxt_0 = cks.encrypt_radix(clear_0, NB_CTXT);

        let f = |x: u64| x * x;

        // multiply together the two ciphertexts
        let vec_res = treepbs_key.two_block_pbs(&sks, &ctxt_0, f);

        // decryption
        let res = cks.decrypt_radix(&vec_res);

        let clear = (clear_0 * clear_0) % modulus;
        // println!(
        //     "clear = {}, f(clear) = {}, res = {}",
        //     clear_0,
        //     f(clear_0) % modulus,
        //     res
        // );
        // assert
        assert_eq!(res, clear);
    }
}

fn integer_two_block_pbs_base(param: Parameters) {
    let (cks, sks) = KEY_CACHE.get_from_params(param);

    //RNG
    let mut rng = rand::thread_rng();

    // message_modulus^vec_length
    let modulus = param.message_modulus.0.pow(2) as u64;
    // println!("modulus = {}", modulus);

    let treepbs_key = KEY_CACHE_TREEPBS.get_from_params(param);

    for _ in 0..NB_TEST_SMALLER {
        let clear_0 = rng.gen::<u64>() % modulus;

        // encryption of an integer
        let ctxt_0 = cks.encrypt_radix(clear_0, NB_CTXT);

        let f = |x: u64| x * x;

        // multiply together the two ciphertexts
        let vec_res = treepbs_key.two_block_pbs_base(&sks, &ctxt_0, f);

        // decryption
        let res = cks.decrypt_radix(&vec_res);

        let clear = (clear_0 * clear_0) % modulus;
        // println!(
        //     "clear = {}, f(clear) = {}, res = {}",
        //     clear_0,
        //     f(clear_0) % modulus,
        //     res
        // );
        // assert
        assert_eq!(res, clear);
    }
}

fn integer_three_block_pbs(param: Parameters) {
    let (cks, sks) = KEY_CACHE.get_from_params(param);

    //RNG
    let mut rng = rand::thread_rng();

    // message_modulus^vec_length
    let modulus = param.message_modulus.0.pow(3) as u64;
    // println!("modulus = {}", modulus);

    let treepbs_key = KEY_CACHE_TREEPBS.get_from_params(param);

    for _ in 0..NB_TEST_SMALLER {
        let clear_0 = rng.gen::<u64>() % modulus;

        // encryption of an integer
        let ctxt_0 = cks.encrypt_radix(clear_0, NB_CTXT);

        let f = |x: u64| x * x;

        // multiply together the two ciphertexts
        let vec_res = treepbs_key.three_block_pbs(&sks, &ctxt_0, f);

        // decryption
        let res = cks.decrypt_radix(&vec_res);

        let clear = (clear_0 * clear_0) % modulus;
        // println!(
        //     "clear = {}, f(clear) = {}, res = {}",
        //     clear_0,
        //     f(clear_0) % modulus,
        //     res
        // );
        // assert
        assert_eq!(res, clear);
    }
}

fn integer_three_block_pbs_base(param: Parameters) {
    let (cks, sks) = KEY_CACHE.get_from_params(param);

    //RNG
    let mut rng = rand::thread_rng();

    // message_modulus^vec_length
    let modulus = param.message_modulus.0.pow(3) as u64;
    // println!("modulus = {}", modulus);

    let treepbs_key = KEY_CACHE_TREEPBS.get_from_params(param);

    for _ in 0..NB_TEST_SMALLER {
        let clear_0 = rng.gen::<u64>() % modulus;

        // encryption of an integer
        let ctxt_0 = cks.encrypt_radix(clear_0, NB_CTXT);

        let f = |x: u64| x * x;

        // multiply together the two ciphertexts
        let vec_res = treepbs_key.three_block_pbs_base(&sks, &ctxt_0, f);

        // decryption
        let res = cks.decrypt_radix(&vec_res);

        let clear = (clear_0 * clear_0) % modulus;
        // println!(
        //     "clear = {}, f(clear) = {}, res = {}",
        //     clear_0,
        //     f(clear_0) % modulus,
        //     res
        // );
        // assert
        assert_eq!(res, clear);
    }
}
