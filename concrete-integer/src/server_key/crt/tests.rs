use crate::keycache::KEY_CACHE;
use concrete_shortint::parameters::*;
use concrete_shortint::Parameters;
use rand::Rng;

create_parametrized_test!(integer_unchecked_mul_crt);
create_parametrized_test!(integer_smart_add_crt);
create_parametrized_test!(integer_unchecked_add_crt);
create_parametrized_test!(integer_smart_mul_crt);

/// Number of loop iteration within randomized tests
const NB_TEST: usize = 30;

/// Smaller number of loop iteration within randomized test,
/// meant for test where the function tested is more expensive
const NB_TEST_SMALLER: usize = 10;

fn make_basis(message_modulus: usize) -> Vec<u64> {
    match message_modulus {
        n if n < 3 => vec![],
        3 => vec![2],
        n if n < 8 => vec![2, 3],
        n if n < 16 => vec![2, 5, 7],
        _ => vec![3, 7, 13],
    }
}

fn integer_unchecked_mul_crt(param: Parameters) {
    // generate the server-client key set
    let (cks, sks) = KEY_CACHE.get_from_params(param);

    //RNG
    let mut rng = rand::thread_rng();

    // Define CRT basis, and global modulus
    let basis = make_basis(param.message_modulus.0);
    let modulus = basis.iter().product::<u64>();

    for _ in 0..NB_TEST {
        let clear_0 = rng.gen::<u64>() % modulus;
        let clear_1 = rng.gen::<u64>() % modulus;

        // encryption of an integer
        let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());
        let ct_one = cks.encrypt_crt(clear_1, basis.clone());

        // add the two ciphertexts
        sks.unchecked_mul_crt_assign(&mut ct_zero, &ct_one);

        // decryption of ct_res
        let dec_res = cks.decrypt_crt(&ct_zero);

        // assert
        assert_eq!((clear_0 * clear_1) % modulus, dec_res % modulus);
    }
}

fn integer_unchecked_add_crt(param: Parameters) {
    // generate the server-client key set
    let (cks, sks) = KEY_CACHE.get_from_params(param);

    // Define CRT basis, and global modulus
    let basis = make_basis(param.message_modulus.0);
    let modulus = basis.iter().product::<u64>();

    //RNG
    let mut rng = rand::thread_rng();

    for _ in 0..NB_TEST {
        let clear_0 = rng.gen::<u64>() % modulus;
        let clear_1 = rng.gen::<u64>() % modulus;

        // encryption of an integer
        let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());
        let ct_one = cks.encrypt_crt(clear_1, basis.clone());

        // add the two ciphertexts
        sks.unchecked_add_crt_assign_parallelized(&mut ct_zero, &ct_one);

        // decryption of ct_res
        let dec_res = cks.decrypt_crt(&ct_zero);

        // assert
        assert_eq!((clear_0 + clear_1) % modulus, dec_res % modulus);
    }
}

fn integer_smart_add_crt(param: Parameters) {
    // Define CRT basis, and global modulus
    let basis = make_basis(param.message_modulus.0);
    let modulus = basis.iter().product::<u64>();

    let (cks, sks) = KEY_CACHE.get_from_params(param);

    //RNG
    let mut rng = rand::thread_rng();

    let mut clear_0 = rng.gen::<u64>() % modulus;
    let clear_1 = rng.gen::<u64>() % modulus;

    // encryption of an integer
    let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());
    let mut ct_one = cks.encrypt_crt(clear_1, basis);

    for _ in 0..NB_TEST {
        // add the two ciphertexts
        sks.smart_add_crt_assign(&mut ct_zero, &mut ct_one);

        // decryption of ct_res
        let dec_res = cks.decrypt_crt(&ct_zero);

        // assert
        clear_0 += clear_1;
        assert_eq!(clear_0 % modulus, dec_res % modulus);
    }
}

fn integer_smart_mul_crt(param: Parameters) {
    // generate the server-client key set
    let (cks, sks) = KEY_CACHE.get_from_params(param);

    // Define CRT basis, and global modulus
    let basis = make_basis(param.message_modulus.0);
    let modulus = basis.iter().product::<u64>();

    //RNG
    let mut rng = rand::thread_rng();

    let mut clear_0 = rng.gen::<u64>() % modulus;
    let clear_1 = rng.gen::<u64>() % modulus;

    // encryption of an integer
    let mut ct_zero = cks.encrypt_crt(clear_0, basis.clone());
    let mut ct_one = cks.encrypt_crt(clear_1, basis);

    for _ in 0..NB_TEST_SMALLER {
        // add the two ciphertexts
        sks.smart_mul_crt_assign(&mut ct_zero, &mut ct_one);

        // decryption of ct_res
        let dec_res = cks.decrypt_crt(&ct_zero);

        // assert
        clear_0 *= clear_1;
        assert_eq!(clear_0 % modulus, dec_res % modulus);
    }
}
