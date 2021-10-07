use std::fs::remove_file;
use std::path::Path;

use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
};
use concrete_core::crypto::bootstrap::FourierBootstrapKey;
use concrete_core::crypto::lwe::LweKeyswitchKey;
use concrete_core::math::fft::Complex64;

use crate::{
    Encoder, LWEParams, LWESecretKey, RLWEParams, RLWESecretKey, VectorLWE, LWEBSK, LWEKSK,
};

fn delete_file<P: AsRef<Path>>(path: P) -> std::io::Result<()> {
    remove_file(path)?;
    Ok(())
}

#[test]
fn test_encoder_save() {
    let filename: &str = "encoder.json";
    let encoder_1 = Encoder::new(-10., -5., 7, 1).unwrap();
    encoder_1.save(filename).unwrap();
    let encoder_2 = Encoder::load(filename).unwrap();
    delete_file(filename).unwrap();
    println!("{} \n {}", encoder_1, encoder_2);
    assert!(encoder_1 == encoder_2, "encoder_1 != encoder_2");
}

#[test]
fn test_lwe_save() {
    let filename: &str = "lwe.json";
    let lwe_1 = VectorLWE::zero(10, 5).unwrap();
    lwe_1.save(filename).unwrap();
    let lwe_2 = VectorLWE::load(filename).unwrap();
    delete_file(filename).unwrap();
    println!("{} \n {}", lwe_1, lwe_2);
    assert!(lwe_1 == lwe_2, "lwe_1 != lwe_2");
}

#[test]
fn test_lwebsk_save() {
    let filename: &str = "lwebsk.json";

    // sk_output.key_size: 16
    // sk_input.key_size: 20

    let a = LWEBSK {
        ciphertexts: FourierBootstrapKey::allocate(
            Complex64::new(2., 0.),
            GlweSize(2),
            PolynomialSize(512),
            DecompositionLevelCount(4),
            DecompositionBaseLog(5),
            LweDimension(20),
        ),
        variance: 0.5,
        dimension: 1,
        polynomial_size: 512,
        base_log: 5,
        level: 4,
    };
    a.save(filename);
    let b = LWEBSK::load(filename);
    delete_file(filename).unwrap();
    println!("{} \n {}", a, b);
    assert!(a == b, "a != b");
}

#[test]
fn test_lweksk_save() {
    let filename: &str = "lweksk.json";
    let ksk1 = LWEKSK {
        ciphertexts: LweKeyswitchKey::allocate(
            0,
            DecompositionLevelCount(7),
            DecompositionBaseLog(2),
            LweDimension(1024),
            LweDimension(512),
        ),
        variance: 0.5,
        dimension_before: 1024,
        dimension_after: 512,
        base_log: 2,
        level: 7,
    };
    ksk1.save(filename);
    let ksk2 = LWEKSK::load(filename);
    delete_file(filename).unwrap();
    println!("{} \n {}", ksk1, ksk2);
    assert!(ksk1 == ksk2, "ksk1 != ksk2");
}

#[test]
fn test_lweparams_save() {
    let filename: &str = "lweparams.json";

    let params1 = LWEParams {
        dimension: 10,
        log2_std_dev: 2,
    };
    params1.save(filename).unwrap();
    let params2 = LWEParams::load(filename).unwrap();
    delete_file(filename).unwrap();
    println!("{} \n {}", params1, params2);
    assert!(params1 == params2, "params1 != params2");
}

#[test]
fn test_lwesecretkey_save() {
    let filename: &str = "lwesk.json";

    let p = LWEParams {
        dimension: 10,
        log2_std_dev: 2,
    };

    let sk1 = LWESecretKey::new(&p);

    sk1.save(filename).unwrap();
    let sk2 = LWESecretKey::load(filename).unwrap();
    delete_file(filename).unwrap();
    println!("{} \n {}", sk1, sk2);
    assert!(sk1 == sk2, "sk1 != sk2");
}

#[test]
fn test_rlweparams_save() {
    let filename: &str = "rlweparams.json";

    let p1 = RLWEParams {
        dimension: 10,
        log2_std_dev: 2,
        polynomial_size: 1024,
    };

    p1.save(filename).unwrap();
    let p2 = RLWEParams::load(filename).unwrap();
    delete_file(filename).unwrap();
    assert!(p1 == p2, "p1 != p2");
}

#[test]
fn test_rlwesecretkey_save() {
    let filename: &str = "rlwesk.json";

    let p = RLWEParams {
        dimension: 10,
        log2_std_dev: 2,
        polynomial_size: 1024,
    };

    let sk1 = RLWESecretKey::new(&p);

    sk1.save(filename).unwrap();
    let sk2 = RLWESecretKey::load(filename).unwrap();
    delete_file(filename).unwrap();
    assert!(sk1 == sk2);
}

// use crate::crypto_api::glwe::VectorRLWE;
// use crate::crypto_api::Plaintext;
// use crate::crypto_api::{LWEParams, LWESecretKey, RLWEParams, RLWESecretKey, LWEBSK, LWEKSK};
// use crate::types::CTorus;

// #[test]
// fn test_new_multiplied_with_bootstrap() {
//     let encoder_1 = Encoder::new(-10., -5., 7, 1).unwrap();
//     let encoder_2 = encoder_1.new_multiplied_with_bootstrap(1).unwrap();

//     println!("{}", encoder_1);
//     println!("{}", encoder_2);

//     panic!();
// }

// #[test]
// fn test_display_test() {
//     let a = Encoder::new(-10., 10., 2, 3).unwrap();
//     println!("a = {}", a);

//     let b = VectorLWE::zero(10, 5);
//     println!("b = {}", b);

//     let c = Plaintext::zero(10);
//     println!("c = {}", c);

//     let d = VectorRLWE::zero(1024, 3, 2);
//     println!("d = {}", d);

//     let e = LWEBSK {
//         ciphertexts: vec![CTorus::new(0., 0.); 10],
//         variance: 0.5,
//         dimension: 100,
//         polynomial_size: 1024,
//         base_log: 2,
//         level: 7,
//     };
//     println!("e = {}", e);

//     let d = LWEKSK {
//         ciphertexts: vec![0; 4],
//         variance: 0.5,
//         dimension_before: 1024,
//         dimension_after: 512,
//         base_log: 2,
//         level: 7,
//     };
//     println!("d = {}", d);

//     let f = LWEParams {
//         dimension: 10,
//         log2_std_dev: 0.2,
//     };
//     println!("f = {}", f);

//     let g = LWESecretKey::new(&f);
//     println!("g = {}", g);

//     let h = RLWEParams {
//         dimension: 10,
//         log2_std_dev: 0.2,
//         polynomial_size: 1024,
//     };
//     println!("h = {}", h);

//     let i = RLWESecretKey::new(&h);
//     println!("i = {}", i);

//     panic!();
// }
