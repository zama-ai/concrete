use crate::ciphertext::Ciphertext;
use crate::client_key::ClientKey;
use crate::parameters::DEFAULT_PARAMETERS;
use crate::server_key::ServerKey;
use crate::{
    random_boolean, random_integer, PLAINTEXT_FALSE, PLAINTEXT_LOG_SCALING_FACTOR, PLAINTEXT_TRUE,
};
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
use concrete_core::crypto::bootstrap::Bootstrap;
use concrete_core::crypto::encoding::Plaintext;
use concrete_core::crypto::glwe::GlweCiphertext;
use concrete_core::crypto::lwe::LweCiphertext;
use concrete_core::crypto::secret::generators::EncryptionRandomGenerator;
use concrete_core::crypto::secret::LweSecretKey;
use concrete_core::math::decomposition::SignedDecomposer;
use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};

/// Number of assert in randomized tests
const NB_TEST: usize = 128;

/// Number of ciphertext in the deep circuit test
const NB_CT: usize = 8;

/// Number of gates computed in the deep circuit test
const NB_GATE: usize = 1 << 11;

#[test]
/// test encryption and decryption with the LWE secret key
fn test_encrypt_decrypt_lwe_secret_key() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    for _ in 0..NB_TEST {
        // encryption of false
        let ct_false = cks.encrypt(false);

        // encryption of true
        let ct_true = cks.encrypt(true);

        // decryption of false
        let dec_false = cks.decrypt(&ct_false);

        // decryption of true
        let dec_true = cks.decrypt(&ct_true);

        // assert
        assert!(!dec_false);
        assert!(dec_true);
    }
}

#[test]
/// test encryption with the LWE secret key and bootstrap
/// and then decryption with the LWE from the RLWE secret key
fn test_encrypt_pbs_decrypt() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    // convert the GLWE secret key into an LWE secret key
    let big_lwe_secret_key =
        LweSecretKey::binary_from_container(cks.glwe_secret_key.as_tensor().clone());

    for _ in 0..NB_TEST {
        // encryption of true
        let ct_true = cks.encrypt(true);

        // encryption of false
        let ct_false = cks.encrypt(false);

        // Allocation of the accumulator
        let mut accumulator = GlweCiphertext::allocate(
            0_u32,
            sks.bootstrapping_key.polynomial_size(),
            sks.bootstrapping_key.glwe_size(),
        );

        // Fill the body of accumulator with the Test Polynomial
        accumulator
            .get_mut_body()
            .as_mut_tensor()
            .fill_with_element(PLAINTEXT_TRUE); // 1/8

        // Allocation for the output of the PBS
        let mut ct_pbs_true = LweCiphertext::allocate(
            0_u32,
            LweSize(
                sks.bootstrapping_key.polynomial_size().0
                    * sks.bootstrapping_key.glwe_size().to_glwe_dimension().0
                    + 1,
            ),
        );
        let mut ct_pbs_false = LweCiphertext::allocate(
            0_u32,
            sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        // Compute the two PBS
        sks.bootstrapping_key
            .bootstrap(&mut ct_pbs_true, &ct_true.0, &accumulator);
        sks.bootstrapping_key
            .bootstrap(&mut ct_pbs_false, &ct_false.0, &accumulator);

        // allocation of the plaintexts
        let mut decrypted_true = Plaintext(0_u32);
        let mut decrypted_false = Plaintext(0_u32);

        // decryption
        big_lwe_secret_key.decrypt_lwe(&mut decrypted_true, &ct_pbs_true);
        big_lwe_secret_key.decrypt_lwe(&mut decrypted_false, &ct_pbs_false);

        // decomposer
        let decomposer: SignedDecomposer<u32> = SignedDecomposer::<u32>::new(
            DecompositionBaseLog(PLAINTEXT_LOG_SCALING_FACTOR),
            DecompositionLevelCount(1),
        );

        // rounding
        let rounded_true = decomposer.closest_representable(decrypted_true.0);
        let rounded_false = decomposer.closest_representable(decrypted_false.0);

        // asserts
        if rounded_true != PLAINTEXT_TRUE {
            panic!(
                "fail with the true: {} -> {}",
                decrypted_true.0, rounded_true
            )
        }
        if rounded_false != PLAINTEXT_FALSE {
            panic!(
                "fail with the false: {} -> {}",
                decrypted_false.0, rounded_false
            )
        }
    }
}

#[test]
/// test encryption with the LWE secret key from the RLWE secret key,
/// and key switch and then decryption with the LWE secret key
fn test_encrypt_ks_decrypt() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    // convert the GLWE secret key into an LWE secret key
    let big_lwe_secret_key =
        LweSecretKey::binary_from_container(cks.glwe_secret_key.as_tensor().clone());

    // instantiate an encryption random generator
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    // allocate the ciphertexts
    let big_size =
        LweSize(DEFAULT_PARAMETERS.polynomial_size.0 * DEFAULT_PARAMETERS.glwe_dimension.0);
    let mut ct_false = LweCiphertext::allocate(0_u32, big_size);
    let mut ct_true = LweCiphertext::allocate(0_u32, big_size);

    for _ in 0..NB_TEST {
        // encryption of false
        big_lwe_secret_key.encrypt_lwe(
            &mut ct_false,
            &Plaintext(PLAINTEXT_FALSE),
            DEFAULT_PARAMETERS.glwe_modular_std_dev,
            &mut encryption_generator,
        );

        // encryption of true
        big_lwe_secret_key.encrypt_lwe(
            &mut ct_true,
            &Plaintext(PLAINTEXT_TRUE),
            DEFAULT_PARAMETERS.glwe_modular_std_dev,
            &mut encryption_generator,
        );

        // key switch of false
        let mut ct_ks_false =
            LweCiphertext::allocate(0_u32, DEFAULT_PARAMETERS.lwe_dimension.to_lwe_size());
        sks.key_switching_key
            .keyswitch_ciphertext(&mut ct_ks_false, &ct_false);

        // key switch of true
        let mut ct_ks_true =
            LweCiphertext::allocate(0_u32, DEFAULT_PARAMETERS.lwe_dimension.to_lwe_size());
        sks.key_switching_key
            .keyswitch_ciphertext(&mut ct_ks_true, &ct_true);

        // decryption of false
        let dec_false = cks.decrypt(&Ciphertext(ct_ks_false));

        // decryption of true
        let dec_true = cks.decrypt(&Ciphertext(ct_ks_true));

        // assert
        assert!(!dec_false);
        assert!(dec_true);
    }
}

#[test]
fn test_and_gate() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    for _ in 0..NB_TEST {
        // generation of two random booleans
        let b1 = random_boolean();
        let b2 = random_boolean();

        // encryption of b1
        let ct1 = cks.encrypt(b1);

        // encryption of b2
        let ct2 = cks.encrypt(b2);

        // AND gate
        let ct_res = sks.and(&ct1, &ct2);

        // decryption
        let dec_and = cks.decrypt(&ct_res);

        // assert
        assert_eq!(b1 && b2, dec_and);
    }
}

#[test]
fn test_mux_gate() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    for _ in 0..NB_TEST {
        // generation of three random booleans
        let b1 = random_boolean();
        let b2 = random_boolean();
        let b3 = random_boolean();

        // encryption of b1
        let ct1 = cks.encrypt(b1);

        // encryption of b2
        let ct2 = cks.encrypt(b2);

        // encryption of b3
        let ct3 = cks.encrypt(b3);

        // MUX gate
        let ct_res = sks.mux(&ct1, &ct2, &ct3);

        // decryption
        let dec_mux = cks.decrypt(&ct_res);

        // assert
        assert_eq!(if b1 { b2 } else { b3 }, dec_mux);
    }
}

#[test]
fn test_nand_gate() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    for _ in 0..NB_TEST {
        // generation of two random booleans
        let b1 = random_boolean();
        let b2 = random_boolean();

        // encryption of b1
        let ct1 = cks.encrypt(b1);

        // encryption of b2
        let ct2 = cks.encrypt(b2);

        // NAND gate
        let ct_res = sks.nand(&ct1, &ct2);

        // decryption
        let dec_nand = cks.decrypt(&ct_res);

        // assert
        assert_eq!(!(b1 && b2), dec_nand);
    }
}

#[test]
fn test_nor_gate() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    for _ in 0..NB_TEST {
        // generation of two random booleans
        let b1 = random_boolean();
        let b2 = random_boolean();

        // encryption of b1
        let ct1 = cks.encrypt(b1);

        // encryption of b2
        let ct2 = cks.encrypt(b2);

        // NOR gate
        let ct_res = sks.nor(&ct1, &ct2);

        // decryption
        let dec_nor = cks.decrypt(&ct_res);

        // assert
        assert_eq!(!(b1 || b2), dec_nor);
    }
}

#[test]
fn test_not_gate() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    for _ in 0..NB_TEST {
        // generation of one random booleans
        let b1 = random_boolean();

        // encryption of b1
        let ct1 = cks.encrypt(b1);

        // NOT gate
        let ct_res = sks.not(&ct1);

        // decryption
        let dec_not = cks.decrypt(&ct_res);

        // assert
        assert_eq!(!b1, dec_not);
    }
}

#[test]
fn test_or_gate() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    for _ in 0..NB_TEST {
        // generation of two random booleans
        let b1 = random_boolean();
        let b2 = random_boolean();

        // encryption of b1
        let ct1 = cks.encrypt(b1);

        // encryption of b2
        let ct2 = cks.encrypt(b2);

        // OR gate
        let ct_res = sks.or(&ct1, &ct2);

        // decryption
        let dec_or = cks.decrypt(&ct_res);

        // assert
        assert_eq!(b1 || b2, dec_or);
    }
}

#[test]
fn test_xnor_gate() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    for _ in 0..NB_TEST {
        // generation of two random booleans
        let b1 = random_boolean();
        let b2 = random_boolean();

        // encryption of b1
        let ct1 = cks.encrypt(b1);

        // encryption of b2
        let ct2 = cks.encrypt(b2);

        // XNOR gate
        let ct_res = sks.xnor(&ct1, &ct2);

        // decryption
        let dec_xnor = cks.decrypt(&ct_res);

        // assert
        assert_eq!(b1 == b2, dec_xnor);
    }
}

#[test]
fn test_xor_gate() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    for _ in 0..NB_TEST {
        // generation of two random booleans
        let b1 = random_boolean();
        let b2 = random_boolean();

        // encryption of b1
        let ct1 = cks.encrypt(b1);

        // encryption of b2
        let ct2 = cks.encrypt(b2);

        // XOR gate
        let ct_res = sks.xor(&ct1, &ct2);

        // decryption
        let dec_xor = cks.decrypt(&ct_res);

        // assert
        assert_eq!(b1 ^ b2, dec_xor);
    }
}

/// generate a random index for the table in the long run tests
fn random_index() -> usize {
    (random_integer() % (NB_CT as u32)) as usize
}

/// randomly select a gate, randomly select inputs and the output,
/// compute the selected gate with the selected inputs
/// and write in the selected output
fn random_gate_all(ct_tab: &mut [Ciphertext], bool_tab: &mut [bool], sks: &ServerKey) {
    // select a random gate in the array [NOT,CMUX,AND,NAND,NOR,OR,XOR,XNOR]
    let gate_id = random_integer() % 8;

    let index_1: usize = random_index();
    let index_2: usize = random_index();

    if gate_id == 0 {
        // NOT gate
        bool_tab[index_2] = !bool_tab[index_1];
        ct_tab[index_2] = sks.not(&ct_tab[index_1]);
    } else if gate_id == 1 {
        // MUX gate
        let index_3: usize = random_index();
        let index_4: usize = random_index();
        bool_tab[index_4] = if bool_tab[index_1] {
            bool_tab[index_2]
        } else {
            bool_tab[index_3]
        };
        ct_tab[index_4] = sks.mux(&ct_tab[index_1], &ct_tab[index_2], &ct_tab[index_3]);
    } else {
        // 2-input gate
        let index_3: usize = random_index();

        if gate_id == 2 {
            // AND gate
            bool_tab[index_3] = bool_tab[index_1] && bool_tab[index_2];
            ct_tab[index_3] = sks.and(&ct_tab[index_1], &ct_tab[index_2]);
        } else if gate_id == 3 {
            // NAND gate
            bool_tab[index_3] = !(bool_tab[index_1] && bool_tab[index_2]);
            ct_tab[index_3] = sks.nand(&ct_tab[index_1], &ct_tab[index_2]);
        } else if gate_id == 4 {
            // NOR gate
            bool_tab[index_3] = !(bool_tab[index_1] || bool_tab[index_2]);
            ct_tab[index_3] = sks.nor(&ct_tab[index_1], &ct_tab[index_2]);
        } else if gate_id == 5 {
            // OR gate
            bool_tab[index_3] = bool_tab[index_1] || bool_tab[index_2];
            ct_tab[index_3] = sks.or(&ct_tab[index_1], &ct_tab[index_2]);
        } else if gate_id == 6 {
            // XOR gate
            bool_tab[index_3] = bool_tab[index_1] ^ bool_tab[index_2];
            ct_tab[index_3] = sks.xor(&ct_tab[index_1], &ct_tab[index_2]);
        } else {
            // XNOR gate
            bool_tab[index_3] = !(bool_tab[index_1] ^ bool_tab[index_2]);
            ct_tab[index_3] = sks.xnor(&ct_tab[index_1], &ct_tab[index_2]);
        }
    }
}

#[test]
fn test_deep_circuit() {
    // generate the client key set
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let sks = ServerKey::new(&cks);

    // create an array of ciphertexts
    let mut ct_tab: Vec<Ciphertext> = vec![cks.encrypt(true); NB_CT];

    // create an array of booleans
    let mut bool_tab: Vec<bool> = vec![true; NB_CT];

    // randomly fill both arrays
    for (ct, boolean) in ct_tab.iter_mut().zip(bool_tab.iter_mut()) {
        *boolean = random_boolean();
        *ct = cks.encrypt(*boolean);
    }

    // compute NB_GATE gates
    for _ in 0..NB_GATE {
        random_gate_all(&mut ct_tab, &mut bool_tab, &sks);
    }

    // decrypt and assert equality
    for (ct, boolean) in ct_tab.iter().zip(bool_tab.iter()) {
        let dec = cks.decrypt(ct);
        assert_eq!(*boolean, dec);
    }
}
