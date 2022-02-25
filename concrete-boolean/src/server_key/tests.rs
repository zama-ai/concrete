use crate::ciphertext::Ciphertext;
use crate::client_key::ClientKey;
use crate::parameters::DEFAULT_PARAMETERS;
use crate::server_key::ServerKey;
use crate::{random_boolean, random_integer};

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
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

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
fn test_and_gate() {
    // generate the client key set
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let mut sks = ServerKey::new(&cks);

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
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let mut sks = ServerKey::new(&cks);

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
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let mut sks = ServerKey::new(&cks);

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
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let mut sks = ServerKey::new(&cks);

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
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let mut sks = ServerKey::new(&cks);

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
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let mut sks = ServerKey::new(&cks);

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
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let mut sks = ServerKey::new(&cks);

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
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let mut sks = ServerKey::new(&cks);

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
fn random_gate_all(ct_tab: &mut [Ciphertext], bool_tab: &mut [bool], sks: &mut ServerKey) {
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
    let mut cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key set
    let mut sks = ServerKey::new(&cks);

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
        random_gate_all(&mut ct_tab, &mut bool_tab, &mut sks);
    }

    // decrypt and assert equality
    for (ct, boolean) in ct_tab.iter().zip(bool_tab.iter()) {
        let dec = cks.decrypt(ct);
        assert_eq!(*boolean, dec);
    }
}
