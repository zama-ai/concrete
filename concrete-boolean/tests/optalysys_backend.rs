use concrete_boolean::*;
use proto_graphec::simulator_2::N_FT_FROM_OTHER_TYPE as N_OPT_FT;

// check that the optical Fourier transform (or the optical simulator) is used when evaluating a
// CMUX gate
#[test]
fn use_opt_ft_cmux() {

    // generate the client and server keys
    let (mut client_key, mut server_key) = gen_keys();

    // encrypt three messages
    let ct_1 = client_key.encrypt(true);
    let ct_2 = client_key.encrypt(false);
    let ct_3 = client_key.encrypt(true);

    // get the current number of evaluations of the optical FT
    let old_n_ft: usize;
    unsafe { old_n_ft = N_OPT_FT; }

    // execute the gate
    let ct_4 = server_key.mux(&ct_3, &ct_1, &ct_2);

    // get the new number of evaluations of the optical FT
    let new_n_ft: usize;
    unsafe { new_n_ft = N_OPT_FT; }

    // check that it is larger than the previous one
    assert!(new_n_ft > old_n_ft);

    // check the result
    assert!(client_key.decrypt(&ct_4));
}

// check that the optical Fourier transform (or the optical simulator) is used when evaluating an
// AND gate
#[test]
fn use_opt_ft_and() {

    // generate the client and server keys
    let (mut client_key, mut server_key) = gen_keys();

    // encrypt three messages
    let ct_1 = client_key.encrypt(true);
    let ct_2 = client_key.encrypt(true);

    // get the current number of evaluations of the optical FT
    let old_n_ft: usize;
    unsafe { old_n_ft = N_OPT_FT; }

    // execute the gate
    let ct_3 = server_key.and(&ct_1, &ct_2);

    // get the new number of evaluations of the optical FT
    let new_n_ft: usize;
    unsafe { new_n_ft = N_OPT_FT; }

    // check that it is larger than the previous one
    assert!(new_n_ft > old_n_ft);

    // check the result
    assert!(client_key.decrypt(&ct_3));
}

// check that the optical Fourier transform (or the optical simulator) is used when evaluating a
// NAND gate
#[test]
fn use_opt_ft_nand() {

    // generate the client and server keys
    let (mut client_key, mut server_key) = gen_keys();

    // encrypt three messages
    let ct_1 = client_key.encrypt(true);
    let ct_2 = client_key.encrypt(true);

    // get the current number of evaluations of the optical FT
    let old_n_ft: usize;
    unsafe { old_n_ft = N_OPT_FT; }

    // execute the gate
    let ct_3 = server_key.nand(&ct_1, &ct_2);

    // get the new number of evaluations of the optical FT
    let new_n_ft: usize;
    unsafe { new_n_ft = N_OPT_FT; }

    // check that it is larger than the previous one
    assert!(new_n_ft > old_n_ft);

    // check the result
    assert!(!client_key.decrypt(&ct_3));
}

// check that the optical Fourier transform (or the optical simulator) is used when evaluating an
// OR gate
#[test]
fn use_opt_ft_or() {

    // generate the client and server keys
    let (mut client_key, mut server_key) = gen_keys();

    // encrypt three messages
    let ct_1 = client_key.encrypt(true);
    let ct_2 = client_key.encrypt(true);

    // get the current number of evaluations of the optical FT
    let old_n_ft: usize;
    unsafe { old_n_ft = N_OPT_FT; }

    // execute the gate
    let ct_3 = server_key.or(&ct_1, &ct_2);

    // get the new number of evaluations of the optical FT
    let new_n_ft: usize;
    unsafe { new_n_ft = N_OPT_FT; }

    // check that it is larger than the previous one
    assert!(new_n_ft > old_n_ft);

    // check the result
    assert!(client_key.decrypt(&ct_3));
}

// check that the optical Fourier transform (or the optical simulator) is used when evaluating a
// XOR gate
#[test]
fn use_opt_ft_xor() {

    // generate the client and server keys
    let (mut client_key, mut server_key) = gen_keys();

    // encrypt three messages
    let ct_1 = client_key.encrypt(true);
    let ct_2 = client_key.encrypt(false);

    // get the current number of evaluations of the optical FT
    let old_n_ft: usize;
    unsafe { old_n_ft = N_OPT_FT; }

    // execute the gate
    let ct_3 = server_key.xor(&ct_1, &ct_2);

    // get the new number of evaluations of the optical FT
    let new_n_ft: usize;
    unsafe { new_n_ft = N_OPT_FT; }

    // check that it is larger than the previous one
    assert!(new_n_ft > old_n_ft);

    // check the result
    assert!(client_key.decrypt(&ct_3));
}

// check that the optical Fourier transform (or the optical simulator) is used when evaluating a
// NOR gate
#[test]
fn use_opt_ft_nor() {

    // generate the client and server keys
    let (mut client_key, mut server_key) = gen_keys();

    // encrypt three messages
    let ct_1 = client_key.encrypt(true);
    let ct_2 = client_key.encrypt(false);

    // get the current number of evaluations of the optical FT
    let old_n_ft: usize;
    unsafe { old_n_ft = N_OPT_FT; }

    // execute the gate
    let ct_3 = server_key.nor(&ct_1, &ct_2);

    // get the new number of evaluations of the optical FT
    let new_n_ft: usize;
    unsafe { new_n_ft = N_OPT_FT; }

    // check that it is larger than the previous one
    assert!(new_n_ft > old_n_ft);

    // check the result
    assert!(!client_key.decrypt(&ct_3));
}

// check that the optical Fourier transform (or the optical simulator) is used when evaluating a
// XNOR gate
#[test]
fn use_opt_ft_xnor() {

    // generate the client and server keys
    let (mut client_key, mut server_key) = gen_keys();

    // encrypt three messages
    let ct_1 = client_key.encrypt(true);
    let ct_2 = client_key.encrypt(false);

    // get the current number of evaluations of the optical FT
    let old_n_ft: usize;
    unsafe { old_n_ft = N_OPT_FT; }

    // execute the gate
    let ct_3 = server_key.xnor(&ct_1, &ct_2);

    // get the new number of evaluations of the optical FT
    let new_n_ft: usize;
    unsafe { new_n_ft = N_OPT_FT; }

    // check that it is larger than the previous one
    assert!(new_n_ft > old_n_ft);

    // check the result
    assert!(!client_key.decrypt(&ct_3));
}
