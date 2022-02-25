use concrete_boolean::client_key::ClientKey;
use concrete_boolean::parameters::{BooleanParameters, DEFAULT_PARAMETERS, TFHE_LIB_PARAMETERS};
use concrete_boolean::prelude::BinaryBooleanGates;
use concrete_boolean::server_key::ServerKey;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

criterion_group!(
    gate_bootstrapping_default_params,
    and_gate_default,
    mux_gate_default,
    nand_gate_default,
    nor_gate_default,
    not_gate_default,
    or_gate_default,
    xnor_gate_default,
    xor_gate_default
);
criterion_group!(
    gate_bootstrapping_tfhelib_params,
    and_gate_tfhelib,
    mux_gate_tfhelib,
    nand_gate_tfhelib,
    nor_gate_tfhelib,
    not_gate_tfhelib,
    or_gate_tfhelib,
    xnor_gate_tfhelib,
    xor_gate_tfhelib
);

criterion_main!(
    gate_bootstrapping_default_params,
    gate_bootstrapping_tfhelib_params
);

fn and_gate(c: &mut Criterion, params: BooleanParameters, option: &str) {
    let mut cks = ClientKey::new(params);
    let mut sks = ServerKey::new(&cks);
    let ct1 = cks.encrypt(true);
    let ct2 = cks.encrypt(false);
    c.bench_function(&*("AND gate ".to_owned() + option), |b| {
        b.iter(|| black_box(sks.and(&ct1, &ct2)))
    });
}

fn and_gate_default(c: &mut Criterion) {
    and_gate(c, DEFAULT_PARAMETERS, "(default parameters)");
}

fn and_gate_tfhelib(c: &mut Criterion) {
    and_gate(c, TFHE_LIB_PARAMETERS, "(TFHE-lib parameters)");
}

fn mux_gate(c: &mut Criterion, params: BooleanParameters, option: &str) {
    let mut cks = ClientKey::new(params);
    let mut sks = ServerKey::new(&cks);
    let ct1 = cks.encrypt(true);
    let ct2 = cks.encrypt(false);
    let ct3 = cks.encrypt(true);
    c.bench_function(&*("MUX gate ".to_owned() + option), |b| {
        b.iter(|| black_box(sks.mux(&ct1, &ct2, &ct3)))
    });
}

fn mux_gate_default(c: &mut Criterion) {
    mux_gate(c, DEFAULT_PARAMETERS, "(default parameters)");
}

fn mux_gate_tfhelib(c: &mut Criterion) {
    mux_gate(c, TFHE_LIB_PARAMETERS, "(TFHE-lib parameters)");
}

fn nand_gate(c: &mut Criterion, params: BooleanParameters, option: &str) {
    let mut cks = ClientKey::new(params);
    let mut sks = ServerKey::new(&cks);
    let ct1 = cks.encrypt(true);
    let ct2 = cks.encrypt(false);
    c.bench_function(&*("NAND gate ".to_owned() + option), |b| {
        b.iter(|| black_box(sks.nand(&ct1, &ct2)))
    });
}

fn nand_gate_default(c: &mut Criterion) {
    nand_gate(c, DEFAULT_PARAMETERS, "(default parameters)");
}

fn nand_gate_tfhelib(c: &mut Criterion) {
    nand_gate(c, TFHE_LIB_PARAMETERS, "(TFHE-lib parameters)");
}

fn nor_gate(c: &mut Criterion, params: BooleanParameters, option: &str) {
    let mut cks = ClientKey::new(params);
    let mut sks = ServerKey::new(&cks);
    let ct1 = cks.encrypt(true);
    let ct2 = cks.encrypt(false);
    c.bench_function(&*("NOR gate ".to_owned() + option), |b| {
        b.iter(|| black_box(sks.nor(&ct1, &ct2)))
    });
}

fn nor_gate_default(c: &mut Criterion) {
    nor_gate(c, DEFAULT_PARAMETERS, "(default parameters)");
}

fn nor_gate_tfhelib(c: &mut Criterion) {
    nor_gate(c, TFHE_LIB_PARAMETERS, "(TFHE-lib parameters)");
}

fn not_gate(c: &mut Criterion, params: BooleanParameters, option: &str) {
    let mut cks = ClientKey::new(params);
    let mut sks = ServerKey::new(&cks);
    let ct = cks.encrypt(true);
    c.bench_function(&*("NOT gate ".to_owned() + option), |b| {
        b.iter(|| black_box(sks.not(&ct)))
    });
}

fn not_gate_default(c: &mut Criterion) {
    not_gate(c, DEFAULT_PARAMETERS, "(default parameters)");
}

fn not_gate_tfhelib(c: &mut Criterion) {
    not_gate(c, TFHE_LIB_PARAMETERS, "(TFHE-lib parameters)");
}

fn or_gate(c: &mut Criterion, params: BooleanParameters, option: &str) {
    let mut cks = ClientKey::new(params);
    let mut sks = ServerKey::new(&cks);
    let ct1 = cks.encrypt(true);
    let ct2 = cks.encrypt(false);
    c.bench_function(&*("OR gate ".to_owned() + option), |b| {
        b.iter(|| black_box(sks.or(&ct1, &ct2)))
    });
}

fn or_gate_default(c: &mut Criterion) {
    or_gate(c, DEFAULT_PARAMETERS, "(default parameters)");
}

fn or_gate_tfhelib(c: &mut Criterion) {
    or_gate(c, TFHE_LIB_PARAMETERS, "(TFHE-lib parameters)");
}

fn xnor_gate(c: &mut Criterion, params: BooleanParameters, option: &str) {
    let mut cks = ClientKey::new(params);
    let mut sks = ServerKey::new(&cks);
    let ct1 = cks.encrypt(true);
    let ct2 = cks.encrypt(false);
    c.bench_function(&*("XNOR gate ".to_owned() + option), |b| {
        b.iter(|| black_box(sks.xnor(&ct1, &ct2)))
    });
}

fn xnor_gate_default(c: &mut Criterion) {
    xnor_gate(c, DEFAULT_PARAMETERS, "(default parameters)");
}

fn xnor_gate_tfhelib(c: &mut Criterion) {
    xnor_gate(c, TFHE_LIB_PARAMETERS, "(TFHE-lib parameters)");
}

fn xor_gate(c: &mut Criterion, params: BooleanParameters, option: &str) {
    let mut cks = ClientKey::new(params);
    let mut sks = ServerKey::new(&cks);
    let ct1 = cks.encrypt(true);
    let ct2 = cks.encrypt(false);
    c.bench_function(&*("XOR gate ".to_owned() + option), |b| {
        b.iter(|| black_box(sks.xor(&ct1, &ct2)))
    });
}

fn xor_gate_default(c: &mut Criterion) {
    xor_gate(c, DEFAULT_PARAMETERS, "(default parameters)");
}

fn xor_gate_tfhelib(c: &mut Criterion) {
    xor_gate(c, TFHE_LIB_PARAMETERS, "(TFHE-lib parameters)");
}
