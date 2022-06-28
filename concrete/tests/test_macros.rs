#![cfg(all(feature = "booleans", feature = "experimental_syntax_sugar"))]
#![allow(clippy::bool_assert_comparison)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::nonminimal_bool)]
use concrete::prelude::*;
use concrete::{condition, generate_keys, set_server_key, ClientKey, ConfigBuilder, FheBool};

fn setup() -> ClientKey {
    let config = ConfigBuilder::all_disabled().enable_default_bool().build();

    let (my_keys, server_keys) = generate_keys(config);

    set_server_key(server_keys);
    my_keys
}

#[test]
fn simple_and() {
    let keys = setup();

    let a = FheBool::encrypt(true, &keys);
    let b = FheBool::encrypt(true, &keys);

    let res = condition!(a && b).decrypt(&keys);
    assert!(res);
}

#[test]
fn simple_or() {
    let keys = setup();

    let a = FheBool::encrypt(true, &keys);
    let b = FheBool::encrypt(false, &keys);

    let res = condition!(a || b).decrypt(&keys);
    assert!(res);
}

#[test]
fn simple_eq() {
    let keys = setup();

    let a = FheBool::encrypt(true, &keys);
    let b = FheBool::encrypt(false, &keys);

    let res = condition!(a == b).decrypt(&keys);
    assert_eq!(res, false);
}

#[test]
fn combine_and_or() {
    let keys = setup();
    let a = FheBool::encrypt(true, &keys);
    let b = FheBool::encrypt(false, &keys);
    let c = FheBool::encrypt(false, &keys);

    let res = condition!(a || b && c).decrypt(&keys);
    assert_eq!(res, (true || false && false));
}

#[test]
fn combine_and_or_with_parens() {
    let keys = setup();
    let a = FheBool::encrypt(true, &keys);
    let b = FheBool::encrypt(false, &keys);
    let c = FheBool::encrypt(false, &keys);

    let res = condition!((a || b) && c).decrypt(&keys);
    assert_eq!(res, ((true || false) && false));
}
