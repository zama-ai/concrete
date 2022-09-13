use std::ops::{BitAnd, BitOr, BitXor};

use crate::booleans::DynFheBoolEncryptor;
use crate::prelude::*;
use crate::{generate_keys, set_server_key, ClientKey, ConfigBuilder, FheBool, FheBoolParameters};

fn setup_static_default() -> ClientKey {
    let config = ConfigBuilder::all_disabled().enable_default_bool().build();

    let (my_keys, server_keys) = generate_keys(config);

    set_server_key(server_keys);
    my_keys
}

fn setup_static_tfhe() -> ClientKey {
    let config = ConfigBuilder::all_disabled()
        .enable_custom_bool(FheBoolParameters::tfhe_lib())
        .build();

    let (my_keys, server_keys) = generate_keys(config);

    set_server_key(server_keys);
    my_keys
}

#[test]
fn test_xor_truth_table_static_default() {
    let keys = setup_static_default();

    let ttrue = FheBool::encrypt(true, &keys);
    let ffalse = FheBool::encrypt(false, &keys);

    xor_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_and_truth_table_static_default() {
    let keys = setup_static_default();

    let ttrue = FheBool::encrypt(true, &keys);
    let ffalse = FheBool::encrypt(false, &keys);

    and_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_or_truth_table_static_default() {
    let keys = setup_static_default();

    let ttrue = FheBool::encrypt(true, &keys);
    let ffalse = FheBool::encrypt(false, &keys);

    or_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_xor_truth_table_static_tfhe() {
    let keys = setup_static_tfhe();

    let ttrue = FheBool::encrypt(true, &keys);
    let ffalse = FheBool::encrypt(false, &keys);

    xor_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_and_truth_table_static_tfhe() {
    let keys = setup_static_tfhe();

    let ttrue = FheBool::encrypt(true, &keys);
    let ffalse = FheBool::encrypt(false, &keys);

    and_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_or_truth_table_static_tfhe() {
    let keys = setup_static_tfhe();

    let ttrue = FheBool::encrypt(true, &keys);
    let ffalse = FheBool::encrypt(false, &keys);

    or_truth_table(&ttrue, &ffalse, &keys);
}

fn setup_dynamic_default() -> (DynFheBoolEncryptor, ClientKey) {
    let mut config = ConfigBuilder::all_disabled();
    let dyn_bool = config.add_bool_type(FheBoolParameters::default());

    let (my_keys, server_keys) = generate_keys(config);

    set_server_key(server_keys);
    (dyn_bool, my_keys)
}

fn setup_dynamic_tfhe() -> (DynFheBoolEncryptor, ClientKey) {
    let mut config = ConfigBuilder::all_disabled();
    let dyn_bool = config.add_bool_type(FheBoolParameters::tfhe_lib());

    let (my_keys, server_keys) = generate_keys(config);

    set_server_key(server_keys);
    (dyn_bool, my_keys)
}

#[test]
fn test_xor_truth_table_dynamic_default() {
    let (dyn_bool, keys) = setup_dynamic_default();

    let ttrue = dyn_bool.encrypt(true, &keys);
    let ffalse = dyn_bool.encrypt(false, &keys);

    xor_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_and_truth_table_dynamic_default() {
    let (dyn_bool, keys) = setup_dynamic_default();

    let ttrue = dyn_bool.encrypt(true, &keys);
    let ffalse = dyn_bool.encrypt(false, &keys);

    and_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_or_truth_table_dynamic_default() {
    let (dyn_bool, keys) = setup_dynamic_default();

    let ttrue = dyn_bool.encrypt(true, &keys);
    let ffalse = dyn_bool.encrypt(false, &keys);

    or_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_xor_truth_table_dynamic_tfhe() {
    let (dyn_bool, keys) = setup_dynamic_tfhe();

    let ttrue = dyn_bool.encrypt(true, &keys);
    let ffalse = dyn_bool.encrypt(false, &keys);

    xor_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_and_truth_table_dynamic_tfhe() {
    let (dyn_bool, keys) = setup_dynamic_tfhe();

    let ttrue = dyn_bool.encrypt(true, &keys);
    let ffalse = dyn_bool.encrypt(false, &keys);

    and_truth_table(&ttrue, &ffalse, &keys);
}

#[test]
fn test_or_truth_table_dynamic_tfhe() {
    let (dyn_bool, keys) = setup_dynamic_tfhe();

    let ttrue = dyn_bool.encrypt(true, &keys);
    let ffalse = dyn_bool.encrypt(false, &keys);

    or_truth_table(&ttrue, &ffalse, &keys);
}

fn xor_truth_table<'a, BoolType>(ttrue: &'a BoolType, ffalse: &'a BoolType, key: &ClientKey)
where
    &'a BoolType: BitXor<&'a BoolType, Output = BoolType>,
    BoolType: FheDecrypt<bool>,
{
    let r = ffalse ^ ffalse;
    assert_eq!(r.decrypt(key), false);

    let r = ffalse ^ ttrue;
    assert_eq!(r.decrypt(key), true);

    let r = ttrue ^ ffalse;
    assert_eq!(r.decrypt(key), true);

    let r = ttrue ^ ttrue;
    assert_eq!(r.decrypt(key), false);
}

fn and_truth_table<'a, BoolType>(ttrue: &'a BoolType, ffalse: &'a BoolType, key: &ClientKey)
where
    &'a BoolType: BitAnd<&'a BoolType, Output = BoolType>,
    BoolType: FheDecrypt<bool>,
{
    let r = ffalse & ffalse;
    assert_eq!(r.decrypt(key), false);

    let r = ffalse & ttrue;
    assert_eq!(r.decrypt(key), false);

    let r = ttrue & ffalse;
    assert_eq!(r.decrypt(key), false);

    let r = ttrue & ttrue;
    assert_eq!(r.decrypt(key), true);
}

fn or_truth_table<'a, BoolType>(ttrue: &'a BoolType, ffalse: &'a BoolType, key: &ClientKey)
where
    &'a BoolType: BitOr<&'a BoolType, Output = BoolType>,
    BoolType: FheDecrypt<bool>,
{
    let r = ffalse | ffalse;
    assert_eq!(r.decrypt(key), false);

    let r = ffalse | ttrue;
    assert_eq!(r.decrypt(key), true);

    let r = ttrue | ffalse;
    assert_eq!(r.decrypt(key), true);

    let r = ttrue | ttrue;
    assert_eq!(r.decrypt(key), true);
}
