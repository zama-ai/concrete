#![cfg(feature = "integers")]

use concrete::prelude::*;
use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint16, FheUint8};

#[test]
fn test_uint8() {
    let config = ConfigBuilder::all_disabled().enable_default_uint8().build();

    let (client_key, server_key) = generate_keys(config);

    set_server_key(server_key);

    let a = FheUint8::encrypt(27, &client_key);
    let b = FheUint8::encrypt(100, &client_key);

    let c: u8 = (a + b).decrypt(&client_key);
    assert_eq!(c, 127);
}

#[test]
fn test_uint16_func_eval_on_clean() {
    const SBOX: [u8; 256] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab,
        0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4,
        0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71,
        0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
        0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6,
        0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb,
        0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45,
        0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44,
        0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a,
        0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49,
        0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
        0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25,
        0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e,
        0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1,
        0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb,
        0x16,
    ];

    let config = ConfigBuilder::all_disabled()
        .enable_default_uint16()
        .build();

    let (client_key, server_key) = generate_keys(config);

    set_server_key(server_key);

    let clear_a: u16 = 123;
    let a = FheUint16::encrypt(clear_a, &client_key);
    let f = |x| SBOX[(x % 256) as usize] as u64;
    let clear_res = f(clear_a.into()) % FheUint16::MODULUS;
    let res = a.map(f);
    let decrypted_res: u64 = res.decrypt(&client_key);
    assert_eq!(decrypted_res, clear_res);
}

#[test]
fn test_uint8_func_eval_on_dirty() {
    let config = ConfigBuilder::all_disabled().enable_default_uint8().build();

    let (client_key, server_key) = generate_keys(config);

    set_server_key(server_key);

    let clear_a = 27u8;
    let clear_b = 128u8;

    let a = FheUint8::encrypt(clear_a, &client_key);
    let b = FheUint8::encrypt(clear_b, &client_key);

    let result = a + b;

    let decrypted_result: u8 = result.decrypt(&client_key);
    let clear_result = clear_a + clear_b;
    assert_eq!(decrypted_result, clear_result);

    let f = |x| x << 3;

    let a = FheUint8::encrypt(clear_a, &client_key);
    let a = a + result;
    let d: u8 = a.decrypt(&client_key);
    assert_eq!(d, decrypted_result + clear_a);

    let clear_res = f((clear_a + decrypted_result).into()) % FheUint8::MODULUS;
    let res = (a).map(f);
    let decrypted_res: u64 = res.decrypt(&client_key);
    assert_eq!(decrypted_res, clear_res);
}

#[test]
fn test_uint8_bivariate_func_eval_on_dirty() {
    let config = ConfigBuilder::all_disabled().enable_default_uint8().build();

    let (client_key, server_key) = generate_keys(config);

    set_server_key(server_key);

    let clear_a = 128u8;
    let clear_b = 43u8;

    let a = FheUint8::encrypt(clear_a, &client_key);
    let b = FheUint8::encrypt(clear_b, &client_key);

    let result = &a + &b;

    let decrypted_result: u8 = result.decrypt(&client_key);
    let clear_result = clear_a + clear_b;
    assert_eq!(decrypted_result, clear_result);

    let func = |lhs, rhs| u64::from(lhs == rhs);

    let is_a_eq_b = a.bivariate_function(&b, func);
    let clear_result: u8 = is_a_eq_b.decrypt(&client_key);
    assert_eq!(clear_result, 0);

    let is_res_eq_b = result.bivariate_function(&b, func);
    let clear_result: u8 = is_res_eq_b.decrypt(&client_key);
    assert_eq!(clear_result, 0);

    let is_a_eq_a = a.bivariate_function(&a, func);
    let clear_result: u8 = is_a_eq_a.decrypt(&client_key);
    assert_eq!(clear_result, 1);
}

mod dynamic {
    use concrete::prelude::*;
    use concrete::{
        generate_keys, set_server_key, ConfigBuilder, CrtParameters, FheUint2Parameters,
        RadixParameters,
    };

    #[test]
    fn test_uint10() {
        let mut config = ConfigBuilder::all_disabled();
        let uint10_type = config.add_integer_type(RadixParameters {
            block_parameters: FheUint2Parameters::with_carry_2().into(),
            num_block: 5,
        });

        let (client_key, server_key) = generate_keys(config);

        set_server_key(server_key);

        let a = uint10_type.encrypt(127, &client_key);
        let b = uint10_type.encrypt(100, &client_key);

        let c: u64 = (a + b).decrypt(&client_key);
        assert_eq!(c, 227);
    }

    #[test]
    fn test_uint16_crt() {
        let mut config = ConfigBuilder::all_disabled();
        let uint10_type = config.add_integer_type(CrtParameters {
            block_parameters: FheUint2Parameters::with_carry_2().into(),
            moduli: vec![7, 8, 9, 11, 13],
        });

        let (client_key, server_key) = generate_keys(config);

        set_server_key(server_key);

        let a = uint10_type.encrypt(552, &client_key);
        let b = uint10_type.encrypt(1232, &client_key);

        let c: u64 = (a + b).decrypt(&client_key);
        assert_eq!(c, 552 + 1232);
    }
}
