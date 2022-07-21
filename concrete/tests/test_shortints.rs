#![cfg(feature = "shortints")]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::identity_op)]

use concrete::prelude::*;
use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint2, FheUint3, FheUint4};

#[test]
fn test_uint2() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::all_disabled().enable_default_uint2().build();
    let (keys, server_keys) = generate_keys(config);
    set_server_key(server_keys);

    let mut a = FheUint2::try_encrypt(0, &keys)?;
    let b = FheUint2::try_encrypt(1, &keys)?;

    a += &b;
    let decrypted = a.decrypt(&keys);
    assert_eq!(decrypted, 1);

    a = a + &b;
    let decrypted = a.decrypt(&keys);
    assert_eq!(decrypted, 2);

    a = a - &b;
    let decrypted = a.decrypt(&keys);
    assert_eq!(decrypted, 1);

    Ok(())
}


#[test]
fn test_scalar_comparison_fhe_uint_3() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::all_disabled().enable_default_uint3().build();
    let (keys, server_keys) = generate_keys(config);
    set_server_key(server_keys);

    let a = FheUint3::try_encrypt(2, &keys)?;

    let mut  b = a.scalar_eq(2);
    let decrypted = b.decrypt(&keys);
    assert_eq!(decrypted, 1);

    b = a.scalar_ge(2);
    let decrypted = b.decrypt(&keys);
    assert_eq!(decrypted, 1);

    b = a.scalar_gt(2);
    let decrypted = b.decrypt(&keys);
    assert_eq!(decrypted, 0);

    b = a.scalar_le(2);
    let decrypted = b.decrypt(&keys);
    assert_eq!(decrypted, 1);

    b = a.scalar_lt(2);
    let decrypted = b.decrypt(&keys);
    assert_eq!(decrypted, 0);

    Ok(())
}

#[test]
fn test_programmable_bootstrap_fhe_uint2() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::all_disabled().enable_default_uint2().build();
    let (keys, server_keys) = generate_keys(config);
    set_server_key(server_keys);

    let mut a = FheUint2::try_encrypt(2, &keys)?;

    let c = a.map(|value| value * value);
    let decrypted = c.decrypt(&keys);
    assert_eq!(decrypted, (2 * 2) % 2);

    a.apply(|value| value * value);
    let decrypted = a.decrypt(&keys);
    assert_eq!(decrypted, (2 * 2) % 2);

    Ok(())
}

#[test]
fn test_programmable_biviariate_bootstrap_fhe_uint3() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::all_disabled().enable_default_uint3().build();
    let (keys, server_keys) = generate_keys(config);
    set_server_key(server_keys);

    for i in 0..FheUint3::MAX {
        let clear_a = i;
        let clear_b = i + 1;

        let a = FheUint3::try_encrypt(clear_a, &keys)?;
        let b = FheUint3::try_encrypt(clear_b, &keys)?;

        let result = a.bivariate_pbs(&b, std::cmp::max);
        let clear_result: u8 = result.decrypt(&keys);
        assert_eq!(clear_result, std::cmp::max(clear_a, clear_b));

        // check reversing lhs and rhs works
        let result = b.bivariate_pbs(&a, std::cmp::max);
        let clear_result: u8 = result.decrypt(&keys);
        assert_eq!(clear_result, std::cmp::max(clear_b, clear_a));
    }

    Ok(())
}

#[test]
fn test_branch_less_min_max() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::all_disabled().enable_default_uint4().build();
    let (keys, server_keys) = generate_keys(config);
    set_server_key(server_keys);

    let x = FheUint4::try_encrypt(12, &keys)?;
    let y = FheUint4::try_encrypt(4, &keys)?;

    let min = &y ^ (&x ^ &y) & -(x.lt(&y));
    let max = &x ^ (&x ^ &y) & -(x.lt(&y));

    assert_eq!(min.decrypt(&keys), 4);
    assert_eq!(max.decrypt(&keys), 12);

    Ok(())
}

mod dynamic {
    use concrete::prelude::*;
    use concrete::{generate_keys, set_server_key, ConfigBuilder, FheUint3Parameters};

    #[test]
    fn test_dynamic_shortint() {
        let mut config_builder = ConfigBuilder::all_disabled();
        // This features is only really interesting if the user provides its own params
        // but here the point is proving that it works
        let my_short_int_type =
            config_builder.add_short_int_type(FheUint3Parameters::with_carry_3().into());

        let (client_key, server_key) = generate_keys(config_builder);

        set_server_key(server_key);

        let a = my_short_int_type.try_encrypt(2, &client_key).unwrap();
        let b = my_short_int_type.try_encrypt(1, &client_key).unwrap();

        let c = a + b;

        let result = c.decrypt(&client_key);

        assert_eq!(result, 3);

        let mut a = my_short_int_type.try_encrypt(2, &client_key).unwrap();
        let c = a.map(|value| value * value);
        let decrypted = c.decrypt(&client_key);
        assert_eq!(decrypted, (2 * 2) % ((2 << 3) - 1));

        a.apply(|value| value * value);
        let decrypted = a.decrypt(&client_key);
        assert_eq!(decrypted, (2 * 2) % ((2 << 3) - 1));
    }
}
