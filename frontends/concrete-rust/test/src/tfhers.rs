mod precompile {
    use concrete_macro::from_concrete_python_export_zip;
    from_concrete_python_export_zip!("src/test_tfhers.zip");
}

#[cfg(test)]
mod test {
    use super::precompile;
    use tfhe::prelude::{FheDecrypt, FheEncrypt};
    use tfhe::shortint::parameters::v0_10::classic::gaussian::p_fail_2_minus_64::ks_pbs::{V0_10_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64, V0_10_PARAM_MESSAGE_2_CARRY_3_KS_PBS_GAUSSIAN_2M64};
    use tfhe::{generate_keys, FheUint8};

    #[test]
    fn test() {
        let mut secret_csprng = concrete::common::SecretCsprng::new(0u128);
        let mut encryption_csprng = concrete::common::EncryptionCsprng::new(0u128);
        let config = tfhe::ConfigBuilder::with_custom_parameters(
            V0_10_PARAM_MESSAGE_2_CARRY_3_KS_PBS_GAUSSIAN_2M64,
        );
        let (client_key, _) = generate_keys(config);
        let keyset = precompile::KeysetBuilder::new()
            .with_key_for_my_func_0_arg(&client_key)
            .generate(secret_csprng.pin_mut(), encryption_csprng.pin_mut());
        let server_keyset = keyset.get_server();
        let mut server = precompile::server::my_func::ServerFunction::new();
        let arg_0 = FheUint8::encrypt(6u8, &client_key);
        let arg_1 = FheUint8::encrypt(4u8, &client_key);
        let output = server.invoke(&server_keyset, arg_0, arg_1);
        let decrypted: u8 = output.decrypt(&client_key);
        assert_eq!(decrypted, 10);
    }

    #[test]
    #[should_panic]
    fn test_reset_key() {
        let mut secret_csprng = concrete::common::SecretCsprng::new(0u128);
        let mut encryption_csprng = concrete::common::EncryptionCsprng::new(0u128);
        let config1 = tfhe::ConfigBuilder::with_custom_parameters(
            V0_10_PARAM_MESSAGE_2_CARRY_3_KS_PBS_GAUSSIAN_2M64,
        );
        let config2 = tfhe::ConfigBuilder::with_custom_parameters(
            V0_10_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64,
        );
        let (client_key1, _) = generate_keys(config1);
        let (client_key2, _) = generate_keys(config2);
        precompile::KeysetBuilder::new()
            .with_key_for_my_func_0_arg(&client_key1)
            .with_key_for_my_func_0_arg(&client_key2)
            .generate(secret_csprng.pin_mut(), encryption_csprng.pin_mut());
    }
}
