mod precompile {
    use concrete_macro::from_concrete_python_export_zip;
    from_concrete_python_export_zip!("src/test.zip");
}

#[cfg(test)]
mod test {
    use super::precompile;
    use concrete::common::{ClientKeyset, ServerKeyset, Tensor, TransportValue};

    #[test]
    fn test() {
        let mut secret_csprng = concrete::common::SecretCsprng::new(0u128);
        let mut encryption_csprng = concrete::common::EncryptionCsprng::new(0u128);
        let keyset = precompile::KeysetBuilder::new()
            .generate(secret_csprng.pin_mut(), encryption_csprng.pin_mut());
        let client_keyset = keyset.get_client();
        let serialized_client_keyset = client_keyset.serialize();
        let deserialized_client_keyset =
            ClientKeyset::deserialize(serialized_client_keyset.as_slice());
        let server_keyset = keyset.get_server();
        let serialized_server_keyset = server_keyset.serialize();
        let deserialized_server_keyset =
            ServerKeyset::deserialize(serialized_server_keyset.as_slice());
        let mut dec_client = precompile::client::dec::ClientFunction::new(
            &deserialized_client_keyset,
            encryption_csprng,
        );
        let mut dec_server = precompile::server::dec::ServerFunction::new();
        let input = Tensor::new(vec![5], vec![]);
        let prepared_input = dec_client.prepare_inputs(input);
        let serialized_input = prepared_input.serialize();
        let deserialized_input = TransportValue::deserialize(serialized_input.as_slice());
        let output = dec_server.invoke(&deserialized_server_keyset, deserialized_input);
        let serialized_output = output.serialize();
        let deserialized_output = TransportValue::deserialize(serialized_output.as_slice());
        let processed_output = dec_client.process_outputs(deserialized_output);
        assert_eq!(processed_output.values(), [4]);
        assert_eq!(processed_output.dimensions().len(), 0);
    }
}
