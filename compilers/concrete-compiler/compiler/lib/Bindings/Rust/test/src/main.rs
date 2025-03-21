use concrete::common::Tensor;

mod precompile {
    use concrete_macro::from_concrete_python_export_zip;
    from_concrete_python_export_zip!("src/test.zip");
}

fn main() {
    let mut secret_csprng = concrete::common::SecretCsprng::new(0u128);
    let mut encryption_csprng = concrete::common::EncryptionCsprng::new(0u128);
    let keyset = precompile::new_keyset(secret_csprng.pin_mut(), encryption_csprng.pin_mut());
    let client_keyset = keyset.get_client();
    let server_keyset = keyset.get_server();
    let mut dec_client =
        precompile::client::dec::ClientFunction::new(&client_keyset, encryption_csprng);
    let mut dec_server = precompile::server::dec::ServerFunction::new();
    let input = Tensor::new(vec![5], vec![]);
    let prepared_input = dec_client.prepare_inputs(input);
    let output = dec_server.invoke(&server_keyset, prepared_input);
    let processed_output = dec_client.process_outputs(output);
    println!("{:?}", processed_output.values());
}
