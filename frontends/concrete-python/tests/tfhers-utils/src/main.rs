use bincode;
use clap::{Arg, ArgAction, Command};
use core::panic;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use tfhe::core_crypto::prelude::LweSecretKey;
use tfhe::shortint::{ClassicPBSParameters, EncryptionKeyChoice};
use tfhe::{generate_keys, set_server_key, ClientKey, FheUint8, ServerKey};
use tfhe::{prelude::*, ConfigBuilder};

const BLOCK_PARAMS: ClassicPBSParameters = tfhe::shortint::prelude::PARAM_MESSAGE_2_CARRY_3_KS_PBS;

fn load_lwe_sk(lwe_sk_path: &String) -> LweSecretKey<Vec<u64>> {
    let path_sk: &Path = Path::new(lwe_sk_path);
    let serialized_sk = fs::read(path_sk).unwrap();
    let mut serialized_data = Cursor::new(serialized_sk);
    bincode::deserialize_from(&mut serialized_data).unwrap()
}

fn set_server_key_from_file(server_key_path: &String) {
    let serialized_sk = fs::read(Path::new(server_key_path)).unwrap();
    let mut serialized_data = Cursor::new(serialized_sk);
    let sk: ServerKey = bincode::deserialize_from(&mut serialized_data).unwrap();
    set_server_key(sk);
}

fn load_client_key(client_path: &String) -> ClientKey {
    let path_key: &Path = Path::new(client_path);
    let serialized_key = fs::read(path_key).unwrap();
    let mut serialized_data = Cursor::new(serialized_key);
    let client_key: ClientKey = bincode::deserialize_from(&mut serialized_data).unwrap();
    client_key
}

fn serialize_fheuint8(fheuint: FheUint8, ciphertext_path: &String) {
    let mut serialized_ct = Vec::new();
    bincode::serialize_into(&mut serialized_ct, &fheuint).unwrap();
    let path_ct: &Path = Path::new(ciphertext_path);
    fs::write(path_ct, serialized_ct).unwrap();
}

fn encrypt_with_key(
    value: u8,
    client_key: ClientKey,
    ciphertext_path: &String,
) -> Result<(), Box<dyn std::error::Error>> {
    let ct = FheUint8::encrypt(value, &client_key);

    serialize_fheuint8(ct, ciphertext_path);

    Ok(())
}

fn deserialize_fheuint8(path: &String) -> FheUint8 {
    let path_fheuint: &Path = Path::new(path);
    let serialized_fheuint = fs::read(path_fheuint).unwrap();
    let mut serialized_data = Cursor::new(serialized_fheuint);
    bincode::deserialize_from(&mut serialized_data).unwrap()
}

fn decrypt_with_key(
    client_key: ClientKey,
    ciphertext_path: &String,
    plaintext_path: Option<&String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let fheuint = deserialize_fheuint8(ciphertext_path);

    let result: u8 = fheuint.decrypt(&client_key);

    if let Some(path) = plaintext_path {
        let pt_path: &Path = Path::new(path);
        fs::write(pt_path, result.to_string())?;
    } else {
        println!("result: {}", result);
    }

    Ok(())
}

fn sum(cts_paths: Vec<&String>, out_ct_path: &String) -> Result<(), Box<dyn std::error::Error>> {
    if cts_paths.len() == 0 {
        panic!("can't call sum with 0 ciphertexts");
    }
    let mut acc = deserialize_fheuint8(cts_paths[0]);
    for ct_path in cts_paths[1..].iter() {
        let fheuint = deserialize_fheuint8(ct_path);
        acc += fheuint;
    }

    serialize_fheuint8(acc, out_ct_path);

    Ok(())
}

fn write_keys(
    client_key_path: &String,
    server_key_path: &String,
    output_lwe_path: &String,
    client_key: Option<ClientKey>,
    server_key: Option<ServerKey>,
    lwe_secret_key: Option<LweSecretKey<Vec<u64>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ck) = client_key {
        let mut serialized_client_key = Vec::new();
        bincode::serialize_into(&mut serialized_client_key, &ck)?;
        let path_client_key: &Path = Path::new(client_key_path);
        fs::write(path_client_key, serialized_client_key).unwrap();
    }

    if let Some(sk) = server_key {
        let mut serialized_server_key = Vec::new();
        bincode::serialize_into(&mut serialized_server_key, &sk)?;
        let path_server_key: &Path = Path::new(server_key_path);
        fs::write(path_server_key, serialized_server_key).unwrap();
    }

    if let Some(lwe_sk) = lwe_secret_key {
        let mut serialized_lwe_key = Vec::new();
        bincode::serialize_into(&mut serialized_lwe_key, &lwe_sk)?;
        let path_lwe_key: &Path = Path::new(output_lwe_path);
        fs::write(path_lwe_key, serialized_lwe_key).unwrap();
    }

    Ok(())
}

fn keygen(
    client_key_path: &String,
    server_key_path: &String,
    output_lwe_path: &String,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::with_custom_parameters(BLOCK_PARAMS).build();

    let (client_key, server_key) = generate_keys(config);
    let (integer_ck, _, _) = client_key.clone().into_raw_parts();
    let shortint_ck = integer_ck.into_raw_parts();
    assert!(BLOCK_PARAMS.encryption_key_choice == EncryptionKeyChoice::Big);
    let (glwe_secret_key, _, _) = shortint_ck.into_raw_parts();
    let lwe_secret_key = glwe_secret_key.into_lwe_secret_key();

    write_keys(
        client_key_path,
        server_key_path,
        output_lwe_path,
        Some(client_key),
        Some(server_key),
        Some(lwe_secret_key),
    )?;
    Ok(())
}

fn keygen_from_lwe(lwe_sk_path: &String) -> ClientKey {
    let lwe_sk = load_lwe_sk(lwe_sk_path);

    let shortint_key =
        tfhe::shortint::ClientKey::try_from_lwe_encryption_key(lwe_sk, BLOCK_PARAMS).unwrap();
    let client_key = ClientKey::from_raw_parts(shortint_key.into(), None, None);
    client_key
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("tfhers-utils")
        .about("TFHErs utilities")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("encrypt-with-key")
                .short_flag('e')
                .long_flag("encrypt")
                .about("Encrypt a value with a given key.")
                .arg(
                    Arg::new("value")
                        .short('v')
                        .long("value")
                        .help("value to encrypt")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1),
                )
                .arg(
                    Arg::new("ciphertext")
                        .short('c')
                        .long("ciphertext")
                        .help("ciphertext path to write to after encryption")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1),
                )
                .arg(
                    Arg::new("client-key")
                        .long("client-key")
                        .conflicts_with("lwe-sk")
                        .help("client key path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1),
                )
                .arg(
                    Arg::new("lwe-sk")
                        .long("lwe-sk")
                        .conflicts_with("client-key")
                        .help("lwe key path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1),
                ),
        )
        .subcommand(
            Command::new("decrypt-with-key")
                .short_flag('d')
                .long_flag("decrypt")
                .about("Decrypt a ciphertext with a given key.")
                .arg(
                    Arg::new("ciphertext")
                        .short('c')
                        .long("ciphertext")
                        .help("ciphertext path to decrypt")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1),
                )
                .arg(
                    Arg::new("plaintext")
                        .long("plaintext")
                        .short('p')
                        .help("output plaintext path")
                        .action(ArgAction::Set)
                        .num_args(1),
                )
                .arg(
                    Arg::new("client-key")
                        .long("client-key")
                        .conflicts_with("lwe-sk")
                        .help("client key path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1),
                )
                .arg(
                    Arg::new("lwe-sk")
                        .long("lwe-sk")
                        .conflicts_with("client-key")
                        .help("lwe key path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1),
                ),
        )
        .subcommand(
            Command::new("add")
                .short_flag('a')
                .long_flag("add")
                .about("Compute an addition of multiple ciphertexts.")
                .arg(
                    Arg::new("server-key")
                        .short('s')
                        .long("server-key")
                        .help("server key path")
                        .required(true)
                        .action(ArgAction::Set)
                        .num_args(1),
                )
                .arg(
                    Arg::new("ciphertexts")
                        .short('c')
                        .long("cts")
                        .help("ciphertexts to add")
                        .action(ArgAction::Append)
                        .required(true)
                        .num_args(2..),
                )
                .arg(
                    Arg::new("output-ciphertext")
                        .long("output-ct")
                        .short('o')
                        .help("output ciphertext path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1),
                ),
        )
        .subcommand(
            Command::new("keygen")
                .short_flag('k')
                .long_flag("keygen")
                .about("Generate client and server key.")
                .arg(
                    Arg::new("lwe-sk")
                        .long("lwe-sk")
                        .help("lwe key path to use for keygen")
                        .action(ArgAction::Set)
                        .required(false)
                        .num_args(1),
                )
                .arg(
                    Arg::new("output-lwe-sk")
                        .long("output-lwe-sk")
                        .default_value("lwe_secret_key")
                        .help("output lwe key path")
                        .action(ArgAction::Set)
                        .num_args(1),
                )
                .arg(
                    Arg::new("client-key")
                        .short('c')
                        .default_value("client_key")
                        .long("client-key")
                        .help("output client key path")
                        .action(ArgAction::Set)
                        .num_args(1),
                )
                .arg(
                    Arg::new("server-key")
                        .short('s')
                        .default_value("server_key")
                        .long("server-key")
                        .help("output server key path")
                        .action(ArgAction::Set)
                        .num_args(1),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("encrypt-with-key", encrypt_matches)) => {
            let value_str = encrypt_matches.get_one::<String>("value").unwrap();
            let value = u8::from_str_radix(value_str, 10).unwrap();
            let ciphertext_path = encrypt_matches.get_one::<String>("ciphertext").unwrap();

            let client_key: ClientKey;
            if let Some(lwe_sk_path) = encrypt_matches.get_one::<String>("lwe-sk") {
                client_key = keygen_from_lwe(lwe_sk_path);
            } else if let Some(client_key_path) = encrypt_matches.get_one::<String>("client-key") {
                client_key = load_client_key(client_key_path);
            } else {
                panic!("no key specified");
            }

            encrypt_with_key(value, client_key, ciphertext_path)
        }
        Some(("decrypt-with-key", decrypt_mtches)) => {
            let ciphertext_path = decrypt_mtches.get_one::<String>("ciphertext").unwrap();
            let plaintext_path = decrypt_mtches.get_one::<String>("plaintext");

            let client_key: ClientKey;
            if let Some(lwe_sk_path) = decrypt_mtches.get_one::<String>("lwe-sk") {
                client_key = keygen_from_lwe(lwe_sk_path);
            } else if let Some(client_key_path) = decrypt_mtches.get_one::<String>("client-key") {
                client_key = load_client_key(client_key_path);
            } else {
                panic!("no key specified");
            }
            decrypt_with_key(client_key, ciphertext_path, plaintext_path)
        }
        Some(("add", add_mtches)) => {
            let server_key_path = add_mtches.get_one::<String>("server-key").unwrap();
            let cts_path = add_mtches.get_many::<String>("ciphertexts").unwrap();
            let output_ct_path = add_mtches.get_one::<String>("output-ciphertext").unwrap();

            set_server_key_from_file(server_key_path);

            sum(cts_path.collect(), output_ct_path)
        }
        Some(("keygen", keygen_mtches)) => {
            let client_key_path = keygen_mtches.get_one::<String>("client-key").unwrap();
            let server_key_path = keygen_mtches.get_one::<String>("server-key").unwrap();
            let output_lwe_path = keygen_mtches.get_one::<String>("output-lwe-sk").unwrap();

            // we keygen based on an initial secret key if provided, otherwise we keygen from scratch
            if let Some(lwe_sk_path) = keygen_mtches.get_one::<String>("lwe-sk") {
                let client_key = keygen_from_lwe(lwe_sk_path);
                let server_key = client_key.generate_server_key();
                let lwe_secret_key = load_lwe_sk(lwe_sk_path);
                write_keys(
                    client_key_path,
                    server_key_path,
                    output_lwe_path,
                    Some(client_key),
                    Some(server_key),
                    Some(lwe_secret_key),
                )
            } else {
                keygen(client_key_path, server_key_path, output_lwe_path)
            }
        }
        _ => unreachable!(), // If all subcommands are defined above, anything else is unreachable
    }
}
