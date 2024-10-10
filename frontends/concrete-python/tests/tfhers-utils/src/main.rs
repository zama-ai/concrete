use clap::{Arg, ArgAction, Command};
use core::panic;
use serde::de::DeserializeOwned;
use std::fs;
use std::path::Path;
use tfhe::core_crypto::prelude::LweSecretKey;
use tfhe::prelude::*;
use tfhe::shortint::{ClassicPBSParameters, EncryptionKeyChoice};
use tfhe::{generate_keys, set_server_key, ClientKey, ConfigBuilder, FheInt8, FheUint8, ServerKey};

use serde::Serialize;
use tfhe::named::Named;
use tfhe::{Unversionize, Versionize};

use tfhe::safe_serialization::{safe_deserialize, safe_serialize};

const BLOCK_PARAMS: ClassicPBSParameters = tfhe::shortint::prelude::PARAM_MESSAGE_2_CARRY_3_KS_PBS;
const SERIALIZE_SIZE_LIMIT: u64 = 1_000_000_000;

// safe_save write to a path a value that implement the tfhe-rs safe serialization
fn safe_save<T: Serialize + Versionize + Named>(path: &String, value: &T) {
    let file = fs::File::create(path).unwrap();
    safe_serialize(value, file, SERIALIZE_SIZE_LIMIT).unwrap()
}

// safe_load read from a path a value that implement the tfhe-rs safe serialization
fn safe_load<T: DeserializeOwned + Unversionize + Named>(path: &String) -> T {
    let file = fs::File::open(path).unwrap();
    safe_deserialize(file, SERIALIZE_SIZE_LIMIT).unwrap()
}

// unsafe_save write to a path a value that NOT implement the tfhe-rs safe serialization
// TODO: Remove me when all object implemennt tfhe-rs safe serialization
fn unsafe_save<T: Serialize>(path: &String, value: &T) {
    let file = fs::File::create(path).unwrap();
    bincode::serialize_into(file, value).unwrap()
}

// unsafe_load read from a path a value that NOT implement the tfhe-rs safe serialization
// TODO: Remove me when all object implemennt tfhe-rs safe serialization
fn unsafe_load<T: DeserializeOwned>(path: &String) -> T {
    let file = fs::File::open(path).unwrap();
    bincode::deserialize_from(file).unwrap()
}

fn set_server_key_from_file(path: &String) {
    let sk: ServerKey = safe_load(path);
    set_server_key(sk);
}

fn encrypt_with_key_u8(
    value: u8,
    client_key: ClientKey,
    ciphertext_path: &String,
) -> Result<(), Box<dyn std::error::Error>> {
    let ct = FheUint8::encrypt(value, &client_key);

    safe_save(ciphertext_path, &ct);

    Ok(())
}

fn encrypt_with_key_i8(
    value: i8,
    client_key: ClientKey,
    ciphertext_path: &String,
) -> Result<(), Box<dyn std::error::Error>> {
    let ct = FheInt8::encrypt(value, &client_key);

    safe_save(ciphertext_path, &ct);

    Ok(())
}

fn decrypt_with_key(
    client_key: ClientKey,
    ciphertext_path: &String,
    plaintext_path: Option<&String>,
    signed: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let string_result: String;

    if signed {
        let fheint: FheInt8 = safe_load(ciphertext_path);
        let result: i8 = fheint.decrypt(&client_key);
        string_result = result.to_string();
    } else {
        let fheuint: FheUint8 = safe_load(ciphertext_path);
        let result: u8 = fheuint.decrypt(&client_key);
        string_result = result.to_string();
    }

    if let Some(path) = plaintext_path {
        let pt_path: &Path = Path::new(path);
        fs::write(pt_path, string_result)?;
    } else {
        println!("result: {}", string_result);
    }
    Ok(())
}

fn sum(cts_paths: Vec<&String>, out_ct_path: &String) {
    if cts_paths.is_empty() {
        panic!("can't call sum with 0 ciphertexts");
    }
    let mut acc: FheUint8 = safe_load(cts_paths[0]);
    for ct_path in cts_paths[1..].iter() {
        let fheuint: FheUint8 = safe_load(ct_path);
        acc += fheuint;
    }
    safe_save(out_ct_path, &acc)
}

fn write_keys(
    client_key_path: &String,
    server_key_path: &String,
    output_lwe_path: &String,
    client_key: Option<ClientKey>,
    server_key: Option<ServerKey>,
    lwe_secret_key: Option<LweSecretKey<Vec<u64>>>,
) {
    if let Some(ck) = client_key {
        safe_save(client_key_path, &ck)
    }

    if let Some(sk) = server_key {
        safe_save(server_key_path, &sk)
    }

    if let Some(lwe_sk) = lwe_secret_key {
        unsafe_save(output_lwe_path, &lwe_sk)
    }
}

fn keygen(client_key_path: &String, server_key_path: &String, output_lwe_path: &String) {
    let config = ConfigBuilder::with_custom_parameters(BLOCK_PARAMS).build();

    let (client_key, server_key) = generate_keys(config);
    let (integer_ck, _, _, _) = client_key.clone().into_raw_parts();
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
    )
}

fn keygen_from_lwe(lwe_sk_path: &String) -> ClientKey {
    let lwe_sk = unsafe_load(lwe_sk_path);

    let shortint_key =
        tfhe::shortint::ClientKey::try_from_lwe_encryption_key(lwe_sk, BLOCK_PARAMS).unwrap();
    ClientKey::from_raw_parts(shortint_key.into(), None, None, tfhe::Tag::default())
}

fn main() {
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
                    Arg::new("signed")
                        .long("signed")
                        .help("encrypt as a signed integer")
                        .action(ArgAction::SetTrue),
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
                    Arg::new("signed")
                        .long("signed")
                        .help("decrypt as a signed integer")
                        .action(ArgAction::SetTrue),
                )
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
            let ciphertext_path = encrypt_matches.get_one::<String>("ciphertext").unwrap();
            let signed = encrypt_matches.get_flag("signed");

            let client_key: ClientKey;
            if let Some(lwe_sk_path) = encrypt_matches.get_one::<String>("lwe-sk") {
                client_key = keygen_from_lwe(lwe_sk_path);
            } else if let Some(client_key_path) = encrypt_matches.get_one::<String>("client-key") {
                client_key = safe_load(client_key_path);
            } else {
                panic!("no key specified");
            }

            if signed {
                let value: i8 = value_str.parse().unwrap();
                encrypt_with_key_i8(value, client_key, ciphertext_path).unwrap()
            } else {
                let value: u8 = value_str.parse().unwrap();
                encrypt_with_key_u8(value, client_key, ciphertext_path).unwrap()
            }
        }
        Some(("decrypt-with-key", decrypt_mtches)) => {
            let ciphertext_path = decrypt_mtches.get_one::<String>("ciphertext").unwrap();
            let plaintext_path = decrypt_mtches.get_one::<String>("plaintext");
            let signed = decrypt_mtches.get_flag("signed");

            let client_key: ClientKey;
            if let Some(lwe_sk_path) = decrypt_mtches.get_one::<String>("lwe-sk") {
                client_key = keygen_from_lwe(lwe_sk_path);
            } else if let Some(client_key_path) = decrypt_mtches.get_one::<String>("client-key") {
                client_key = safe_load(client_key_path);
            } else {
                panic!("no key specified");
            }
            decrypt_with_key(client_key, ciphertext_path, plaintext_path, signed).unwrap()
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
                let lwe_secret_key = unsafe_load(lwe_sk_path);
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
