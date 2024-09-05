use bincode;
use clap::{Arg, ArgAction, Command};
use core::panic;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use tfhe::core_crypto::prelude::{GlweSecretKey, LweSecretKey};
use tfhe::shortint::{ClassicPBSParameters, EncryptionKeyChoice, ShortintParameterSet};
use tfhe::{generate_keys, integer, ClientKey, FheUint8};
use tfhe::{prelude::*, ConfigBuilder};

const BLOCK_PARAMS: ClassicPBSParameters = tfhe::shortint::prelude::PARAM_MESSAGE_2_CARRY_3_KS_PBS;

fn load_client_key_from_glwe(glwe_sk_path: &String) -> ClientKey {
    let path_sk: &Path = Path::new(glwe_sk_path);
    let serialized_sk = fs::read(path_sk).unwrap();
    let mut serialized_data = Cursor::new(serialized_sk);
    let sk: GlweSecretKey<Vec<u64>> = bincode::deserialize_from(&mut serialized_data).unwrap();

    let classic_pbsparameters = BLOCK_PARAMS;
    let shortint_params = ShortintParameterSet::new_pbs_param_set(
        tfhe::shortint::PBSParameters::PBS(classic_pbsparameters),
    );
    let client_key = ClientKey::from_raw_parts(
        integer::ClientKey::from_raw_parts(tfhe::shortint::ClientKey::from_raw_parts(
            sk,
            LweSecretKey::new_empty_key(0, classic_pbsparameters.lwe_dimension),
            shortint_params,
        )),
        None,
        None,
        None,
    );
    client_key
}

fn load_client_key(client_path: &String) -> ClientKey {
    let path_key: &Path = Path::new(client_path);
    let serialized_key = fs::read(path_key).unwrap();
    let mut serialized_data = Cursor::new(serialized_key);
    let client_key: ClientKey = bincode::deserialize_from(&mut serialized_data).unwrap();
    client_key
}

fn encrypt_with_key(
    value: u8,
    client_key: ClientKey,
    ciphertext_path: &String,
) -> Result<(), Box<dyn std::error::Error>> {
    let ct = FheUint8::encrypt(value, &client_key);

    let mut serialized_ct = Vec::new();
    bincode::serialize_into(&mut serialized_ct, &ct)?;
    let path_ct: &Path = Path::new(ciphertext_path);
    fs::write(path_ct, serialized_ct).unwrap();

    Ok(())
}

fn decrypt_with_key(
    client_key: ClientKey,
    ciphertext_path: &String,
    plaintext_path: Option<&String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path_fheuint: &Path = Path::new(ciphertext_path);
    let serialized_fheuint = fs::read(path_fheuint)?;
    let mut serialized_data = Cursor::new(serialized_fheuint);
    let fheuint: FheUint8 = bincode::deserialize_from(&mut serialized_data)?;

    let result: u8 = fheuint.decrypt(&client_key);

    if let Some(path) = plaintext_path {
        let pt_path: &Path = Path::new(path);
        fs::write(pt_path, result.to_string())?;
    } else {
        println!("result: {}", result);
    }

    Ok(())
}

fn keygen(
    client_key_path: &String,
    server_key_path: &String,
    output_glwe_path: &String,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::with_custom_parameters(BLOCK_PARAMS, None).build();

    let (client_key, server_key) = generate_keys(config);
    let (integer_ck, _, _, _) = client_key.clone().into_raw_parts();
    let shortint_ck = integer_ck.into_raw_parts();
    assert!(BLOCK_PARAMS.encryption_key_choice == EncryptionKeyChoice::Big);
    let (glwe_secret_key, _, _) = shortint_ck.into_raw_parts();

    let mut serialized_client_key = Vec::new();
    let mut serialized_server_key = Vec::new();
    let mut serialized_glwe_key = Vec::new();
    bincode::serialize_into(&mut serialized_client_key, &client_key)?;
    bincode::serialize_into(&mut serialized_server_key, &server_key)?;
    bincode::serialize_into(&mut serialized_glwe_key, &glwe_secret_key)?;

    let path_client_key: &Path = Path::new(client_key_path);
    let path_server_key: &Path = Path::new(server_key_path);
    let path_glwe_key: &Path = Path::new(output_glwe_path);
    fs::write(path_client_key, serialized_client_key).unwrap();
    fs::write(path_server_key, serialized_server_key).unwrap();
    fs::write(path_glwe_key, serialized_glwe_key).unwrap();
    Ok(())
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
                        .num_args(1..),
                )
                .arg(
                    Arg::new("ciphertext")
                        .short('c')
                        .long("ciphertext")
                        .help("ciphertext path to write to after encryption")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1..),
                )
                .arg(
                    Arg::new("client-key")
                        .long("client-key")
                        .short('l')
                        .conflicts_with("glwe-sk")
                        .help("client key path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1..),
                )
                .arg(
                    Arg::new("glwe-sk")
                        .long("glwe-sk")
                        .short('g')
                        .conflicts_with("client-key")
                        .help("glwe key path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1..),
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
                        .num_args(1..),
                )
                .arg(
                    Arg::new("plaintext")
                        .long("plaintext")
                        .short('p')
                        .help("output plaintext path")
                        .action(ArgAction::Set)
                        .num_args(1..),
                )
                .arg(
                    Arg::new("client-key")
                        .long("client-key")
                        .short('l')
                        .conflicts_with("glwe-sk")
                        .help("client key path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1..),
                )
                .arg(
                    Arg::new("glwe-sk")
                        .long("glwe-sk")
                        .short('g')
                        .conflicts_with("client-key")
                        .help("glwe key path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1..),
                ),
        )
        .subcommand(
            Command::new("keygen")
                .short_flag('k')
                .long_flag("keygen")
                .about("Generate client and server key.")
                // TODO: generate from a glwe sk
                // .arg(
                //     Arg::new("glwe-sk")
                //         .long("glwe-sk")
                //         .short('g')
                //         .conflicts_with("client-key")
                //         .help("glwe key path")
                //         .action(ArgAction::Set)
                //         .required(true)
                //         .num_args(1..),
                // )
                .arg(
                    Arg::new("output-glwe-sk")
                        .long("output-glwe-sk")
                        .short('g')
                        .default_value("output-glwe-sk")
                        .help("output glwe key path")
                        .action(ArgAction::Set)
                        .num_args(1..),
                )
                .arg(
                    Arg::new("client-key")
                        .short('c')
                        .default_value("client_key")
                        .long("client-key")
                        .help("output client key path")
                        .action(ArgAction::Set)
                        .num_args(1..),
                )
                .arg(
                    Arg::new("server-key")
                        .short('s')
                        .default_value("server_key")
                        .long("server-key")
                        .help("output server key path")
                        .action(ArgAction::Set)
                        .num_args(1..),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("encrypt-with-key", encrypt_matches)) => {
            let value_str = encrypt_matches.get_one::<String>("value").unwrap();
            let value = u8::from_str_radix(value_str, 10).unwrap();
            let ciphertext_path = encrypt_matches.get_one::<String>("ciphertext").unwrap();

            let client_key: ClientKey;
            if let Some(glwe_sk_path) = encrypt_matches.get_one::<String>("glwe-sk") {
                client_key = load_client_key_from_glwe(glwe_sk_path);
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
            if let Some(glwe_sk_path) = decrypt_mtches.get_one::<String>("glwe-sk") {
                client_key = load_client_key_from_glwe(glwe_sk_path);
            } else if let Some(client_key_path) = decrypt_mtches.get_one::<String>("client-key") {
                client_key = load_client_key(client_key_path);
            } else {
                panic!("no key specified");
            }
            decrypt_with_key(client_key, ciphertext_path, plaintext_path)
        }
        Some(("keygen", keygen_mtches)) => {
            let client_key_path = keygen_mtches.get_one::<String>("client-key").unwrap();
            let server_key_path = keygen_mtches.get_one::<String>("server-key").unwrap();
            let output_glwe_path = keygen_mtches.get_one::<String>("output-glwe-sk").unwrap();

            keygen(client_key_path, server_key_path, output_glwe_path)
        }
        _ => unreachable!(), // If all subcommands are defined above, anything else is unreachable
    }
}
