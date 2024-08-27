use bincode;
use clap::{Arg, ArgAction, Command};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use tfhe::core_crypto::prelude::{GlweSecretKey, LweSecretKey};
use tfhe::prelude::*;
use tfhe::shortint::ShortintParameterSet;
use tfhe::{integer, ClientKey, FheUint8};

fn load_client_key(glwe_sk_path: &String) -> ClientKey {
    let path_sk: &Path = Path::new(glwe_sk_path);
    let serialized_sk = fs::read(path_sk).unwrap();
    let mut serialized_data = Cursor::new(serialized_sk);
    let sk: GlweSecretKey<Vec<u64>> = bincode::deserialize_from(&mut serialized_data).unwrap();

    let classic_pbsparameters = tfhe::shortint::prelude::PARAM_MESSAGE_2_CARRY_3_KS_PBS;
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

fn encrypt_with_key(
    value: u8,
    glwe_sk_path: &String,
    ciphertext_path: &String,
) -> Result<(), Box<dyn std::error::Error>> {
    let client_key = load_client_key(glwe_sk_path);

    let ct = FheUint8::encrypt(value, &client_key);

    let mut serialized_ct = Vec::new();
    bincode::serialize_into(&mut serialized_ct, &ct)?;
    let path_ct: &Path = Path::new(ciphertext_path);
    fs::write(path_ct, serialized_ct).unwrap();

    Ok(())
}

fn decrypt_with_key(
    glwe_sk_path: &String,
    ciphertext_path: &String,
    plaintext_path: Option<&String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client_key = load_client_key(glwe_sk_path);

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
                    Arg::new("glwe-sk")
                        .long("glwe-sk")
                        .short('g')
                        // .conflicts_with("other-key")
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
                    Arg::new("glwe-sk")
                        .long("glwe-sk")
                        .short('g')
                        // .conflicts_with("other-key")
                        .help("glwe key path")
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1..),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("encrypt-with-key", encrypt_matches)) => {
            let value_str = encrypt_matches.get_one::<String>("value").unwrap();
            let value = u8::from_str_radix(value_str, 10).unwrap();
            let ciphertext_path = encrypt_matches.get_one::<String>("ciphertext").unwrap();
            let glwe_sk_path = encrypt_matches.get_one::<String>("glwe-sk").unwrap();

            encrypt_with_key(value, glwe_sk_path, ciphertext_path)
        }
        Some(("decrypt-with-key", decrypt_mtches)) => {
            let ciphertext_path = decrypt_mtches.get_one::<String>("ciphertext").unwrap();
            let glwe_sk_path = decrypt_mtches.get_one::<String>("glwe-sk").unwrap();

            let plaintext_path = decrypt_mtches.get_one::<String>("plaintext");

            decrypt_with_key(glwe_sk_path, ciphertext_path, plaintext_path)
        }
        _ => unreachable!(), // If all subcommands are defined above, anything else is unreachable
    }
}
