use std::io::{Read, Write};

use clap::{Arg, Command};
use concrete_keygen::generate_keyset_from_buffers;

pub fn main() {
    let matches = Command::new("concrete-keygen")
        .about("Concrete Keygen: generate keys for Concrete using a keyset info.")
        .arg_required_else_help(true)
        .arg(
            Arg::new("keyset-info")
                .help("Path to the input keyset info file")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("keyset")
                .help("Path to the output keyset file")
                .required(true)
                .index(2),
        )
        .arg(
            Arg::new("sec-seed")
                .help("Secret seed for key generation")
                .required(false)
                .default_value("0")
                .long("secret-seed"),
        )
        .arg(
            Arg::new("enc-seed")
                .help("Encryption seed for key generation")
                .required(false)
                .default_value("0")
                .long("enc-seed"),
        )
        .arg(
            Arg::new("no-bootstrapping-keys")
                .help("Do not generate bootstrapping keys")
                .required(false)
                .long("no-bsk")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("ignore-bsk")
                .help("Ignore a specific bootstrapping key by id")
                .required(false)
                .long("ignore-bsk")
                .num_args(1)
                .action(clap::ArgAction::Append),
        )
        .arg(
            Arg::new("no-keyswitching-keys")
                .help("Do not generate keyswitching keys")
                .required(false)
                .long("no-ksk")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("ignore-ksk")
                .help("Ignore a specific keyswitching key by id")
                .required(false)
                .long("ignore-ksk")
                .num_args(1)
                .action(clap::ArgAction::Append),
        )
        .arg(
            Arg::new("initial-secret-key")
                .help("Initial secret key for key generation")
                .required(false)
                .long("sk")
                .num_args(1)
                .action(clap::ArgAction::Append),
        )
        .get_matches();

    let keyset_info_path = matches.get_one::<String>("keyset-info").unwrap();
    let keyset_path = matches.get_one::<String>("keyset").unwrap();
    let sec_seed = matches
        .get_one::<String>("sec-seed")
        .unwrap()
        .parse::<u128>()
        .expect("Invalid secret seed");
    let enc_seed = matches
        .get_one::<String>("enc-seed")
        .unwrap()
        .parse::<u128>()
        .expect("Invalid encryption seed");
    let no_bootstrapping_keys = matches.get_one::<bool>("no-bootstrapping-keys").unwrap();
    let ignore_bootstrapping_keys = matches
        .get_many::<u32>("ignore-bsk")
        .unwrap_or_default()
        .cloned()
        .collect();
    let no_keyswitching_keys = matches.get_one::<bool>("no-keyswitching-keys").unwrap();
    let ignore_keyswitching_keys = matches
        .get_many::<u32>("ignore-ksk")
        .unwrap_or_default()
        .cloned()
        .collect();
    let initial_secret_keys: Vec<Vec<u8>> = matches
        .get_many::<String>("initial-secret-key")
        .unwrap_or_default()
        .map(|s| {
            let mut file = std::fs::File::open(s).expect("Failed to open initial secret key file");
            let mut buffer: Vec<u8> = vec![];
            file.read_to_end(&mut buffer)
                .expect("Failed to read initial secret key file");
            buffer
        })
        .collect();

    let mut keyset_info_file =
        std::fs::File::open(keyset_info_path).expect("Failed to open keyset info file");
    let keyset_info_file_metadata =
        std::fs::metadata(&keyset_info_path).expect("unable to read metadata");
    let mut keyset_info_buffer: Vec<u8> = vec![0; keyset_info_file_metadata.len() as usize];
    keyset_info_file
        .read(&mut keyset_info_buffer)
        .expect("buffer overflow");

    let keyset_buffer = generate_keyset_from_buffers(
        keyset_info_buffer.as_slice(),
        sec_seed,
        enc_seed,
        *no_bootstrapping_keys,
        ignore_bootstrapping_keys,
        *no_keyswitching_keys,
        ignore_keyswitching_keys,
        &initial_secret_keys.iter().map(|s| s.as_slice()).collect(),
    );
    let mut keyset_file = std::fs::File::create(keyset_path).expect("Failed to create keyset file");
    keyset_file
        .write(keyset_buffer.as_slice())
        .expect("Failed to write keyset file");
}
