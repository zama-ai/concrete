use clap::{Arg, Command};
use concrete_rust::{generate_keyset, read_secret_key_from_file};

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
    let initial_secret_keys_paths: Vec<String> = matches
        .get_many::<String>("initial-secret-key")
        .unwrap_or_default()
        .map(|s| s.to_string())
        .collect();
    let initial_secret_keys: std::collections::HashMap<
        u32,
        capnp::message::Reader<capnp::serialize::OwnedSegments>,
    > = initial_secret_keys_paths
        .iter()
        .map(|s| {
            let (id, reader) = read_secret_key_from_file(s);
            (id, reader)
        })
        .collect();

    generate_keyset(
        keyset_info_path,
        sec_seed,
        enc_seed,
        keyset_path,
        &initial_secret_keys,
    );
}
