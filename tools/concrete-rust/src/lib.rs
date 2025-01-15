//! This module provides functionality for generating cryptographic keys for the Concrete library.
//! It includes functions for generating secret keys, bootstrap keys, keyswitch keys, and packing keyswitch keys.
//! The keys are generated based on the provided keyset information and seeds for randomness.
//!
//! # Modules
//! - `concrete_protocol_capnp`: Contains the Cap'n Proto schema for the Concrete protocol.
//!
//! # Constants
//! - `MAX_TEXT_SIZE`: The maximum text size for Cap'n Proto messages.
//!
//! # Functions
//! - `u64_slice_to_u8_vector`: Converts a slice of `u64` values to a `Vec<u8>`.
//! - `u8_slice_to_u64_vector`: Converts a slice of `u8` values to a `Vec<u64>`.
//! - `reader_to_lwe_secret_key`: Converts a Cap'n Proto reader to an `LweSecretKey`.
//! - `vector_to_payload`: Converts a slice of `u64` values to a Cap'n Proto payload message.
//! - `generate_sk`: Generates an `LweSecretKey` with the specified dimension and random generator.
//! - `generate_bsk`: Generates an `LweBootstrapKey` with the specified parameters and random generator.
//! - `generate_ksk`: Generates an `LweKeyswitchKey` with the specified parameters and random generator.
//! - `generate_pksk`: Generates an `LwePrivateFunctionalPackingKeyswitchKeyList` with the specified parameters and random generator.
//! - `generate_keyset_message`: Generates a Cap'n Proto message containing the keyset based on the provided keyset information and seeds.
//! - `generate_keyset`: Reads keyset information from a file, generates the keyset, and writes it to an output file.
//!
//! # Macros
//! - `build_key`: A macro for building a Cap'n Proto message for a key with the specified key information and builder type.
use std::u128;

use crate::concrete_protocol_capnp::*;
use capnp::message::{HeapAllocator, ReaderOptions};
use capnp::serialize;
use concrete_csprng::generators::SoftwareRandomGenerator;
use tfhe::core_crypto::prelude::{
    generate_lwe_keyswitch_key, par_generate_circuit_bootstrap_lwe_pfpksk_list,
    par_generate_lwe_bootstrap_key, CiphertextModulus, EncryptionRandomGenerator,
    FunctionalPackingKeyswitchKeyCount, Gaussian, GlweSecretKey, LweBootstrapKey, LweKeyswitchKey,
    LwePrivateFunctionalPackingKeyswitchKeyList, LweSecretKey, SecretRandomGenerator, Variance,
};
use tfhe::core_crypto::seeders::new_seeder;
use tfhe::shortint::prelude::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use tfhe::Seed;

mod concrete_protocol_capnp {
    include!(concat!(
        env!("OUT_DIR"),
        "/capnp/concrete_protocol_capnp.rs"
    ));
}

// FIXME: this is the value of capnp::MAX_TEXT_SIZE in Cpp in my setup, but I can't seem to find a similar API in Rust
const MAX_TEXT_SIZE: usize = 536870910usize;

/// Converts a slice of `u64` values to a `Vec<u8>`.
///
/// This function takes a slice of `u64` values and converts each `u64` value to its little-endian
/// byte representation, then appends these bytes to a `Vec<u8>`.
fn u64_slice_to_u8_vector(data: &[u64]) -> Vec<u8> {
    let mut output = Vec::new();
    for i in data {
        // we assert on the Cpp side that we use LE
        output.extend_from_slice(&i.to_le_bytes());
    }
    output
}

/// Converts a slice of `u8` values to a `Vec<u64>`.
///
/// This function takes a slice of `u8` values and converts each group of 8 bytes to a `u64` value
/// using little-endian byte order.
fn u8_slice_to_u64_vector(data: &[u8]) -> Vec<u64> {
    let mut output = Vec::new();
    for i in data.chunks(std::mem::size_of::<u64>()) {
        output.push(u64::from_le_bytes(i.try_into().unwrap()));
    }
    output
}

/// Converts a Cap'n Proto reader to an `LweSecretKey`.
fn reader_to_lwe_secret_key(reader: &lwe_secret_key::Reader) -> LweSecretKey<Vec<u64>> {
    let payload = reader.get_payload().unwrap();
    let mut sk_data = Vec::new();
    let data_list = payload.get_data().unwrap();
    for data in data_list.iter() {
        sk_data.extend_from_slice(&u8_slice_to_u64_vector(data.unwrap()));
    }
    LweSecretKey::from_container(sk_data)
}

/// Converts a slice of `u64` values to a Cap'n Proto payload message.
fn vector_to_payload(data_u64: &[u64]) -> capnp::message::Builder<HeapAllocator> {
    let data = u64_slice_to_u8_vector(data_u64);
    let mut builder = capnp::message::Builder::new_default();
    let payload_builder = builder.init_root::<payload::Builder>();
    let elms_per_blob = MAX_TEXT_SIZE / std::mem::size_of::<u8>();
    let remaining_elms = data.len() % elms_per_blob;
    let nb_blobs = (data.len() / elms_per_blob) + (remaining_elms > 0) as usize;
    let mut data_builder = payload_builder.init_data(nb_blobs as u32);
    for (i, blob) in data.chunks(elms_per_blob).enumerate() {
        data_builder.set(i as u32, blob);
    }
    builder
}

/// Generates an `LweSecretKey` with the specified dimension and random generator.
fn generate_sk(
    lwe_dim: usize,
    mut secret_random_generator: &mut SecretRandomGenerator<SoftwareRandomGenerator>,
) -> LweSecretKey<Vec<u64>> {
    let mut sk: LweSecretKey<Vec<u64>> = LweSecretKey::new_empty_key(0, LweDimension(lwe_dim));
    tfhe::core_crypto::algorithms::generate_binary_lwe_secret_key(
        &mut sk,
        &mut secret_random_generator,
    );
    sk
}

/// Generates an `LweBootstrapKey` with the specified parameters and random generator.
fn generate_bsk(
    compression: Compression,
    enc_random_generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
    input_sk: &LweSecretKey<Vec<u64>>,
    output_sk: &LweSecretKey<Vec<u64>>,
    input_lwe_dimension: usize,
    output_polynomial_size: usize,
    output_glwe_dimension: usize,
    decomp_level_count: usize,
    decomp_base_log: usize,
    variance: f64,
) -> LweBootstrapKey<Vec<u64>> {
    // TODO: support compressed keys
    assert!(compression == Compression::None);
    let mut bsk = LweBootstrapKey::new(
        0,
        GlweDimension(output_glwe_dimension).to_glwe_size(),
        PolynomialSize(output_polynomial_size),
        DecompositionBaseLog(decomp_base_log),
        DecompositionLevelCount(decomp_level_count),
        LweDimension(input_lwe_dimension),
        CiphertextModulus::new_native(),
    );

    let output_glwe_sk = GlweSecretKey::from_container(
        output_sk.clone().into_container(),
        PolynomialSize(output_polynomial_size),
    );
    par_generate_lwe_bootstrap_key(
        input_sk,
        &output_glwe_sk,
        &mut bsk,
        Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
        enc_random_generator,
    );
    bsk
}

/// Generates an `LweKeyswitchKey` with the specified parameters and random generator.
fn generate_ksk(
    compression: Compression,
    enc_random_generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
    input_sk: &LweSecretKey<Vec<u64>>,
    output_sk: &LweSecretKey<Vec<u64>>,
    input_lwe_dimension: usize,
    output_lwe_dimension: usize,
    decomp_level_count: usize,
    decomp_base_log: usize,
    variance: f64,
) -> LweKeyswitchKey<Vec<u64>> {
    // TODO: support compressed keys
    assert!(compression == Compression::None);
    let mut ksk: LweKeyswitchKey<Vec<u64>> = LweKeyswitchKey::new(
        0,
        DecompositionBaseLog(decomp_base_log),
        DecompositionLevelCount(decomp_level_count),
        LweDimension(input_lwe_dimension),
        LweDimension(output_lwe_dimension),
        CiphertextModulus::new_native(),
    );
    generate_lwe_keyswitch_key(
        input_sk,
        output_sk,
        &mut ksk,
        Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
        enc_random_generator,
    );
    ksk
}

/// Generates an `LwePrivateFunctionalPackingKeyswitchKeyList` with the specified parameters and random generator.
fn generate_pksk(
    enc_random_generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
    input_sk: &LweSecretKey<Vec<u64>>,
    output_sk: &LweSecretKey<Vec<u64>>,
    input_lwe_dimension: usize,
    output_glwe_dimension: usize,
    output_polynomial_size: usize,
    decomp_level_count: usize,
    decomp_base_log: usize,
    variance: f64,
) -> LwePrivateFunctionalPackingKeyswitchKeyList<Vec<u64>> {
    let output_glwe_size = GlweDimension(output_glwe_dimension).to_glwe_size();
    let mut fpksk_list: LwePrivateFunctionalPackingKeyswitchKeyList<Vec<u64>> =
        LwePrivateFunctionalPackingKeyswitchKeyList::new(
            0,
            DecompositionBaseLog(decomp_base_log),
            DecompositionLevelCount(decomp_level_count),
            LweDimension(input_lwe_dimension),
            output_glwe_size,
            PolynomialSize(output_polynomial_size),
            FunctionalPackingKeyswitchKeyCount(output_glwe_size.0),
            CiphertextModulus::new_native(),
        );
    let output_glwe_sk = GlweSecretKey::from_container(
        output_sk.clone().into_container(),
        PolynomialSize(output_polynomial_size),
    );
    par_generate_circuit_bootstrap_lwe_pfpksk_list(
        &mut fpksk_list,
        input_sk,
        &output_glwe_sk,
        Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
        enc_random_generator,
    );

    fpksk_list
}

/// Build a Cap'n Proto message containing a specific key.
///
/// The key is generated using the specified key data, key information, and builder type.
macro_rules! build_key {
    ($key:expr, $key_info:expr, $builder_type:ty) => {{
        let mut key_builder = capnp::message::Builder::new_default();
        let mut key_root_builder = key_builder.init_root::<$builder_type>();
        let key_data = $key.into_container();
        key_root_builder.set_info($key_info).unwrap();
        key_root_builder
            .set_payload(
                vector_to_payload(key_data.as_slice())
                    .get_root_as_reader()
                    .unwrap(),
            )
            .unwrap();
        key_builder
    }};
}

/// Generates a Cap'n Proto message containing the keyset based on the provided keyset information and seeds.
///
/// The function generates secret keys, bootstrap keys, keyswitch keys, and packing keyswitch keys based on the keyset information.
/// It uses initial secret keys instead of generating them if they are provided.
///
/// # Arguments
///
/// * `info`: The keyset information.
/// * `secret_seed`: The seed for the secret random generator.
/// * `enc_seed`: The seed for the encryption random generator.
/// * `init_lwe_secret_keys`: A map of initial secret keys to use instead of generating them.
///
/// # Returns
/// A Cap'n Proto message containing the generated keyset.
fn generate_keyset_message(
    info: keyset_info::Reader,
    secret_seed: u128,
    enc_seed: u128,
    init_lwe_secret_keys: &mut std::collections::HashMap<u32, lwe_secret_key::Reader>,
) -> capnp::message::Builder<HeapAllocator> {
    let mut builder = capnp::message::Builder::new_default();
    let mut keyset_builder = builder.init_root::<keyset::Builder>();

    // Random generators
    let mut secret_random_generator: SecretRandomGenerator<SoftwareRandomGenerator> =
        SecretRandomGenerator::new(Seed(secret_seed));
    let mut boxed_seeder = new_seeder();
    let seeder = boxed_seeder.as_mut();
    let mut enc_random_generator: EncryptionRandomGenerator<SoftwareRandomGenerator> =
        EncryptionRandomGenerator::new(Seed(enc_seed), seeder);

    let mut client_keyset = keyset_builder.reborrow().init_client();
    client_keyset
        .reborrow()
        .init_lwe_secret_keys(info.get_lwe_secret_keys().unwrap().len());
    // Generate secret keys
    let mut generated_secret_keys: std::collections::HashMap<u32, LweSecretKey<Vec<u64>>> =
        std::collections::HashMap::new();
    for sk_info in info.get_lwe_secret_keys().unwrap().iter() {
        if init_lwe_secret_keys.contains_key(&sk_info.get_id()) {
            // we initialize the secret key with the provided secret key
            let sk_reader = init_lwe_secret_keys.remove(&sk_info.get_id()).unwrap();
            generated_secret_keys.insert(sk_info.get_id(), reader_to_lwe_secret_key(&sk_reader));
            client_keyset
                .reborrow()
                .get_lwe_secret_keys()
                .unwrap()
                .set_with_caveats(sk_info.get_id(), sk_reader)
                .unwrap();
        } else {
            // we generate a new secret key
            let lwe_dim = sk_info.get_params().unwrap().get_lwe_dimension() as usize;
            let sk = generate_sk(lwe_dim, &mut secret_random_generator);
            generated_secret_keys.insert(sk_info.get_id(), sk.clone());
            let sk_builder = build_key!(sk, sk_info, lwe_secret_key::Builder);
            client_keyset
                .reborrow()
                .get_lwe_secret_keys()
                .unwrap()
                .set_with_caveats(sk_info.get_id(), sk_builder.get_root_as_reader().unwrap())
                .unwrap();
        }
    }

    let mut server_keyset = keyset_builder.init_server();
    server_keyset
        .reborrow()
        .init_lwe_bootstrap_keys(info.get_lwe_bootstrap_keys().unwrap().len());
    server_keyset
        .reborrow()
        .init_lwe_keyswitch_keys(info.get_lwe_keyswitch_keys().unwrap().len());
    server_keyset
        .reborrow()
        .init_packing_keyswitch_keys(info.get_packing_keyswitch_keys().unwrap().len());
    // Generate bootstrap keys
    for bsk_info in info.get_lwe_bootstrap_keys().unwrap().iter() {
        // we generate a new bootstrapping key
        let params = bsk_info.get_params().unwrap();
        let input_lwe_dim = params.get_input_lwe_dimension() as usize;
        let glwe_dim = params.get_glwe_dimension() as usize;
        let poly_size = params.get_polynomial_size() as usize;
        let decomp_level_count = params.get_level_count() as usize;
        let decomp_base_log = params.get_base_log() as usize;
        let variance = params.get_variance();
        let compression = bsk_info.get_compression().unwrap();
        let bsk = generate_bsk(
            compression,
            &mut enc_random_generator,
            generated_secret_keys.get(&bsk_info.get_input_id()).unwrap(),
            generated_secret_keys
                .get(&&bsk_info.get_output_id())
                .unwrap(),
            input_lwe_dim,
            poly_size,
            glwe_dim,
            decomp_level_count,
            decomp_base_log,
            variance,
        );
        let bsk_builder = build_key!(bsk, bsk_info, lwe_bootstrap_key::Builder);
        server_keyset
            .reborrow()
            .get_lwe_bootstrap_keys()
            .unwrap()
            .set_with_caveats(bsk_info.get_id(), bsk_builder.get_root_as_reader().unwrap())
            .unwrap();
    }
    // Generate keyswitch keys
    for ksk_info in info.get_lwe_keyswitch_keys().unwrap().iter() {
        // we generate a new keyswitching key
        let params = ksk_info.get_params().unwrap();
        let input_lwe_dimension = params.get_input_lwe_dimension() as usize;
        let output_lwe_dimension = params.get_output_lwe_dimension() as usize;
        let decomp_level_count = params.get_level_count() as usize;
        let decomp_base_log = params.get_base_log() as usize;
        let variance = params.get_variance();
        let compression = ksk_info.get_compression().unwrap();
        let ksk = generate_ksk(
            compression,
            &mut enc_random_generator,
            generated_secret_keys.get(&ksk_info.get_input_id()).unwrap(),
            generated_secret_keys
                .get(&&ksk_info.get_output_id())
                .unwrap(),
            input_lwe_dimension,
            output_lwe_dimension,
            decomp_level_count,
            decomp_base_log,
            variance,
        );
        let ksk_builder = build_key!(ksk, ksk_info, lwe_keyswitch_key::Builder);
        server_keyset
            .reborrow()
            .get_lwe_keyswitch_keys()
            .unwrap()
            .set_with_caveats(ksk_info.get_id(), ksk_builder.get_root_as_reader().unwrap())
            .unwrap();
    }
    // Generate packing keyswitch keys
    for pksk_info in info.get_packing_keyswitch_keys().unwrap().iter() {
        // we generate a new packing keyswitch key
        let params = pksk_info.get_params().unwrap();
        let input_lwe_dimension = params.get_input_lwe_dimension() as usize;
        let glwe_dimension = params.get_glwe_dimension() as usize;
        let decomp_level_count = params.get_level_count() as usize;
        let decomp_base_log = params.get_base_log() as usize;
        let poly_size = params.get_polynomial_size() as usize;
        let variance = params.get_variance();
        let pksk = generate_pksk(
            &mut enc_random_generator,
            generated_secret_keys
                .get(&pksk_info.get_input_id())
                .unwrap(),
            generated_secret_keys
                .get(&&pksk_info.get_output_id())
                .unwrap(),
            input_lwe_dimension,
            glwe_dimension,
            poly_size,
            decomp_level_count,
            decomp_base_log,
            variance,
        );
        let pksk_builder = build_key!(pksk, pksk_info, packing_keyswitch_key::Builder);
        server_keyset
            .reborrow()
            .get_packing_keyswitch_keys()
            .unwrap()
            .set_with_caveats(
                pksk_info.get_id(),
                pksk_builder.get_root_as_reader().unwrap(),
            )
            .unwrap();
    }

    builder
}

/// Generate a Concrete keyset based on the provided keyset information and seeds.
///
/// The function generates secret keys, bootstrap keys, keyswitch keys, and packing keyswitch keys based on the keyset information.
/// It uses initial secret keys instead of generating them if they are provided.
/// # Arguments
///
/// * `keyset_info_path`: The path to the keyset information file.
/// * `secret_seed`: The seed for the secret random generator.
/// * `enc_seed`: The seed for the encryption random generator.
/// * `keyset_path`: The path to the output keyset file.
/// * `init_lwe_secret_keys`: A map of initial secret keys to use instead of generating them.
///
/// # Panics
/// This function panics if it fails to open the keyset info file or create the keyset file.
pub fn generate_keyset(
    keyset_info_path: &str,
    secret_seed: u128,
    enc_seed: u128,
    keyset_path: &str,
    init_lwe_secret_keys: std::collections::HashMap<u32, lwe_secret_key::Reader>,
) {
    // read keyset info from input file
    let file = std::fs::File::open(keyset_info_path).expect("Failed to open keyset info file");
    let reader =
        serialize::read_message(std::io::BufReader::new(file), ReaderOptions::new()).unwrap();
    let key_set_info = reader.get_root::<keyset_info::Reader>().unwrap();
    // keygen
    let builder = generate_keyset_message(
        key_set_info,
        secret_seed,
        enc_seed,
        &mut init_lwe_secret_keys.clone(),
    );
    // write keyset to output file
    let output = std::fs::File::create(keyset_path).expect("Failed to create keyset file");
    serialize::write_message(output, &builder).unwrap();
}
