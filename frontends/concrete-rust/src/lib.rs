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
use std::collections::HashSet;
use std::u128;

use tfhe::core_crypto::commons::math::random::CompressionSeed;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::concrete_protocol_capnp::*;
use capnp::message::{HeapAllocator, ReaderOptions};
use capnp::serialize::{self, OwnedSegments};
use tfhe::core_crypto::prelude::*;
use tfhe::core_crypto::seeders::new_seeder;
use tfhe_csprng::generators::SoftwareRandomGenerator;
use tfhe_csprng::seeders::Seed;

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
fn reader_to_lwe_secret_key(
    reader: &concrete_protocol_capnp::lwe_secret_key::Reader,
) -> LweSecretKey<Vec<u64>> {
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

/// Generates a `SeededLweBootstrapKey` with the specified parameters and random generator.
fn generate_seeded_bsk(
    input_sk: &LweSecretKey<Vec<u64>>,
    output_sk: &LweSecretKey<Vec<u64>>,
    input_lwe_dimension: usize,
    output_polynomial_size: usize,
    output_glwe_dimension: usize,
    decomp_level_count: usize,
    decomp_base_log: usize,
    variance: f64,
) -> SeededLweBootstrapKey<Vec<u64>> {
    let mut boxed_seeder = new_seeder();
    let mut bsk = SeededLweBootstrapKey::new(
        0,
        GlweDimension(output_glwe_dimension).to_glwe_size(),
        PolynomialSize(output_polynomial_size),
        DecompositionBaseLog(decomp_base_log),
        DecompositionLevelCount(decomp_level_count),
        LweDimension(input_lwe_dimension),
        CompressionSeed {
            seed: boxed_seeder.seed(),
        },
        CiphertextModulus::new_native(),
    );
    let output_glwe_sk = GlweSecretKey::from_container(
        output_sk.clone().into_container(),
        PolynomialSize(output_polynomial_size),
    );
    let seeder = boxed_seeder.as_mut();
    par_generate_seeded_lwe_bootstrap_key(
        input_sk,
        &output_glwe_sk,
        &mut bsk,
        Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
        seeder,
    );
    bsk
}

/// Generates an `LweKeyswitchKey` with the specified parameters and random generator.
fn generate_ksk(
    enc_random_generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
    input_sk: &LweSecretKey<Vec<u64>>,
    output_sk: &LweSecretKey<Vec<u64>>,
    input_lwe_dimension: usize,
    output_lwe_dimension: usize,
    decomp_level_count: usize,
    decomp_base_log: usize,
    variance: f64,
) -> LweKeyswitchKey<Vec<u64>> {
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

/// Generates a `SeededLweKeyswitchKey` with the specified parameters and random generator.
fn generate_seeded_ksk(
    input_sk: &LweSecretKey<Vec<u64>>,
    output_sk: &LweSecretKey<Vec<u64>>,
    input_lwe_dimension: usize,
    output_lwe_dimension: usize,
    decomp_level_count: usize,
    decomp_base_log: usize,
    variance: f64,
) -> SeededLweKeyswitchKey<Vec<u64>> {
    let mut boxed_seeder = new_seeder();
    let mut ksk: SeededLweKeyswitchKey<Vec<u64>> = SeededLweKeyswitchKey::new(
        0,
        DecompositionBaseLog(decomp_base_log),
        DecompositionLevelCount(decomp_level_count),
        LweDimension(input_lwe_dimension),
        LweDimension(output_lwe_dimension),
        CompressionSeed {
            seed: boxed_seeder.seed(),
        },
        CiphertextModulus::new_native(),
    );
    let seeder = boxed_seeder.as_mut();
    generate_seeded_lwe_keyswitch_key(
        input_sk,
        output_sk,
        &mut ksk,
        Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
        seeder,
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

/// Build a Cap'n Proto message containing appropriate bootstrapping key.
fn build_bsk(
    bsk_info: concrete_protocol_capnp::lwe_bootstrap_key_info::Reader,
    generated_secret_keys: &std::collections::HashMap<u32, LweSecretKey<Vec<u64>>>,
    mut enc_random_generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
) -> capnp::message::Builder<HeapAllocator> {
    let params = bsk_info.get_params().unwrap();
    let input_lwe_dimension = params.get_input_lwe_dimension() as usize;
    let output_glwe_dimension = params.get_glwe_dimension() as usize;
    let output_polynomial_size = params.get_polynomial_size() as usize;
    let decomp_level_count = params.get_level_count() as usize;
    let decomp_base_log = params.get_base_log() as usize;
    let variance = params.get_variance();
    let compression = bsk_info.get_compression().unwrap();
    let input_sk = generated_secret_keys.get(&bsk_info.get_input_id()).unwrap();
    let output_sk = generated_secret_keys
        .get(&&bsk_info.get_output_id())
        .unwrap();
    match compression {
        Compression::None => {
            let bsk = generate_bsk(
                &mut enc_random_generator,
                input_sk,
                output_sk,
                input_lwe_dimension,
                output_polynomial_size,
                output_glwe_dimension,
                decomp_level_count,
                decomp_base_log,
                variance,
            );
            build_key!(
                bsk,
                bsk_info,
                concrete_protocol_capnp::lwe_bootstrap_key::Builder
            )
        }
        Compression::Seed => {
            let bsk = generate_seeded_bsk(
                input_sk,
                output_sk,
                input_lwe_dimension,
                output_polynomial_size,
                output_glwe_dimension,
                decomp_level_count,
                decomp_base_log,
                variance,
            );
            build_key!(
                bsk,
                bsk_info,
                concrete_protocol_capnp::lwe_bootstrap_key::Builder
            )
        }
        Compression::Paillier => panic!("Paillier compression is not supported"),
    }
}

/// Build a Cap'n Proto message containing appropriate keyswitching key.
fn build_ksk(
    ksk_info: concrete_protocol_capnp::lwe_keyswitch_key_info::Reader,
    generated_secret_keys: &std::collections::HashMap<u32, LweSecretKey<Vec<u64>>>,
    mut enc_random_generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
) -> capnp::message::Builder<HeapAllocator> {
    let params = ksk_info.get_params().unwrap();
    let input_lwe_dimension = params.get_input_lwe_dimension() as usize;
    let output_lwe_dimension = params.get_output_lwe_dimension() as usize;
    let decomp_level_count = params.get_level_count() as usize;
    let decomp_base_log = params.get_base_log() as usize;
    let variance = params.get_variance();
    let compression = ksk_info.get_compression().unwrap();
    let input_sk = generated_secret_keys.get(&ksk_info.get_input_id()).unwrap();
    let output_sk = generated_secret_keys
        .get(&&ksk_info.get_output_id())
        .unwrap();
    match compression {
        Compression::None => {
            let ksk = generate_ksk(
                &mut enc_random_generator,
                input_sk,
                output_sk,
                input_lwe_dimension,
                output_lwe_dimension,
                decomp_level_count,
                decomp_base_log,
                variance,
            );
            build_key!(
                ksk,
                ksk_info,
                concrete_protocol_capnp::lwe_keyswitch_key::Builder
            )
        }
        Compression::Seed => {
            let ksk = generate_seeded_ksk(
                input_sk,
                output_sk,
                input_lwe_dimension,
                output_lwe_dimension,
                decomp_level_count,
                decomp_base_log,
                variance,
            );
            build_key!(
                ksk,
                ksk_info,
                concrete_protocol_capnp::lwe_keyswitch_key::Builder
            )
        }
        Compression::Paillier => panic!("Paillier compression is not supported"),
    }
}

/// Build a Cap'n Proto message containing appropriate packing keyswitch key.
fn build_pksk(
    pksk_info: concrete_protocol_capnp::packing_keyswitch_key_info::Reader,
    generated_secret_keys: &std::collections::HashMap<u32, LweSecretKey<Vec<u64>>>,
    mut enc_random_generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
) -> capnp::message::Builder<HeapAllocator> {
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
    build_key!(
        pksk,
        pksk_info,
        concrete_protocol_capnp::packing_keyswitch_key::Builder
    )
}

/// Generates a Cap'n Proto message containing the keyset based on the provided keyset information and seeds.
///
/// The function generates secret keys, bootstrap keys, keyswitch keys, and packing keyswitch keys based on the keyset information.
/// It uses initial secret keys instead of generating them if they are provided. Bootstrap keys can be ignored in bulk or individually.
///
/// # Arguments
///
/// * `info`: The keyset information.
/// * `secret_seed`: The seed for the secret random generator.
/// * `enc_seed`: The seed for the encryption random generator.
/// * `no_bsk`: A flag indicating to not generate bootstrap keys.
/// * `ignore_bsk`: A list of bootstrap key IDs to ignore.
/// * `init_lwe_secret_keys`: A map of initial secret keys to use instead of generating them.
///
/// # Returns
/// A Cap'n Proto message containing the generated keyset.
fn generate_keyset_message(
    info: concrete_protocol_capnp::keyset_info::Reader,
    secret_seed: u128,
    enc_seed: u128,
    no_bsk: bool,
    ignore_bsk: Vec<u32>,
    init_lwe_secret_keys: &mut std::collections::HashMap<
        u32,
        concrete_protocol_capnp::lwe_secret_key::Reader,
    >,
) -> capnp::message::Builder<HeapAllocator> {
    let mut builder = capnp::message::Builder::new_default();
    let mut keyset_builder = builder.init_root::<concrete_protocol_capnp::keyset::Builder>();

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
            let sk_builder = build_key!(
                sk,
                sk_info,
                concrete_protocol_capnp::lwe_secret_key::Builder
            );
            client_keyset
                .reborrow()
                .get_lwe_secret_keys()
                .unwrap()
                .set_with_caveats(sk_info.get_id(), sk_builder.get_root_as_reader().unwrap())
                .unwrap();
        }
    }

    let mut server_keyset = keyset_builder.init_server();
    // Generate bootstrap keys
    if !no_bsk {
        let ignore_bsk_set: HashSet<&u32> = HashSet::from_iter(ignore_bsk.iter());
        let ignore_count: u32 = if ignore_bsk_set.is_empty() {
            0
        } else {
            // we need to count the keys that are gonna be ignored as the list might have invalid IDs
            let mut ignore_count = 0u32;
            for bsk_info in info.get_lwe_bootstrap_keys().unwrap().iter() {
                if ignore_bsk_set.contains(&bsk_info.get_id()) {
                    ignore_count += 1;
                }
            }
            ignore_count
        };
        server_keyset
            .reborrow()
            .init_lwe_bootstrap_keys(info.get_lwe_bootstrap_keys().unwrap().len() - ignore_count);
        for bsk_info in info.get_lwe_bootstrap_keys().unwrap().iter() {
            if ignore_bsk_set.contains(&bsk_info.get_id()) {
                continue;
            }
            let bsk_builder =
                build_bsk(bsk_info, &generated_secret_keys, &mut enc_random_generator);
            server_keyset
                .reborrow()
                .get_lwe_bootstrap_keys()
                .unwrap()
                .set_with_caveats(bsk_info.get_id(), bsk_builder.get_root_as_reader().unwrap())
                .unwrap();
        }
    }
    // Generate keyswitch keys
    server_keyset
        .reborrow()
        .init_lwe_keyswitch_keys(info.get_lwe_keyswitch_keys().unwrap().len());
    for ksk_info in info.get_lwe_keyswitch_keys().unwrap().iter() {
        let ksk_builder = build_ksk(ksk_info, &generated_secret_keys, &mut enc_random_generator);
        server_keyset
            .reborrow()
            .get_lwe_keyswitch_keys()
            .unwrap()
            .set_with_caveats(ksk_info.get_id(), ksk_builder.get_root_as_reader().unwrap())
            .unwrap();
    }
    // Generate packing keyswitch keys
    server_keyset
        .reborrow()
        .init_packing_keyswitch_keys(info.get_packing_keyswitch_keys().unwrap().len());
    for pksk_info in info.get_packing_keyswitch_keys().unwrap().iter() {
        let pksk_builder = build_pksk(pksk_info, &generated_secret_keys, &mut enc_random_generator);
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
/// It uses initial secret keys instead of generating them if they are provided. Bootstrap keys can be ignored in bulk or individually.
///
/// # Arguments
///
/// * `keyset_info_buffer`: The serialized keyset information.
/// * `secret_seed`: The seed for the secret random generator.
/// * `enc_seed`: The seed for the encryption random generator.
/// * `no_bsk`: A flag indicating to not generate bootstrap keys.
/// * `ignore_bsk`: A list of bootstrap key IDs to ignore.
/// * `init_secret_keys`: A map of initial secret keys to use instead of generating them.
/// # Returns
/// A serialized keyset.
pub fn generate_keyset(
    mut keyset_info_buffer: &[u8],
    secret_seed: u128,
    enc_seed: u128,
    no_bsk: bool,
    ignore_bsk: Vec<u32>,
    init_secret_keys: &std::collections::HashMap<u32, capnp::message::Reader<OwnedSegments>>,
) -> Vec<u8> {
    let reader =
        serialize::read_message_from_flat_slice(&mut keyset_info_buffer, ReaderOptions::new())
            .unwrap();
    let key_set_info = reader
        .get_root::<concrete_protocol_capnp::keyset_info::Reader>()
        .unwrap();
    // keygen
    let mut init_lwe_secret_keys = init_secret_keys
        .iter()
        .map(|(k, v)| {
            (
                *k,
                v.get_root::<concrete_protocol_capnp::lwe_secret_key::Reader>()
                    .unwrap()
                    .clone(),
            )
        })
        .collect::<std::collections::HashMap<u32, concrete_protocol_capnp::lwe_secret_key::Reader>>(
        );
    let builder = generate_keyset_message(
        key_set_info,
        secret_seed as u128,
        enc_seed as u128,
        no_bsk,
        ignore_bsk,
        &mut init_lwe_secret_keys,
    );
    serialize::write_message_to_words(&builder)
}

/// Generate a Concrete keyset based on the provided keyset information and seeds.
///
/// The function generates secret keys, bootstrap keys, keyswitch keys, and packing keyswitch keys based on the keyset information.
///
/// # Arguments
///
/// * `keyset_info_buffer`: The serialized keyset information.
/// * `no_bsk`: A flag indicating to not generate bootstrap keys.
/// * `ignore_bsk`: A list of bootstrap key IDs to ignore.
/// * `secret_seed_lsb`: The seed for the secret random generator (lsb part).
/// * `secret_seed_msb`: The seed for the secret random generator (msb part).
/// * `enc_seed_lsb`: The seed for the encryption random generator (lsb part).
/// * `enc_seed_lsb`: The seed for the encryption random generator (msb part).
/// # Returns
/// A serialized keyset.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn generate_keyset_wasm(
    keyset_info_buffer: &[u8],
    no_bsk: bool,
    ignore_bsk: Vec<u32>,
    secret_seed_lsb: u64,
    secret_seed_msb: u64,
    enc_seed_lsb: u64,
    enc_seed_msb: u64,
) -> Vec<u8> {
    let secret_seed = (secret_seed_msb as u128) << 64 | secret_seed_lsb as u128;
    let enc_seed = (enc_seed_msb as u128) << 64 | enc_seed_lsb as u128;
    generate_keyset(
        keyset_info_buffer,
        secret_seed,
        enc_seed,
        no_bsk,
        ignore_bsk,
        // TODO: support init secret keys
        &mut std::collections::HashMap::new(),
    )
}

/// Reads a secret key from a file and returns the key ID and Cap'n Proto reader.
///
/// # Arguments
///
/// * `path`: The path to the secret key file.
///
/// # Returns
///
/// A tuple containing the key ID and Cap'n Proto reader for the secret key.
pub fn read_secret_key_from_file(path: &str) -> (u32, capnp::message::Reader<OwnedSegments>) {
    let file = std::fs::File::open(path).expect("Failed to open secret key file");
    let reader =
        serialize::read_message(std::io::BufReader::new(file), ReaderOptions::new()).unwrap();
    let key = reader
        .get_root::<concrete_protocol_capnp::lwe_secret_key::Reader>()
        .unwrap();
    let id = key.get_info().unwrap().get_id();
    (id, reader)
}
