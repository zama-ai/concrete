//! This module provides functionality for generating cryptographic keys for the Concrete library.
//! It includes functions for generating secret keys, bootstrap keys, keyswitch keys, and packing keyswitch keys.
//! The keys are generated based on the provided keyset information and seeds for randomness.
//!
//! # Modules
//! - `concrete_protocol_capnp`: Contains the Cap'n Proto schema for the Concrete protocol.
//! - `wasm`: Contains functions for generating keys in a WebAssembly environment.
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
//! - `generate_seeded_bsk`: Generates a `SeededLweBootstrapKey` with the specified parameters and random generator.
//! - `generate_ksk`: Generates an `LweKeyswitchKey` with the specified parameters and random generator.
//! - `generate_seeded_ksk`: Generates a `SeededLweKeyswitchKey` with the specified parameters and random generator.
//! - `generate_pksk`: Generates an `LwePrivateFunctionalPackingKeyswitchKeyList` with the specified parameters and random generator.
//! - `get_lwe_secret_key_from_client_keyset`: Retrieves an LWE secret key from a client keyset based on the provided key ID.
//! - `chunked_bsk_keygen`: Generates an LWE bootstrap key in chunks and sends them to a `MessagePort`.
//! - `generate_keyset`: Generates a Cap'n Proto message containing the keyset based on the provided keyset information and seeds.
//! - `generate_keyset_from_buffers`: Reads keyset information from a file, generates the keyset, and writes it to an output file.
//! - `get_client_keyset`: Extracts the client keyset from a keyset and returns it as a Cap'n Proto message builder.
//! - `get_client_keyset_from_buffers`: Extracts the client keyset from a keyset and returns it as a serialized message.
//! - `explain_keyset_info`: Explains a keyset info buffer in JSON representation.
//! - `add_bsk_keys_to_keyset`: Adds a list of bootstrap keys to an existing keyset.
//! - `add_bsk_keys_to_keyset_from_buffer`: Adds a list of bootstrap keys to an existing keyset buffer.
//!
//! # Macros
//! - `build_key`: A macro for building a Cap'n Proto message for a key with the specified key information and builder type.
use std::collections::HashSet;
use std::u128;

use crate::concrete_protocol_capnp::*;
use capnp::message::HeapAllocator;
use capnp::serialize;
use tfhe::core_crypto::commons::math::random::CompressionSeed;
use tfhe::core_crypto::prelude::*;
use tfhe::core_crypto::seeders::new_seeder;
use tfhe_csprng::generators::SoftwareRandomGenerator;
use tfhe_csprng::seeders::Seed;

#[cfg(feature = "wasm")]
use wasm_bindgen::UnwrapThrowExt;

// We want to have meaningful error messages when using wasm. This can be done by using expect_throw,
// which is only available during wasm build.
#[allow(dead_code)]
trait UnwrapOrThrowExt<T>: Sized {
    fn expect_or_throw(self, message: &str) -> T;
}

impl<T, E> UnwrapOrThrowExt<T> for Result<T, E>
where
    E: core::fmt::Debug,
{
    #[cfg(feature = "wasm")]
    fn expect_or_throw(self, message: &str) -> T {
        self.expect_throw(message)
    }

    #[cfg(not(feature = "wasm"))]
    fn expect_or_throw(self, message: &str) -> T {
        self.expect(message)
    }
}

impl<T> UnwrapOrThrowExt<T> for Option<T> {
    #[cfg(feature = "wasm")]
    fn expect_or_throw(self, message: &str) -> T {
        self.expect_throw(message)
    }

    #[cfg(not(feature = "wasm"))]
    fn expect_or_throw(self, message: &str) -> T {
        self.expect(message)
    }
}

pub mod concrete_protocol_capnp {
    use capnp::serialize;
    use capnp::traits::FromPointerReader;

    include!(concat!(
        env!("OUT_DIR"),
        "/capnp/concrete_protocol_capnp.rs"
    ));

    /// Reads a secret key from a buffer and returns the key ID and Cap'n Proto message.
    ///
    /// # Arguments
    ///
    /// * `buffer`: The buffer containing the secret key.
    ///
    /// # Returns
    ///
    /// A tuple containing the key ID and Cap'n Proto reader for the secret key.
    pub fn read_secret_key_from_buffer(
        buffer: &[u8],
    ) -> capnp::Result<(u32, capnp::message::Reader<serialize::OwnedSegments>)> {
        let message = read_capnp_from_buffer(buffer)?;
        let key = get_reader_from_message::<lwe_secret_key::Reader>(&message)?;
        let id = key.get_info()?.get_id();
        Ok((id, message))
    }

    pub fn get_reader_from_message<'a, T>(
        message: &'a capnp::message::Reader<serialize::OwnedSegments>,
    ) -> capnp::Result<T>
    where
        T: FromPointerReader<'a>,
    {
        message.get_root::<'a, T>()
    }

    pub fn get_reader_from_builder<'a, T>(
        message: &'a mut capnp::message::Builder<capnp::message::HeapAllocator>,
    ) -> capnp::Result<T>
    where
        T: FromPointerReader<'a>,
    {
        message.get_root_as_reader::<'a, T>()
    }

    /// Reads a Cap'n Proto from a buffer and returns a message reader.
    ///
    /// # Arguments
    ///
    /// * `buffer`: The buffer containing the Cap'n Proto.
    ///
    /// # Returns
    ///
    /// A Cap'n Proto message reader.
    pub fn read_capnp_from_buffer(
        mut buffer: &[u8],
    ) -> capnp::Result<capnp::message::Reader<serialize::OwnedSegments>> {
        let mut reader_options = capnp::message::ReaderOptions::new();
        reader_options.traversal_limit_in_words(Some(buffer.len()));
        serialize::read_message(&mut buffer, reader_options)
    }
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
        output.push(u64::from_le_bytes(
            i.try_into().expect_or_throw("Failed to convert to u64"),
        ));
    }
    output
}

/// Converts a Cap'n Proto reader to an `LweSecretKey`.
fn reader_to_lwe_secret_key(
    reader: &concrete_protocol_capnp::lwe_secret_key::Reader,
) -> LweSecretKey<Vec<u64>> {
    let payload = reader
        .get_payload()
        .expect_or_throw("Failed to get payload");
    let mut sk_data = Vec::new();
    let data_list = payload.get_data().expect_or_throw("Failed to get data");
    for data in data_list.iter() {
        sk_data.extend_from_slice(&u8_slice_to_u64_vector(
            data.expect_or_throw("Failed to get data"),
        ));
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
        key_root_builder
            .set_info($key_info)
            .expect_or_throw("Failed to set key info");
        key_root_builder
            .set_payload(
                vector_to_payload(key_data.as_slice())
                    .get_root_as_reader()
                    .expect_or_throw("Failed to get payload"),
            )
            .expect_or_throw("Failed to set payload");
        key_builder
    }};
}

/// Build a Cap'n Proto message containing appropriate bootstrapping key.
fn build_bsk(
    bsk_info: concrete_protocol_capnp::lwe_bootstrap_key_info::Reader,
    generated_secret_keys: &std::collections::HashMap<u32, LweSecretKey<Vec<u64>>>,
    mut enc_random_generator: &mut EncryptionRandomGenerator<SoftwareRandomGenerator>,
) -> capnp::message::Builder<HeapAllocator> {
    let params = bsk_info
        .get_params()
        .expect_or_throw("Failed to get params");
    let input_lwe_dimension = params.get_input_lwe_dimension() as usize;
    let output_glwe_dimension = params.get_glwe_dimension() as usize;
    let output_polynomial_size = params.get_polynomial_size() as usize;
    let decomp_level_count = params.get_level_count() as usize;
    let decomp_base_log = params.get_base_log() as usize;
    let variance = params.get_variance();
    let compression = bsk_info
        .get_compression()
        .expect_or_throw("Failed to get compression for bootstrap key");
    let input_sk = generated_secret_keys
        .get(&bsk_info.get_input_id())
        .expect_or_throw("Failed to find input secret key for bootstrap key");
    let output_sk = generated_secret_keys
        .get(&bsk_info.get_output_id())
        .expect_or_throw("Failed to find output secret key for bootstrap key");
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
    let params = ksk_info
        .get_params()
        .expect_or_throw("Failed to get keyswitch key parameters");
    let input_lwe_dimension = params.get_input_lwe_dimension() as usize;
    let output_lwe_dimension = params.get_output_lwe_dimension() as usize;
    let decomp_level_count = params.get_level_count() as usize;
    let decomp_base_log = params.get_base_log() as usize;
    let variance = params.get_variance();
    let compression = ksk_info
        .get_compression()
        .expect_or_throw("Failed to get compression for keyswitch key");
    let input_sk = generated_secret_keys
        .get(&ksk_info.get_input_id())
        .expect_or_throw("Failed to find input secret key for keyswitch key");
    let output_sk = generated_secret_keys
        .get(&ksk_info.get_output_id())
        .expect_or_throw("Failed to find output secret key for keyswitch key");
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
    let params = pksk_info
        .get_params()
        .expect_or_throw("Failed to get packing keyswitch key parameters");
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
            .expect_or_throw("Failed to find input secret key for packing keyswitch key"),
        generated_secret_keys
            .get(&pksk_info.get_output_id())
            .expect_or_throw("Failed to find output secret key for packing keyswitch key"),
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

/// Retrieves an LWE secret key from a client keyset based on the provided key ID.
///
/// This function searches for the LWE secret key with the specified ID in the given client keyset
/// and returns a Cap'n Proto message builder containing the secret key.
///
/// # Arguments
///
/// * `client_keyset` - A reference to the Cap'n Proto client keyset reader.
/// * `id` - The ID of the LWE secret key to retrieve.
///
/// # Returns
///
/// A Cap'n Proto message builder containing the LWE secret key.
///
/// # Panics
///
/// This function will panic if the LWE secret keys cannot be retrieved, or if the secret key
/// with the specified ID is not found in the keyset.
fn get_lwe_secret_key_from_client_keyset(
    client_keyset: &concrete_protocol_capnp::client_keyset::Reader,
    id: u32,
) -> capnp::message::Builder<HeapAllocator> {
    let sk = client_keyset
        .get_lwe_secret_keys()
        .expect_or_throw("Failed to get LWE secret keys")
        .iter()
        .find(|sk| {
            sk.get_info()
                .expect_or_throw("Failed to get secret key info")
                .get_id()
                == id
        })
        .expect_or_throw(format!("Secret key (id:{}) not found in keyset", id).as_str());
    let mut builder = capnp::message::Builder::new_default();
    let mut sk_builder = builder.init_root::<concrete_protocol_capnp::lwe_secret_key::Builder>();
    sk_builder
        .set_info(
            sk.get_info()
                .expect_or_throw("Failed to get secret key info"),
        )
        .expect_or_throw("Failed to set secret key info");
    sk_builder
        .set_payload(
            sk.get_payload()
                .expect_or_throw("Failed to get secret key payload"),
        )
        .expect_or_throw("Failed to set secret key payload");
    builder
}

pub fn get_lwe_secret_key_from_client_keyset_from_buffers(
    client_keyset_buffer: &[u8],
    id: u32,
) -> Vec<u8> {
    let reader = read_capnp_from_buffer(client_keyset_buffer)
        .expect_or_throw("Failed to read client keyset");
    let client_keyset =
        get_reader_from_message(&reader).expect_or_throw("Failed to get client keyset");
    let sk_builder = get_lwe_secret_key_from_client_keyset(&client_keyset, id);
    serialize::write_message_to_words(&sk_builder)
}

/// Builds a Cap'n Proto message containing a bootstrap key.
///
/// # Arguments
///
/// * `keyset_info`: The keyset information. This is used to get info about the bootstrap key.
/// * `bsk_id`: The ID of the bootstrap key.
/// * `bsk`: The bootstrap key to be included in the message.
///
/// # Returns
///
/// A Cap'n Proto message builder containing the bootstrap key.
pub fn build_bsk_proto(
    keyset_info: concrete_protocol_capnp::keyset_info::Reader,
    bsk_id: u32,
    bsk: &LweBootstrapKey<Vec<u64>>,
) -> capnp::message::Builder<HeapAllocator> {
    let mut builder = capnp::message::Builder::new_default();
    let mut bsk_builder =
        builder.init_root::<concrete_protocol_capnp::lwe_bootstrap_key::Builder>();
    bsk_builder
        .set_info(
            keyset_info
                .get_lwe_bootstrap_keys()
                .expect_or_throw("Failed to get LWE bootstrap key info")
                .get(bsk_id),
        )
        .expect_or_throw("Failed to set bootstrap key info");
    bsk_builder
        .set_payload(
            vector_to_payload(bsk.as_view().into_container())
                .get_root_as_reader()
                .expect_or_throw("Failed to get payload"),
        )
        .expect_or_throw("Failed to set bootstrap key payload");
    builder
}

/// Adds a list of bootstrap keys to an existing keyset.
///
/// This function takes an existing keyset, a list of bootstrap keys, and constructs a new keyset
/// by adding the bootstrap keys to the existing keyset.
///
/// # Arguments
///
/// * `keyset`: The existing keyset to which the bootstrap keys will be added.
/// * `bsks`: A vector of bootstrap keys to be added.
///
/// # Returns
///
/// A new keyset with the additional bootstrap keys.
pub fn add_bsk_keys_to_keyset(
    keyset: concrete_protocol_capnp::keyset::Reader,
    bsks: Vec<concrete_protocol_capnp::lwe_bootstrap_key::Reader>,
) -> capnp::message::Builder<HeapAllocator> {
    let mut builder = capnp::message::Builder::new_default();
    let mut new_keyset = builder.init_root::<concrete_protocol_capnp::keyset::Builder>();

    // Copy existing client and server keysets
    new_keyset
        .reborrow()
        .set_client(
            keyset
                .get_client()
                .expect_or_throw("Failed to get client keyset"),
        )
        .expect_or_throw("Failed to set client keyset");
    let mut server_keyset = new_keyset.init_server();
    server_keyset
        .reborrow()
        .set_lwe_keyswitch_keys(
            keyset
                .get_server()
                .expect_or_throw("Failed to get server keyset")
                .get_lwe_keyswitch_keys()
                .expect_or_throw("Failed to get LWE keyswitch keys"),
        )
        .expect_or_throw("Failed to set LWE keyswitch keys");
    server_keyset
        .reborrow()
        .set_packing_keyswitch_keys(
            keyset
                .get_server()
                .expect_or_throw("Failed to get server keyset")
                .get_packing_keyswitch_keys()
                .expect_or_throw("Failed to get packing keyswitch keys"),
        )
        .expect_or_throw("Failed to set packing keyswitch keys");

    // Initialize bootstrap keys with existing ones
    let existing_bsk_keys = keyset
        .get_server()
        .expect_or_throw("Failed to get server keyset")
        .get_lwe_bootstrap_keys()
        .expect_or_throw("Failed to get LWE bootstrap keys");
    let mut new_bsk_keys =
        server_keyset.init_lwe_bootstrap_keys(existing_bsk_keys.len() + bsks.len() as u32);
    for bsk in existing_bsk_keys.iter() {
        new_bsk_keys
            .set_with_caveats(
                bsk.get_info()
                    .expect_or_throw("Failed to get bootstrap key info")
                    .get_id(),
                bsk,
            )
            .expect_or_throw("Failed to set existing bootstrap key");
    }
    // Add new bootstrap keys
    for bsk in bsks.iter() {
        new_bsk_keys
            .set_with_caveats(
                bsk.get_info()
                    .expect_or_throw("Failed to get bootstrap key info")
                    .get_id(),
                *bsk,
            )
            .expect_or_throw("Failed to set new bootstrap key");
    }

    builder
}

/// Adds a list of bootstrap keys to an existing keyset using buffers.
///
/// This function takes an existing keyset buffer, a list of bootstrap key buffers, and constructs a new keyset
/// by adding the bootstrap keys to the existing keyset.
///
/// # Arguments
///
/// * `keyset_buffer`: A byte slice containing the serialized existing keyset (Capnp).
/// * `bsk_buffers`: A vector of byte slices, each containing a serialized bootstrap key (Capnp).
///
/// # Returns
///
/// A `Vec<u8>` containing the serialized new keyset.
pub fn add_bsk_keys_to_keyset_from_buffers(
    keyset_buffer: &[u8],
    bsk_buffers: Vec<&[u8]>,
) -> Vec<u8> {
    // Deserialize the keyset buffer
    let keyset_message =
        read_capnp_from_buffer(keyset_buffer).expect_or_throw("Failed to read keyset buffer");
    let keyset_reader =
        get_reader_from_message(&keyset_message).expect_or_throw("Failed to read keyset reader");

    // Deserialize each bootstrap key buffer
    let bsks_readers = bsk_buffers
        .iter()
        .map(|bsk_buffer| {
            read_capnp_from_buffer(bsk_buffer).expect_or_throw("Failed to read bootstrap key")
        })
        .collect::<Vec<_>>();
    let bsks = bsks_readers
        .iter()
        .map(|bsk_reader| {
            get_reader_from_message(bsk_reader)
                .expect_or_throw("Failed to get bootstrap key reader")
        })
        .collect::<Vec<_>>();

    // Call the original function
    let builder = add_bsk_keys_to_keyset(keyset_reader, bsks);

    // Serialize the result to a buffer
    serialize::write_message_to_words(&builder)
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
/// * `no_ksk`: A flag indicating to not generate keyswitch keys.
/// * `ignore_ksk`: A list of keyswitch key IDs to ignore.
/// * `init_lwe_secret_keys`: A map of initial secret keys to use instead of generating them.
///
/// # Returns
/// A Cap'n Proto message containing the generated keyset.
fn generate_keyset(
    info: concrete_protocol_capnp::keyset_info::Reader,
    secret_seed: u128,
    enc_seed: u128,
    no_bsk: bool,
    ignore_bsk: Vec<u32>,
    no_ksk: bool,
    ignore_ksk: Vec<u32>,
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
    client_keyset.reborrow().init_lwe_secret_keys(
        info.get_lwe_secret_keys()
            .expect_or_throw("Failed to get LWE secret keys")
            .len(),
    );
    // Generate secret keys
    let mut generated_secret_keys: std::collections::HashMap<u32, LweSecretKey<Vec<u64>>> =
        std::collections::HashMap::new();
    for sk_info in info
        .get_lwe_secret_keys()
        .expect_or_throw("Failed to get LWE secret keys")
        .iter()
    {
        if init_lwe_secret_keys.contains_key(&sk_info.get_id()) {
            // we initialize the secret key with the provided secret key
            let sk_reader = init_lwe_secret_keys
                .remove(&sk_info.get_id())
                .expect_or_throw("Failed to get initial secret key");
            generated_secret_keys.insert(sk_info.get_id(), reader_to_lwe_secret_key(&sk_reader));
            client_keyset
                .reborrow()
                .get_lwe_secret_keys()
                .expect_or_throw("Failed to get LWE secret keys")
                .set_with_caveats(sk_info.get_id(), sk_reader)
                .expect_or_throw("Failed to set LWE secret key");
        } else {
            // we generate a new secret key
            let lwe_dim = sk_info
                .get_params()
                .expect_or_throw("Failed to get secret key parameters")
                .get_lwe_dimension() as usize;
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
                .expect_or_throw("Failed to get LWE secret keys")
                .set_with_caveats(
                    sk_info.get_id(),
                    sk_builder
                        .get_root_as_reader()
                        .expect_or_throw("Failed to get root as reader"),
                )
                .expect_or_throw("Failed to set LWE secret key");
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
            for bsk_info in info
                .get_lwe_bootstrap_keys()
                .expect_or_throw("Failed to get LWE bootstrap keys")
                .iter()
            {
                if ignore_bsk_set.contains(&bsk_info.get_id()) {
                    ignore_count += 1;
                }
            }
            ignore_count
        };
        server_keyset.reborrow().init_lwe_bootstrap_keys(
            info.get_lwe_bootstrap_keys()
                .expect_or_throw("Failed to get LWE bootstrap keys")
                .len()
                - ignore_count,
        );
        for bsk_info in info
            .get_lwe_bootstrap_keys()
            .expect_or_throw("Failed to get LWE bootstrap keys")
            .iter()
        {
            if ignore_bsk_set.contains(&bsk_info.get_id()) {
                continue;
            }
            let bsk_builder =
                build_bsk(bsk_info, &generated_secret_keys, &mut enc_random_generator);
            server_keyset
                .reborrow()
                .get_lwe_bootstrap_keys()
                .expect_or_throw("Failed to get LWE bootstrap keys")
                .set_with_caveats(
                    bsk_info.get_id(),
                    bsk_builder
                        .get_root_as_reader()
                        .expect_or_throw("Failed to get root as reader"),
                )
                .expect_or_throw("Failed to set LWE bootstrap key");
        }
    }
    // Generate keyswitch keys
    if !no_ksk {
        let ignore_ksk_set: HashSet<&u32> = HashSet::from_iter(ignore_ksk.iter());
        let ignore_count: u32 = if ignore_ksk_set.is_empty() {
            0
        } else {
            // we need to count the keys that are gonna be ignored as the list might have invalid IDs
            let mut ignore_count = 0u32;
            for ksk_info in info
                .get_lwe_keyswitch_keys()
                .expect_or_throw("Failed to get LWE keyswitch keys")
                .iter()
            {
                if ignore_ksk_set.contains(&ksk_info.get_id()) {
                    ignore_count += 1;
                }
            }
            ignore_count
        };
        server_keyset.reborrow().init_lwe_keyswitch_keys(
            info.get_lwe_keyswitch_keys()
                .expect_or_throw("Failed to get LWE keyswitch keys")
                .len()
                - ignore_count,
        );
        for ksk_info in info
            .get_lwe_keyswitch_keys()
            .expect_or_throw("Failed to get LWE keyswitch keys")
            .iter()
        {
            if ignore_ksk_set.contains(&ksk_info.get_id()) {
                continue;
            }
            let ksk_builder =
                build_ksk(ksk_info, &generated_secret_keys, &mut enc_random_generator);
            server_keyset
                .reborrow()
                .get_lwe_keyswitch_keys()
                .expect_or_throw("Failed to get LWE keyswitch keys")
                .set_with_caveats(
                    ksk_info.get_id(),
                    ksk_builder
                        .get_root_as_reader()
                        .expect_or_throw("Failed to get root as reader"),
                )
                .expect_or_throw("Failed to set LWE keyswitch key");
        }
    }
    // Generate packing keyswitch keys
    server_keyset.reborrow().init_packing_keyswitch_keys(
        info.get_packing_keyswitch_keys()
            .expect_or_throw("Failed to get packing keyswitch keys")
            .len(),
    );
    for pksk_info in info
        .get_packing_keyswitch_keys()
        .expect_or_throw("Failed to get packing keyswitch keys")
        .iter()
    {
        let pksk_builder = build_pksk(pksk_info, &generated_secret_keys, &mut enc_random_generator);
        server_keyset
            .reborrow()
            .get_packing_keyswitch_keys()
            .expect_or_throw("Failed to get packing keyswitch keys")
            .set_with_caveats(
                pksk_info.get_id(),
                pksk_builder
                    .get_root_as_reader()
                    .expect_or_throw("Failed to get root as reader"),
            )
            .expect_or_throw("Failed to set packing keyswitch key");
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
/// * `no_ksk`: A flag indicating to not generate keyswitch keys.
/// * `ignore_ksk`: A list of keyswitch key IDs to ignore.
/// * `init_secret_keys`: A map of initial secret keys to use instead of generating them.
/// # Returns
/// A serialized keyset.
pub fn generate_keyset_from_buffers(
    keyset_info_buffer: &[u8],
    secret_seed: u128,
    enc_seed: u128,
    no_bsk: bool,
    ignore_bsk: Vec<u32>,
    no_ksk: bool,
    ignore_ksk: Vec<u32>,
    initial_secret_key_buffers: &Vec<&[u8]>,
) -> Vec<u8> {
    let keyset_info_message =
        read_capnp_from_buffer(keyset_info_buffer).expect_or_throw("Failed to read keyset info");
    let keyset_info_reader =
        get_reader_from_message(&keyset_info_message).expect_or_throw("Failed to get keyset info");
    let initial_secret_keys_owned: std::collections::HashMap<
        u32,
        capnp::message::Reader<capnp::serialize::OwnedSegments>,
    > = initial_secret_key_buffers
        .iter()
        .map(|buffer| {
            let (id, reader) =
                read_secret_key_from_buffer(buffer).expect_or_throw("Failed to read secret key");
            (id, reader)
        })
        .collect();
    let mut init_lwe_secret_keys: std::collections::HashMap<
        u32,
        concrete_protocol_capnp::lwe_secret_key::Reader,
    > = initial_secret_keys_owned
        .iter()
        .map(|(k, v)| {
            (
                *k,
                get_reader_from_message(&v).expect_or_throw("Failed to get secret key reader"),
            )
        })
        .collect();
    // keygen
    let builder = generate_keyset(
        keyset_info_reader,
        secret_seed as u128,
        enc_seed as u128,
        no_bsk,
        ignore_bsk,
        no_ksk,
        ignore_ksk,
        &mut init_lwe_secret_keys,
    );
    serialize::write_message_to_words(&builder)
}

/// Get the client keyset from a keyset.
///
/// This function extracts the client keyset from a keyset and returns it as a Cap'n Proto message builder.
///
/// # Arguments
///
/// * `keyset`: The keyset to extract the client keyset from.
///
/// # Returns
///
/// A Cap'n Proto message builder containing the client keyset.
fn get_client_keyset(
    keyset: concrete_protocol_capnp::keyset::Reader,
) -> capnp::message::Builder<HeapAllocator> {
    let mut builder = capnp::message::Builder::new_default();
    let mut client_keyset = builder.init_root::<concrete_protocol_capnp::client_keyset::Builder>();

    client_keyset.reborrow().init_lwe_secret_keys(
        keyset
            .get_client()
            .expect_or_throw("Failed to get client keyset")
            .get_lwe_secret_keys()
            .expect_or_throw("Failed to get LWE secret keys from client keyset")
            .len(),
    );
    // Copy secret keys
    for sk in keyset
        .get_client()
        .expect_or_throw("Failed to get client keyset")
        .get_lwe_secret_keys()
        .expect_or_throw("Failed to get LWE secret keys from client keyset")
        .iter()
    {
        client_keyset
            .reborrow()
            .get_lwe_secret_keys()
            .expect_or_throw("Failed to get LWE secret keys builder")
            .set_with_caveats(
                sk.get_info()
                    .expect_or_throw("Failed to get secret key info")
                    .get_id(),
                sk,
            )
            .expect_or_throw("Failed to set LWE secret key");
    }
    builder
}

/// Get the client keyset from a keyset.
///
/// # Arguments
/// * `keyset`: The keyset to extract the client keyset from.
///
/// # Returns
/// A serialized client keyset.
pub fn get_client_keyset_from_buffers(keyset_buffer: &[u8]) -> Vec<u8> {
    let keyset_message =
        read_capnp_from_buffer(keyset_buffer).expect_or_throw("Failed to read keyset");
    let keyset_reader =
        get_reader_from_message(&keyset_message).expect_or_throw("Failed to get keyset reader");
    let builder = get_client_keyset(keyset_reader);
    serialize::write_message_to_words(&builder)
}

#[cfg(feature = "wasm")]
pub mod wasm {
    /// This module provides WebAssembly (WASM) bindings for cryptographic key generation and manipulation
    /// in the Concrete library. It includes functions for generating LWE bootstrap keys, retrieving LWE
    /// secret keys, generating keysets, and explaining keyset information in JSON format.
    ///
    /// The functions in this module are designed to be used in a WebAssembly environment and leverage
    /// the `wasm-bindgen` crate for interoperability with JavaScript.
    ///
    /// # Features
    ///
    /// - **Chunked LWE Bootstrap Key Generation**: Generate LWE bootstrap keys in chunks and send them
    ///   to a `MessagePort`.
    /// - **Keyset Manipulation**: Generate keysets, client keysets, and retrieve specific keys from
    ///   keysets.
    /// - **JSON Representation**: Explain keyset information in a JSON format for easier debugging and
    ///   analysis.
    ///
    /// # Usage
    ///
    /// To use this module, ensure that the `wasm` feature is enabled in your build configuration.
    use crate::*;
    use serde_json;
    use tfhe::safe_serialization::{safe_serialize, safe_serialized_size};
    use wasm_bindgen::prelude::*;

    /// Generates an LWE bootstrap key in chunks.
    ///
    /// This function reads the keyset information, input secret key, and output secret key from byte buffers,
    /// generates the LWE bootstrap key in chunks, and sends each chunk to a specified `MessagePort`.
    ///
    /// # Arguments
    ///
    /// * `keyset_info_buffer` - A byte slice containing the serialized keyset information (Capnp).
    /// * `input_secret_key_buffer` - A byte slice containing the serialized input secret key (Capnp).
    /// * `output_secret_key_buffer` - A byte slice containing the serialized output secret key (Capnp).
    /// * `bsk_id` - The ID of the bootstrap key to generate.
    /// * `enc_seed` - The seed for the encryption random generator (used if no compression).
    /// * `compression_seed` - The seed for the compression random generator (used if seeded key).
    /// * `chunk_size` - The size of each chunk to generate.
    /// * `port` - The `MessagePort` to send the generated chunks to.
    ///
    /// # Returns
    ///
    /// The total length of the generated bootstrap key in bytes.
    ///
    /// # Panics
    ///
    /// This function will throw an exception if it fails to read the keyset info buffer, input secret key buffer,
    /// or output secret key buffer, or if it fails to post a message to the specified `MessagePort`.
    #[wasm_bindgen]
    pub async fn chunked_bsk_keygen(
        keyset_info_buffer: &[u8],
        input_secret_key_buffer: &[u8],
        output_secret_key_buffer: &[u8],
        bsk_id: u32,
        enc_seed: u128,
        compression_seed: u128,
        chunk_size: usize,
        port: web_sys::MessagePort,
    ) -> usize {
        // deserialize inputs
        let message = read_capnp_from_buffer(keyset_info_buffer)
            .expect_throw("Failed to read keyset info buffer");
        let key_set_info =
            get_reader_from_message::<concrete_protocol_capnp::keyset_info::Reader>(&message)
                .expect_throw("Failed to get root keyset info reader");
        let message = read_capnp_from_buffer(input_secret_key_buffer)
            .expect_throw("Failed to read input secret key buffer");
        let input_sk_reader = get_reader_from_message(&message)
            .expect_throw("Failed to get root input secret key reader");
        let message = read_capnp_from_buffer(output_secret_key_buffer)
            .expect_throw("Failed to read output secret key buffer");
        let output_sk_reader = get_reader_from_message(&message)
            .expect_throw("Failed to get root output secret key reader");

        // Parameters
        let bsk_info = key_set_info
            .get_lwe_bootstrap_keys()
            .expect_throw("Failed to get LWE bootstrap keys")
            .get(bsk_id);
        let params = bsk_info
            .get_params()
            .expect_throw("Failed to get bootstrap key parameters");
        let compression = bsk_info
            .get_compression()
            .expect_throw("Failed to get compression");

        let input_lwe_dimension = LweDimension(params.get_input_lwe_dimension() as usize);
        let decomp_base_log = DecompositionBaseLog(params.get_base_log() as usize);
        let decomp_level_count = DecompositionLevelCount(params.get_level_count() as usize);
        let glwe_dimension = GlweDimension(params.get_glwe_dimension() as usize);
        let polynomial_size = PolynomialSize(params.get_polynomial_size() as usize);
        let glwe_noise_distribution = Gaussian::from_dispersion_parameter(
            Variance::from_variance(params.get_variance()).get_standard_dev(),
            0.0,
        );
        let ciphertext_modulus: CiphertextModulus<u64> = CiphertextModulus::new_native();

        // Input and output secret keys
        let input_lwe_secret_key = reader_to_lwe_secret_key(&input_sk_reader);
        let output_lwe_secret_key = reader_to_lwe_secret_key(&output_sk_reader);
        let output_glwe_secret_key =
            GlweSecretKey::from_container(output_lwe_secret_key.into_container(), polynomial_size);

        let mut total_len = 0;
        let mut seeder = new_seeder();
        let seeder = seeder.as_mut();
        match compression {
            Compression::None => {
                let mut encryption_generator =
                    EncryptionRandomGenerator::<DefaultRandomGenerator>::new(
                        Seed(enc_seed),
                        seeder,
                    );
                let chunk_generator = LweBootstrapKeyChunkGenerator::new(
                    &mut encryption_generator,
                    ChunkSize(chunk_size),
                    input_lwe_dimension,
                    glwe_dimension.to_glwe_size(),
                    polynomial_size,
                    decomp_base_log,
                    decomp_level_count,
                    ciphertext_modulus,
                    &input_lwe_secret_key,
                    &output_glwe_secret_key,
                    glwe_noise_distribution,
                    false,
                );
                for chunk in chunk_generator {
                    let mut serialized_data = Vec::new();
                    let serialized_size = safe_serialized_size(&chunk)
                        .expect_throw("couldn't guess size of serialized chunk");
                    safe_serialize(&chunk, &mut serialized_data, serialized_size)
                        .expect_throw("couldn't serialize chunk");
                    total_len += serialized_data.len();
                    let js_array = web_sys::js_sys::Uint8Array::from(serialized_data.as_slice());
                    port.post_message(&js_array)
                        .expect_throw("Failed to post message to port");
                }
            }
            Compression::Seed => {
                let chunk_generator = SeededLweBootstrapKeyChunkGenerator::new(
                    ChunkSize(chunk_size),
                    input_lwe_dimension,
                    glwe_dimension.to_glwe_size(),
                    polynomial_size,
                    decomp_base_log,
                    decomp_level_count,
                    ciphertext_modulus,
                    &input_lwe_secret_key,
                    &output_glwe_secret_key,
                    glwe_noise_distribution,
                    CompressionSeed {
                        seed: Seed(compression_seed),
                    },
                    seeder,
                    false,
                );
                for chunk in chunk_generator {
                    let mut serialized_data = Vec::new();
                    let serialized_size = safe_serialized_size(&chunk)
                        .expect_throw("couldn't guess size of serialized chunk");
                    safe_serialize(&chunk, &mut serialized_data, serialized_size)
                        .expect_throw("couldn't serialize chunk");
                    total_len += serialized_data.len();
                    let js_array = web_sys::js_sys::Uint8Array::from(serialized_data.as_slice());
                    port.post_message(&js_array)
                        .expect_throw("Failed to post message to port");
                }
            }
            Compression::Paillier => todo!("Paillier compression not implemented"),
        }
        port.close();
        total_len
    }

    /// Retrieves an LWE secret key from a client keyset based on the provided key ID.
    ///
    /// This function is intended to be used in a WebAssembly (WASM) environment.
    /// It reads the client keyset from a byte buffer (Capnp), finds the LWE secret key
    /// with the specified ID, and returns the serialized key as a byte vector (Capnp).
    ///
    /// # Arguments
    ///
    /// * `client_keyset_buffer` - A byte slice containing the serialized client keyset (Capnp).
    /// * `id` - The ID of the LWE secret key to retrieve.
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` containing the serialized LWE secret key (Capnp).
    ///
    /// # Panics
    ///
    /// This function will throw an exception if it fails to read the client keyset buffer, or retrieve
    /// the secret key.
    #[wasm_bindgen]
    pub fn get_lwe_secret_key_from_client_keyset(
        mut client_keyset_buffer: &[u8],
        id: u32,
    ) -> Vec<u8> {
        get_lwe_secret_key_from_client_keyset_from_buffers(&mut client_keyset_buffer, id)
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
    /// * `no_ksk`: A flag indicating to not generate keyswitch keys.
    /// * `ignore_ksk`: A list of keyswitch key IDs to ignore.
    /// * `secret_seed`: The seed for the secret random generator.
    /// * `enc_seed`: The seed for the encryption random generator.
    /// * `client_keyset_buffer`: The serialized client keyset information (Optional: empty buffer means not provided).
    /// # Returns
    /// A serialized keyset.
    #[wasm_bindgen]
    pub fn generate_keyset(
        keyset_info_buffer: &[u8],
        no_bsk: bool,
        ignore_bsk: Vec<u32>,
        no_ksk: bool,
        ignore_ksk: Vec<u32>,
        secret_seed: u128,
        enc_seed: u128,
        client_keyset_buffer: &[u8],
    ) -> Vec<u8> {
        let mut init_secret_keys = std::collections::HashMap::new();
        let client_keyset_message;
        if !client_keyset_buffer.is_empty() {
            client_keyset_message = read_capnp_from_buffer(client_keyset_buffer)
                .expect_throw("Failed to read client keyset buffer");
            let client_keyset = get_reader_from_message::<
                concrete_protocol_capnp::client_keyset::Reader,
            >(&client_keyset_message)
            .expect_throw("Failed to get root client keyset reader");

            for sk in client_keyset
                .get_lwe_secret_keys()
                .expect_throw("Failed to get LWE secret keys")
                .iter()
            {
                let id = sk
                    .get_info()
                    .expect_throw("Failed to get secret key info")
                    .get_id();
                init_secret_keys.insert(id, sk);
            }
        }

        let keyset_info_message = read_capnp_from_buffer(keyset_info_buffer)
            .expect_throw("Failed to read keyset info buffer");
        let key_set_info =
            get_reader_from_message(&keyset_info_message).expect_throw("Failed to get keyset info");

        let builder = crate::generate_keyset(
            key_set_info,
            secret_seed,
            enc_seed,
            no_bsk,
            ignore_bsk,
            no_ksk,
            ignore_ksk,
            &mut init_secret_keys,
        );
        serialize::write_message_to_words(&builder)
    }

    /// Generate a Concrete client keyset based on the provided keyset information and seeds.
    ///
    /// The function generates secret keys based on the keyset information.
    ///
    /// # Arguments
    ///
    /// * `keyset_info_buffer`: The serialized keyset information.
    /// * `secret_seed`: The seed for the secret random generator.
    ///
    /// # Returns
    /// A serialized client keyset.
    #[wasm_bindgen]
    pub fn generate_client_keyset(keyset_info_buffer: &[u8], secret_seed: u128) -> Vec<u8> {
        let message = read_capnp_from_buffer(keyset_info_buffer)
            .expect_throw("Failed to read keyset info buffer");
        let key_set_info =
            get_reader_from_message::<concrete_protocol_capnp::keyset_info::Reader>(&message)
                .expect_throw("Failed to get root keyset info reader");

        let mut builder = capnp::message::Builder::new_default();
        let mut client_keyset =
            builder.init_root::<concrete_protocol_capnp::client_keyset::Builder>();

        let mut secret_random_generator: SecretRandomGenerator<SoftwareRandomGenerator> =
            SecretRandomGenerator::new(Seed(secret_seed));

        client_keyset.reborrow().init_lwe_secret_keys(
            key_set_info
                .get_lwe_secret_keys()
                .expect_throw("Failed to get LWE secret keys")
                .len(),
        );

        for sk_info in key_set_info
            .get_lwe_secret_keys()
            .expect_throw("Failed to get LWE secret keys")
            .iter()
        {
            let lwe_dim = sk_info
                .get_params()
                .expect_throw("Failed to get secret key parameters")
                .get_lwe_dimension() as usize;
            let sk = generate_sk(lwe_dim, &mut secret_random_generator);
            let sk_builder = build_key!(
                sk,
                sk_info,
                concrete_protocol_capnp::lwe_secret_key::Builder
            );
            client_keyset
                .reborrow()
                .get_lwe_secret_keys()
                .expect_throw("Failed to get LWE secret keys builder")
                .set_with_caveats(
                    sk_info.get_id(),
                    sk_builder
                        .get_root_as_reader()
                        .expect_throw("Failed to get root as reader"),
                )
                .expect_throw("Failed to set LWE secret key");
        }

        serialize::write_message_to_words(&builder)
    }

    /// Get the client keyset from a keyset.
    ///
    /// # Arguments
    /// * `keyset`: The keyset to extract the client keyset from.
    ///
    /// # Returns
    /// A serialized client keyset.
    #[wasm_bindgen]
    pub fn get_client_keyset(keyset_buffer: &[u8]) -> Vec<u8> {
        get_client_keyset_from_buffers(keyset_buffer)
    }

    /// Explains a keyset info buffer in JSON representation.
    ///
    /// This function is intended to be used in a WebAssembly (WASM) environment.
    /// It reads the keyset info from a byte buffer (Capnp) and explains it in a JSON string.
    ///
    /// # Arguments
    ///
    /// * `keyset_info_buffer` - A byte slice containing the serialized keyset info (Capnp).
    ///
    /// # Returns
    ///
    /// A `JsValue` containing the JSON representation of the keyset info.
    #[wasm_bindgen]
    pub fn explain_keyset_info(keyset_info_buffer: &[u8]) -> JsValue {
        let message = read_capnp_from_buffer(keyset_info_buffer)
            .expect_throw("Failed to read keyset info buffer");
        let keyset_info =
            get_reader_from_message::<concrete_protocol_capnp::keyset_info::Reader>(&message)
                .expect_throw("Failed to get root keyset info reader");

        // Convert the keyset info to JSON
        let mut json = serde_json::Map::new();

        // Add LWE secret keys
        let lwe_secret_keys = keyset_info
            .get_lwe_secret_keys()
            .expect_throw("Failed to get LWE secret keys")
            .iter()
            .map(|key| {
                let mut key_json = serde_json::Map::new();
                {
                    let params = key
                        .get_params()
                        .expect_throw("Failed to get LWE secret key params");
                    let mut params_json = serde_json::Map::new();

                    params_json.insert("id".to_string(), serde_json::Value::from(key.get_id()));

                    params_json.insert(
                        "lwe_dimension".to_string(),
                        serde_json::Value::from(params.get_lwe_dimension()),
                    );

                    params_json.insert(
                        "key_type".to_string(),
                        serde_json::Value::from(
                            match params.get_key_type().expect_throw("Failed to get key type") {
                                KeyType::Binary => "binary",
                                KeyType::Ternary => "ternary",
                            },
                        ),
                    );

                    params_json.insert(
                        "integer_precision".to_string(),
                        serde_json::Value::from(params.get_integer_precision()),
                    );

                    key_json.insert("params".to_string(), serde_json::Value::Object(params_json));
                }
                serde_json::Value::Object(key_json)
            })
            .collect::<Vec<_>>();
        json.insert(
            "lwe_secret_keys".to_string(),
            serde_json::Value::Array(lwe_secret_keys),
        );

        // Add LWE bootstrap keys
        let lwe_bootstrap_keys = keyset_info
            .get_lwe_bootstrap_keys()
            .expect_throw("Failed to get LWE bootstrap keys")
            .iter()
            .map(|key| {
                let mut key_json = serde_json::Map::new();

                key_json.insert("id".to_string(), serde_json::Value::from(key.get_id()));
                key_json.insert(
                    "input_id".to_string(),
                    serde_json::Value::from(key.get_input_id()),
                );
                key_json.insert(
                    "output_id".to_string(),
                    serde_json::Value::from(key.get_output_id()),
                );
                key_json.insert(
                    "compression".to_string(),
                    serde_json::Value::from(
                        match key
                            .get_compression()
                            .expect_throw("Failed to get compression")
                        {
                            Compression::None => "none",
                            Compression::Seed => "seed",
                            Compression::Paillier => "paillier",
                        },
                    ),
                );

                {
                    let mut params_json = serde_json::Map::new();
                    let params = key
                        .get_params()
                        .expect_throw("Failed to get LWE bootstrap key params");

                    params_json.insert(
                        "input_lwe_dimension".to_string(),
                        serde_json::Value::from(params.get_input_lwe_dimension()),
                    );

                    params_json.insert(
                        "glwe_dimension".to_string(),
                        serde_json::Value::from(params.get_glwe_dimension()),
                    );

                    params_json.insert(
                        "polynomial_size".to_string(),
                        serde_json::Value::from(params.get_polynomial_size()),
                    );

                    params_json.insert(
                        "level_count".to_string(),
                        serde_json::Value::from(params.get_level_count()),
                    );

                    params_json.insert(
                        "base_log".to_string(),
                        serde_json::Value::from(params.get_base_log()),
                    );

                    params_json.insert(
                        "variance".to_string(),
                        serde_json::Value::from(params.get_variance()),
                    );

                    params_json.insert(
                        "integer_precision".to_string(),
                        serde_json::Value::from(params.get_integer_precision()),
                    );

                    params_json.insert(
                        "key_type".to_string(),
                        serde_json::Value::from(
                            match params.get_key_type().expect_throw("Failed to get key type") {
                                KeyType::Binary => "binary",
                                KeyType::Ternary => "ternary",
                            },
                        ),
                    );
                    key_json.insert("params".to_string(), serde_json::Value::Object(params_json));
                }
                serde_json::Value::Object(key_json)
            })
            .collect::<Vec<_>>();
        json.insert(
            "lwe_bootstrap_keys".to_string(),
            serde_json::Value::Array(lwe_bootstrap_keys),
        );

        // Add LWE keyswitch keys
        let lwe_keyswitch_keys = keyset_info
            .get_lwe_keyswitch_keys()
            .expect_throw("Failed to get LWE keyswitch keys")
            .iter()
            .map(|key| {
                let mut key_json = serde_json::Map::new();

                key_json.insert("id".to_string(), serde_json::Value::from(key.get_id()));
                key_json.insert(
                    "input_id".to_string(),
                    serde_json::Value::from(key.get_input_id()),
                );
                key_json.insert(
                    "output_id".to_string(),
                    serde_json::Value::from(key.get_output_id()),
                );
                key_json.insert(
                    "compression".to_string(),
                    serde_json::Value::from(
                        match key
                            .get_compression()
                            .expect_throw("Failed to get compression")
                        {
                            Compression::None => "none",
                            Compression::Seed => "seed",
                            Compression::Paillier => "paillier",
                        },
                    ),
                );

                {
                    let mut params_json = serde_json::Map::new();
                    let params = key
                        .get_params()
                        .expect_throw("Failed to get LWE keyswitch key params");

                    params_json.insert(
                        "input_lwe_dimension".to_string(),
                        serde_json::Value::from(params.get_input_lwe_dimension()),
                    );

                    params_json.insert(
                        "output_lwe_dimension".to_string(),
                        serde_json::Value::from(params.get_output_lwe_dimension()),
                    );

                    params_json.insert(
                        "level_count".to_string(),
                        serde_json::Value::from(params.get_level_count()),
                    );

                    params_json.insert(
                        "base_log".to_string(),
                        serde_json::Value::from(params.get_base_log()),
                    );

                    params_json.insert(
                        "variance".to_string(),
                        serde_json::Value::from(params.get_variance()),
                    );

                    params_json.insert(
                        "integer_precision".to_string(),
                        serde_json::Value::from(params.get_integer_precision()),
                    );

                    params_json.insert(
                        "key_type".to_string(),
                        serde_json::Value::from(
                            match params.get_key_type().expect_throw("Failed to get key type") {
                                KeyType::Binary => "binary",
                                KeyType::Ternary => "ternary",
                            },
                        ),
                    );
                    key_json.insert("params".to_string(), serde_json::Value::Object(params_json));
                }
                serde_json::Value::Object(key_json)
            })
            .collect::<Vec<_>>();
        json.insert(
            "lwe_keyswitch_keys".to_string(),
            serde_json::Value::Array(lwe_keyswitch_keys),
        );

        web_sys::js_sys::JSON::parse(
            &serde_json::to_string(&json).expect_throw("Failed serializing built JSON"),
        )
        .expect_throw("Failed parsing built JSON")
    }
}
