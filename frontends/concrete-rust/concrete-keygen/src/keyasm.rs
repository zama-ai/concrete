use std::io::Read;

use concrete_keygen::concrete_protocol_capnp;
use tfhe::core_crypto::prelude::{
    allocate_and_assemble_lwe_bootstrap_key_from_chunks, LweBootstrapKeyChunk,
};
use tfhe::safe_serialization::safe_deserialize;

use clap::{Arg, Command};
use zip::result::ZipResult;
use zip::ZipArchive;

const KEYSET_INFO_FILENAME: &str = "keyset_info.capnp";
const KEYSET_FILENAME: &str = "keyset_no_bsk.capnp";

fn bsk_chunk_filename(bsk_id: u32, chunk_id: u32) -> String {
    format!("bsk_{}_chunk_{}", bsk_id, chunk_id)
}

fn new_zip_archive(path: &str) -> ZipResult<ZipArchive<std::fs::File>> {
    let file = std::fs::File::open(path).unwrap();
    ZipArchive::new(file)
}

fn assemble_keyset_from_zip(
    zip_path: &str,
    output_keyset_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut archive = new_zip_archive(zip_path).unwrap();
    // Read keyset info
    let mut keyset_info_file = archive.by_name(KEYSET_INFO_FILENAME).unwrap();
    let mut keyset_info_buffer = Vec::new();
    keyset_info_file
        .read_to_end(&mut keyset_info_buffer)
        .unwrap();
    let keyset_info_message =
        concrete_protocol_capnp::read_capnp_from_buffer(&keyset_info_buffer).unwrap();
    let keyset_info_proto: concrete_protocol_capnp::keyset_info::Reader<'_> =
        concrete_protocol_capnp::get_reader_from_message(&keyset_info_message).unwrap();

    // Read and assemble bootstrap keys
    let bsk_ids: Vec<u32> = keyset_info_proto
        .get_lwe_bootstrap_keys()
        .unwrap()
        .into_iter()
        .map(|bsk| bsk.get_id())
        .collect();
    let mut bsk_protos = Vec::new();
    for bsk_id in bsk_ids {
        let mut chunks = Vec::new();
        let mut chunk_id = 0;

        loop {
            let chunk_filename = bsk_chunk_filename(bsk_id, chunk_id);
            let mut archive = new_zip_archive(zip_path).unwrap();
            if let Ok(mut chunk_file) = archive.by_name(&chunk_filename) {
                let mut chunk_buffer = Vec::new();
                chunk_file.read_to_end(&mut chunk_buffer).unwrap();
                let serialized_size_limit = 10_000_000_000;
                let chunk: LweBootstrapKeyChunk<Vec<u64>> =
                    safe_deserialize(chunk_buffer.as_slice(), serialized_size_limit).unwrap();
                chunks.push(chunk);
                chunk_id += 1;
            } else {
                break;
            };
        }

        let bsk = allocate_and_assemble_lwe_bootstrap_key_from_chunks(&chunks);
        let bsk_proto = concrete_keygen::build_bsk_proto(keyset_info_proto, bsk_id, &bsk);
        bsk_protos.push(bsk_proto);
    }

    let bsk_readers = bsk_protos
        .iter_mut()
        .map(|bsk_proto| {
            concrete_protocol_capnp::get_reader_from_builder(bsk_proto)
                .expect("Failed to get bsk reader")
        })
        .collect();

    // Read keyset without BSks
    let mut archive = new_zip_archive(zip_path).unwrap();
    let mut keyset_file = archive.by_name(KEYSET_FILENAME).unwrap();
    let mut keyset_buffer = Vec::new();
    keyset_file.read_to_end(&mut keyset_buffer).unwrap();
    let mut reader_options = capnp::message::ReaderOptions::new();
    reader_options.traversal_limit_in_words(Some(10_000_000_000));
    let keyset_message = concrete_protocol_capnp::read_capnp_from_buffer(&keyset_buffer).unwrap();
    let keyset = concrete_protocol_capnp::get_reader_from_message(&keyset_message).unwrap();

    // Add BSks to keyset
    let keyset_with_bsks = concrete_keygen::add_bsk_keys_to_keyset(keyset, bsk_readers);

    // Write the final keyset to the output file
    let mut output_file = std::fs::File::create(output_keyset_path).unwrap();
    capnp::serialize::write_message(&mut output_file, &keyset_with_bsks).unwrap();

    Ok(())
}

pub fn main() {
    let matches = Command::new("concrete-keyasm")
        .about("Concrete Key Assembler: assemble a keyset generated in a chunked way.")
        .arg_required_else_help(true)
        .arg(
            Arg::new("chunked_keyset_zip")
                .help("Path to the input chunked keyset file (Zip)")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("output_keyset")
                .help("Path to the output keyset file")
                .required(true)
                .index(2),
        )
        .get_matches();

    let chunked_keyset_zip = matches.get_one::<String>("chunked_keyset_zip").unwrap();
    let output_keyset_path = matches.get_one::<String>("output_keyset").unwrap();
    assemble_keyset_from_zip(&chunked_keyset_zip, &output_keyset_path).unwrap();
}
