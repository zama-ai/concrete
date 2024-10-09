use bincode;
use tfhe::core_crypto::prelude::LweSecretKey;
use tfhe::{ConfigBuilder, generate_keys, FheUint8};
use tfhe::shortint::{ClassicPBSParameters, EncryptionKeyChoice};
use tfhe::prelude::*;
use std::path::Path;
use std::fs;
use std::io::Cursor;
use std::process::Command;
use std::io::{self, Write};
use rand::Rng;

const BLOCK_PARAMS: ClassicPBSParameters = tfhe::shortint::prelude::PARAM_MESSAGE_2_CARRY_3_KS_PBS;

fn serialize_lwesecretkey(lwe_secret_key: LweSecretKey<Vec<u64>>, lwe_secret_key_path: &str) {
    let mut serialized_lwe_secret_key = Vec::new();
    bincode::serialize_into(&mut serialized_lwe_secret_key, &lwe_secret_key).unwrap();
    let path_lwe_secret_key: &Path = Path::new(lwe_secret_key_path);
    fs::write(path_lwe_secret_key, serialized_lwe_secret_key).unwrap();
}

fn serialize_fheuint8(fheuint: FheUint8, ciphertext_path: &str) {
    let mut serialized_ct = Vec::new();
    bincode::serialize_into(&mut serialized_ct, &fheuint).unwrap();
    let path_ct: &Path = Path::new(ciphertext_path);
    fs::write(path_ct, serialized_ct).unwrap();
}

fn deserialize_fheuint8(path: &str) -> FheUint8 {
    let path_fheuint: &Path = Path::new(path);
    let serialized_fheuint = fs::read(path_fheuint).unwrap();
    let mut serialized_data = Cursor::new(serialized_fheuint);
    bincode::deserialize_from(&mut serialized_data).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>>{
    let config = ConfigBuilder::with_custom_parameters(BLOCK_PARAMS).build();

    // Step A: Key Generation
    // This is done only once, on the client side

    // Client key and server keys, in TFHE-rs format
    let (client_key, _) = generate_keys(config);

    // Get lwe_secret_key, to be reused to generate the server key in Concrete format
    let (integer_ck, _, _) = client_key.clone().into_raw_parts();
    let shortint_ck = integer_ck.into_raw_parts();
    assert!(BLOCK_PARAMS.encryption_key_choice == EncryptionKeyChoice::Big);
    let (glwe_secret_key, _, _) = shortint_ck.into_raw_parts();
    let lwe_secret_key = glwe_secret_key.into_lwe_secret_key();

    // Save the LWE secret key, for a compatible keygen to be possible in Concrete
    serialize_lwesecretkey(lwe_secret_key.clone(),  "client_dir/tfhers_sk.txt");

    // Call the Python script, to compute the evaluation key in the Concrete
    let output_keygen = Command::new("python")
            .arg("test.py")
            .arg("keygen")
            .arg("-s")
            .arg("client_dir/tfhers_sk.txt")
            .arg("-k")
            .arg("server_dir/concrete_keyset.txt")
            .output()
            .expect("command failed to start");

    if output_keygen.status.code() != Some(0)
    {
        println!("status: {}", output_keygen.status);
        io::stdout().write_all(&output_keygen.stdout).unwrap();
        io::stderr().write_all(&output_keygen.stderr).unwrap();
    }

    assert!(output_keygen.status.success());

    // Step B: Encryptions
    // This is done on the client side, several times
    let mut i = 0;

    let nb_samples = 1;
    let nb_parameters_in_function = 4;

    while i < nb_samples {

        i = i + 1;

        // Pick random UInt8's
        let mut rng = rand::thread_rng();
        let mut vec_clear = Vec::new();
        let mut j = 0;
        let mut filenames_for_commandline: String = "".to_owned();

        while j < nb_parameters_in_function {
            // FIXME: remove the modulo 128
            let clear: u8 = rng.gen_range(0..128);

            println!("Encrypting: {clear}");

            vec_clear.push(clear);

            let ciphertext = FheUint8::encrypt(clear, &client_key);

            let filename = format!("server_dir/ciphertext_{}.txt", j);

            serialize_fheuint8(ciphertext.clone(), &filename);

            j = j + 1;
            filenames_for_commandline = [filenames_for_commandline, filename].join(" ");
        }

        // Step C: Computations in Concrete
        // Computations are done on the server side
        let output_run = Command::new("python")
                .arg("test.py")
                .arg("run")
                .arg("-k")
                .arg("server_dir/concrete_keyset.txt")
                .arg("-c")
                .arg(filenames_for_commandline.clone())
                .arg("-o")
                .arg("server_dir/ciphertext_r.txt")
                .output()
                .expect("command failed to start");

        if output_run.status.code() != Some(0)
        {
            println!("status: {}", output_run.status);
            io::stdout().write_all(&output_run.stdout).unwrap();
            io::stderr().write_all(&output_run.stderr).unwrap();
        }

        assert!(output_run.status.success());

        // Step D: Decryptions
        // This is done on the client side

        // Read in the file and unserialize
        let result = deserialize_fheuint8("server_dir/ciphertext_r.txt");

        //Client-side
        let decrypted_result: u8 = result.decrypt(&client_key);

        // Check the result was computed correctly
        let clear_result_u16: u16 = (u16::from(vec_clear[0]) +
                                     u16::from(vec_clear[1]) +
                                     (2 * u16::from(vec_clear[2])) -
                                     u16::from(vec_clear[3])
                                    ) % 47;
        let clear_result: u8 = clear_result_u16 as u8;

        println!("Expecting {clear_result}. Got {decrypted_result}");

        assert_eq!(decrypted_result, clear_result);
    }

    println!("Successful end");
    Ok(())
}
