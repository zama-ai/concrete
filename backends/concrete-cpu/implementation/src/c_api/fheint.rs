use super::utils::nounwind;
use core::slice;
use std::io::Cursor;
use tfhe::core_crypto::prelude::*;
use tfhe::integer::ciphertext::Expandable;
use tfhe::integer::IntegerCiphertext;
use tfhe::shortint::parameters::{Degree, NoiseLevel};
use tfhe::shortint::{CarryModulus, Ciphertext, MessageModulus};
use tfhe::FheUint8;

#[repr(C)]
pub struct TfhersFheIntDescription {
    width: usize,
    is_signed: bool,
    lwe_size: usize,
    n_cts: usize,
    degree: usize,
    noise_level: usize,
    message_modulus: usize,
    carry_modulus: usize,
    ks_first: bool,
}

impl TfhersFheIntDescription {
    fn zero() -> TfhersFheIntDescription {
        TfhersFheIntDescription {
            width: 0,
            is_signed: false,
            lwe_size: 0,
            n_cts: 0,
            degree: 0,
            noise_level: 0,
            message_modulus: 0,
            carry_modulus: 0,
            ks_first: false,
        }
    }

    /// Create a `Ciphertext` from the lwe and `self` metadata
    fn ct_from_lwe(&self, lwe: LweCiphertext<Vec<u64>>) -> Ciphertext {
        // all this if/else due to the fact that we can't construct a NoiseLevel (not public)
        let noise_level = if self.noise_level == NoiseLevel::ZERO.get() {
            NoiseLevel::ZERO
        } else if self.noise_level == NoiseLevel::NOMINAL.get() {
            NoiseLevel::NOMINAL
        } else if self.noise_level == NoiseLevel::MAX.get() {
            NoiseLevel::MAX
        } else if self.noise_level == NoiseLevel::UNKNOWN.get() {
            NoiseLevel::UNKNOWN
        } else {
            panic!("invalid noise level value");
        };

        Ciphertext::new(
            lwe,
            Degree::new(self.degree),
            noise_level,
            MessageModulus(self.message_modulus),
            CarryModulus(self.carry_modulus),
            if self.ks_first {
                PBSOrder::KeyswitchBootstrap
            } else {
                PBSOrder::BootstrapKeyswitch
            },
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_tfhers_unknown_noise_level() -> usize {
    NoiseLevel::UNKNOWN.get()
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_tfhers_uint8_description(
    serialized_data_ptr: *const u8,
    serialized_data_len: usize,
) -> TfhersFheIntDescription {
    nounwind(|| {
        // deserialize fheuint8
        let mut serialized_data = Cursor::new(slice::from_raw_parts(
            serialized_data_ptr,
            serialized_data_len,
        ));
        let fheuint: FheUint8 = match bincode::deserialize_from(&mut serialized_data) {
            Ok(value) => value,
            Err(_) => {
                return TfhersFheIntDescription::zero();
            }
        };
        // get metadata from fheuint's ciphertext
        let (radix, _) = fheuint.into_raw_parts();
        let blocks = radix.blocks();
        let ct = match blocks.first() {
            Some(value) => &value.ct,
            None => {
                return TfhersFheIntDescription::zero();
            }
        };
        TfhersFheIntDescription {
            width: 8,
            is_signed: false,
            lwe_size: ct.lwe_size().0,
            n_cts: blocks.len(),
            degree: blocks[0].degree.get(),
            noise_level: blocks[0].noise_level().get(),
            message_modulus: blocks[0].message_modulus.0,
            carry_modulus: blocks[0].carry_modulus.0,
            ks_first: blocks[0].pbs_order == PBSOrder::KeyswitchBootstrap,
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_tfhers_uint8_to_lwe_array(
    serialized_data_ptr: *const u8,
    serialized_data_len: usize,
    lwe_vec_buffer: *mut u64,
) -> i64 {
    nounwind(|| {
        // deserialize fheuint8
        let mut serialized_data = Cursor::new(slice::from_raw_parts(
            serialized_data_ptr,
            serialized_data_len,
        ));
        // TODO: can we have a generic deserialize?
        let fheuint: FheUint8 = match bincode::deserialize_from(&mut serialized_data) {
            Ok(value) => value,
            Err(_) => {
                return 1;
            }
        };
        // collect LWEs from fheuint
        let (radix, _) = fheuint.into_raw_parts();
        let blocks = radix.blocks();
        let first_ct = match blocks.first() {
            Some(value) => &value.ct,
            None => return 1,
        };
        let lwe_size = first_ct.lwe_size().0;
        let n_cts = blocks.len();
        // copy LWEs to C buffer. Note that lsb is cts[0]
        let lwe_vector: &mut [u64] = slice::from_raw_parts_mut(lwe_vec_buffer, n_cts * lwe_size);
        for (i, block) in blocks.iter().enumerate() {
            lwe_vector[i * lwe_size..(i + 1) * lwe_size]
                .copy_from_slice(block.ct.clone().into_container().as_slice());
        }
        0
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_lwe_array_to_tfhers_uint8(
    lwe_vec_buffer: *const u64,
    fheuint_buffer: *mut u8,
    fheuint_buffer_size: usize,
    fheuint_desc: TfhersFheIntDescription,
) -> usize {
    nounwind(|| {
        // we want to trigger a PBS on TFHErs side
        assert!(
            fheuint_desc.noise_level == NoiseLevel::UNKNOWN.get(),
            "noise_level must be unknown"
        );
        // we want to use the max degree as we don't track it on Concrete side
        assert!(
            fheuint_desc.degree == fheuint_desc.message_modulus - 1,
            "degree must be the max value (msg_modulus - 1)"
        );

        let lwe_size = fheuint_desc.lwe_size;
        let n_cts = fheuint_desc.n_cts;
        // construct fheuint from LWEs
        let lwe_vector: &[u64] = slice::from_raw_parts(lwe_vec_buffer, n_cts * lwe_size);
        let mut blocks: Vec<Ciphertext> = Vec::new();
        for i in 0..n_cts {
            let lwe_ct = LweCiphertext::<Vec<u64>>::from_container(
                lwe_vector[i * lwe_size..(i + 1) * lwe_size].to_vec(),
                CiphertextModulus::new_native(),
            );
            blocks.push(fheuint_desc.ct_from_lwe(lwe_ct));
        }
        let fheuint = match FheUint8::from_expanded_blocks(
            blocks,
            tfhe::integer::ciphertext::DataKind::Unsigned(4),
        ) {
            Ok(value) => value,
            Err(_) => {
                return 0;
            }
        };
        // serialize fheuint
        let mut serialized_data = Vec::new();
        match bincode::serialize_into(&mut serialized_data, &fheuint) {
            Ok(_) => (),
            Err(_) => {
                return 0;
            }
        }
        if serialized_data.len() > fheuint_buffer_size {
            return 0;
        }
        let buff: &mut [u8] = slice::from_raw_parts_mut(fheuint_buffer, serialized_data.len());
        buff.copy_from_slice(&serialized_data);
        serialized_data.len()
    })
}
