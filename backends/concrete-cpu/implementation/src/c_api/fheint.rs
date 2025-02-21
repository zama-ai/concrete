use super::utils::nounwind;
use core::slice;
use tfhe::core_crypto::prelude::*;
use tfhe::integer::ciphertext::Expandable;
use tfhe::integer::IntegerCiphertext;
use tfhe::shortint::parameters::{Degree, NoiseLevel};
use tfhe::shortint::{CarryModulus, Ciphertext, MessageModulus};
use tfhe::{
    FheInt10, FheInt12, FheInt14, FheInt16, FheInt2, FheInt4, FheInt6, FheInt8, FheUint10,
    FheUint12, FheUint128, FheUint14, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8,
};

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

    /// Get DataKind
    fn data_kind(&self) -> tfhe::integer::ciphertext::DataKind {
        match self.is_signed {
            true => tfhe::integer::ciphertext::DataKind::Signed(self.width),
            false => tfhe::integer::ciphertext::DataKind::Unsigned(self.width),
        }
    }

    /// Does the main parameters match
    fn is_similar(&self, other: &TfhersFheIntDescription) -> bool {
        if self.width != other.width {
            return false;
        }
        if self.is_signed != other.is_signed {
            return false;
        }
        if self.lwe_size != other.lwe_size {
            return false;
        }
        if self.n_cts != other.n_cts {
            return false;
        }
        if self.message_modulus != other.message_modulus {
            return false;
        }
        if self.carry_modulus != other.carry_modulus {
            return false;
        }
        if self.ks_first != other.ks_first {
            return false;
        }
        true
    }
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_tfhers_unknown_noise_level() -> usize {
    NoiseLevel::UNKNOWN.get()
}

macro_rules! tfhers_description {
    ($name:ident, $type:ty, $width:expr, $is_signed:expr) => {
        pub fn $name(fheint: $type) -> TfhersFheIntDescription {
            // get metadata from fheint's ciphertext
            let (radix, _, _) = fheint.into_raw_parts();
            let blocks = radix.blocks();
            let ct = match blocks.first() {
                Some(value) => &value.ct,
                None => {
                    return TfhersFheIntDescription::zero();
                }
            };
            TfhersFheIntDescription {
                width: $width,
                is_signed: $is_signed,
                lwe_size: ct.lwe_size().0,
                n_cts: blocks.len(),
                degree: blocks[0].degree.get(),
                noise_level: blocks[0].noise_level().get(),
                message_modulus: blocks[0].message_modulus.0,
                carry_modulus: blocks[0].carry_modulus.0,
                ks_first: blocks[0].pbs_order == PBSOrder::KeyswitchBootstrap,
            }
        }
    };
}

tfhers_description!(tfhers_uint2_description, FheUint2, 2, false);
tfhers_description!(tfhers_int2_description, FheInt2, 2, true);
tfhers_description!(tfhers_uint4_description, FheUint4, 4, false);
tfhers_description!(tfhers_int4_description, FheInt4, 4, true);
tfhers_description!(tfhers_uint6_description, FheUint6, 6, false);
tfhers_description!(tfhers_int6_description, FheInt6, 6, true);
tfhers_description!(tfhers_uint8_description, FheUint8, 8, false);
tfhers_description!(tfhers_int8_description, FheInt8, 8, true);
tfhers_description!(tfhers_uint10_description, FheUint10, 10, false);
tfhers_description!(tfhers_int10_description, FheInt10, 10, true);
tfhers_description!(tfhers_uint12_description, FheUint12, 12, false);
tfhers_description!(tfhers_int12_description, FheInt12, 12, true);
tfhers_description!(tfhers_uint14_description, FheUint14, 14, false);
tfhers_description!(tfhers_int14_description, FheInt14, 14, true);
tfhers_description!(tfhers_uint16_description, FheUint16, 16, false);
tfhers_description!(tfhers_int16_description, FheInt16, 16, true);

macro_rules! tfhers_int_to_lwe_array {
    ($name:ident, $description_func:ident, $type:ty) => {
        unsafe fn $name(
            buffer: *const u8,
            buffer_len: usize,
            lwe_vec_buffer: *mut u64,
            desc: TfhersFheIntDescription,
        ) -> i64 {
            nounwind(|| {
                let fheint: $type = super::utils::safe_deserialize(buffer, buffer_len);
                // TODO - Use conformance check
                let fheint_desc = $description_func(fheint.clone());
                if !fheint_desc.is_similar(&desc) {
                    return 1;
                }

                let lwe_size = desc.lwe_size;
                let n_cts = desc.n_cts;
                // collect LWEs from fheint
                let (radix, _, _) = fheint.into_raw_parts();
                let blocks = radix.blocks();
                // copy LWEs to C buffer. Note that lsb is cts[0]
                let lwe_vector: &mut [u64] =
                    slice::from_raw_parts_mut(lwe_vec_buffer, n_cts * lwe_size);
                for (i, block) in blocks.iter().enumerate() {
                    lwe_vector[i * lwe_size..(i + 1) * lwe_size]
                        .copy_from_slice(block.ct.clone().into_container().as_slice());
                }
                0
            })
        }
    };
}

tfhers_int_to_lwe_array!(
    tfhers_uint2_to_lwe_array,
    tfhers_uint2_description,
    FheUint2
);
tfhers_int_to_lwe_array!(tfhers_int2_to_lwe_array, tfhers_int2_description, FheInt2);
tfhers_int_to_lwe_array!(
    tfhers_uint4_to_lwe_array,
    tfhers_uint4_description,
    FheUint4
);
tfhers_int_to_lwe_array!(tfhers_int4_to_lwe_array, tfhers_int4_description, FheInt4);
tfhers_int_to_lwe_array!(
    tfhers_uint6_to_lwe_array,
    tfhers_uint6_description,
    FheUint6
);
tfhers_int_to_lwe_array!(tfhers_int6_to_lwe_array, tfhers_int6_description, FheInt6);
tfhers_int_to_lwe_array!(
    tfhers_uint8_to_lwe_array,
    tfhers_uint8_description,
    FheUint8
);

tfhers_int_to_lwe_array!(tfhers_int8_to_lwe_array, tfhers_int8_description, FheInt8);
tfhers_int_to_lwe_array!(
    tfhers_uint10_to_lwe_array,
    tfhers_uint10_description,
    FheUint10
);
tfhers_int_to_lwe_array!(
    tfhers_int10_to_lwe_array,
    tfhers_int10_description,
    FheInt10
);
tfhers_int_to_lwe_array!(
    tfhers_uint12_to_lwe_array,
    tfhers_uint12_description,
    FheUint12
);
tfhers_int_to_lwe_array!(
    tfhers_int12_to_lwe_array,
    tfhers_int12_description,
    FheInt12
);
tfhers_int_to_lwe_array!(
    tfhers_uint14_to_lwe_array,
    tfhers_uint14_description,
    FheUint14
);
tfhers_int_to_lwe_array!(
    tfhers_int14_to_lwe_array,
    tfhers_int14_description,
    FheInt14
);
tfhers_int_to_lwe_array!(
    tfhers_uint16_to_lwe_array,
    tfhers_uint16_description,
    FheUint16
);

tfhers_int_to_lwe_array!(
    tfhers_int16_to_lwe_array,
    tfhers_int16_description,
    FheInt16
);

macro_rules! tfhers_array_to_lwe_array {
    ($name:ident, $description_func:ident, $type:ty) => {
        unsafe fn $name(
            buffer: *const u8,
            buffer_len: usize,
            mut lwe_vec_buffer: *mut u64,
            desc: TfhersFheIntDescription,
        ) -> i64 {
            nounwind(|| {
                let fheint_array: Vec<$type> = super::utils::unsafe_deserialize(buffer, buffer_len);
                // TODO - Use conformance check
                let fhe_desc = $description_func(fheint_array[0].clone());
                if !fhe_desc.is_similar(&desc) {
                    return 1;
                }

                let lwe_size: usize = desc.lwe_size;
                let n_cts: usize = desc.n_cts;
                let blocks_size = n_cts * lwe_size;
                // collect LWEs from fhe
                for fheint in fheint_array {
                    let (radix, _, _) = fheint.into_raw_parts();
                    let blocks = radix.blocks();
                    // copy LWEs to C buffer. Note that lsb is cts[0]
                    let lwe_vector: &mut [u64] =
                        slice::from_raw_parts_mut(lwe_vec_buffer, n_cts * lwe_size);
                    for (i, block) in blocks.iter().enumerate() {
                        lwe_vector[i * lwe_size..(i + 1) * lwe_size]
                            .copy_from_slice(block.ct.clone().into_container().as_slice());
                    }
                    // shift to next block
                    lwe_vec_buffer = lwe_vec_buffer.add(blocks_size);
                }
                0
            })
        }
    };
}

tfhers_array_to_lwe_array!(
    tfhers_uint2_array_to_lwe_array,
    tfhers_uint2_description,
    FheUint2
);
tfhers_array_to_lwe_array!(
    tfhers_int2_array_to_lwe_array,
    tfhers_int2_description,
    FheInt2
);
tfhers_array_to_lwe_array!(
    tfhers_uint4_array_to_lwe_array,
    tfhers_uint4_description,
    FheUint4
);
tfhers_array_to_lwe_array!(
    tfhers_int4_array_to_lwe_array,
    tfhers_int4_description,
    FheInt4
);
tfhers_array_to_lwe_array!(
    tfhers_uint6_array_to_lwe_array,
    tfhers_uint6_description,
    FheUint6
);
tfhers_array_to_lwe_array!(
    tfhers_int6_array_to_lwe_array,
    tfhers_int6_description,
    FheInt6
);
tfhers_array_to_lwe_array!(
    tfhers_uint8_array_to_lwe_array,
    tfhers_uint8_description,
    FheUint8
);
tfhers_array_to_lwe_array!(
    tfhers_int8_array_to_lwe_array,
    tfhers_int8_description,
    FheInt8
);
tfhers_array_to_lwe_array!(
    tfhers_uint10_array_to_lwe_array,
    tfhers_uint10_description,
    FheUint10
);
tfhers_array_to_lwe_array!(
    tfhers_int10_array_to_lwe_array,
    tfhers_int10_description,
    FheInt10
);
tfhers_array_to_lwe_array!(
    tfhers_uint12_array_to_lwe_array,
    tfhers_uint12_description,
    FheUint12
);
tfhers_array_to_lwe_array!(
    tfhers_int12_array_to_lwe_array,
    tfhers_int12_description,
    FheInt12
);
tfhers_array_to_lwe_array!(
    tfhers_uint14_array_to_lwe_array,
    tfhers_uint14_description,
    FheUint14
);
tfhers_array_to_lwe_array!(
    tfhers_int14_array_to_lwe_array,
    tfhers_int14_description,
    FheInt14
);
tfhers_array_to_lwe_array!(
    tfhers_uint16_array_to_lwe_array,
    tfhers_uint16_description,
    FheUint16
);
tfhers_array_to_lwe_array!(
    tfhers_int16_array_to_lwe_array,
    tfhers_int16_description,
    FheInt16
);

macro_rules! lwe_array_to_tfhers {
    ($name:ident, $type:ty) => {
        unsafe fn $name(
            lwe_vec_buffer: *const u64,
            buffer: *mut u8,
            buffer_len: usize,
            desc: TfhersFheIntDescription,
        ) -> usize {
            nounwind(|| {
                // we want to trigger a PBS on TFHErs side
                assert!(
                    desc.noise_level == NoiseLevel::UNKNOWN.get(),
                    "noise_level must be unknown"
                );
                // we want to use the max degree as we don't track it on Concrete side
                assert!(
                    desc.degree == desc.message_modulus - 1,
                    "degree must be the max value (msg_modulus - 1)"
                );

                let lwe_size = desc.lwe_size;
                let n_cts = desc.n_cts;
                // construct fheint from LWEs
                let lwe_vector: &[u64] = slice::from_raw_parts(lwe_vec_buffer, n_cts * lwe_size);
                let mut blocks: Vec<Ciphertext> = Vec::with_capacity(n_cts);
                for i in 0..n_cts {
                    let lwe_ct = LweCiphertext::<Vec<u64>>::from_container(
                        lwe_vector[i * lwe_size..(i + 1) * lwe_size].to_vec(),
                        CiphertextModulus::new_native(),
                    );
                    blocks.push(desc.ct_from_lwe(lwe_ct));
                }
                let fheint = match <$type>::from_expanded_blocks(blocks, desc.data_kind()) {
                    Ok(value) => value,
                    Err(_e) => {
                        return 0;
                    }
                };
                super::utils::safe_serialize(&fheint, buffer, buffer_len)
            })
        }
    };
}

lwe_array_to_tfhers!(lwe_array_to_tfhers_uint2, FheUint2);
lwe_array_to_tfhers!(lwe_array_to_tfhers_int2, FheInt2);
lwe_array_to_tfhers!(lwe_array_to_tfhers_uint4, FheUint4);
lwe_array_to_tfhers!(lwe_array_to_tfhers_int4, FheInt4);
lwe_array_to_tfhers!(lwe_array_to_tfhers_uint6, FheUint6);
lwe_array_to_tfhers!(lwe_array_to_tfhers_int6, FheInt6);
lwe_array_to_tfhers!(lwe_array_to_tfhers_uint8, FheUint8);
lwe_array_to_tfhers!(lwe_array_to_tfhers_int8, FheInt8);
lwe_array_to_tfhers!(lwe_array_to_tfhers_uint10, FheUint10);
lwe_array_to_tfhers!(lwe_array_to_tfhers_int10, FheInt10);
lwe_array_to_tfhers!(lwe_array_to_tfhers_uint12, FheUint12);
lwe_array_to_tfhers!(lwe_array_to_tfhers_int12, FheInt12);
lwe_array_to_tfhers!(lwe_array_to_tfhers_uint14, FheUint14);
lwe_array_to_tfhers!(lwe_array_to_tfhers_int14, FheInt14);
lwe_array_to_tfhers!(lwe_array_to_tfhers_uint16, FheUint16);
lwe_array_to_tfhers!(lwe_array_to_tfhers_int16, FheInt16);

macro_rules! lwe_array_to_tfhers_array {
    ($name:ident, $type:ty) => {
        unsafe fn $name(
            lwe_vec_buffer: *const u64,
            buffer: *mut u8,
            buffer_len: usize,
            n_elem: usize,
            desc: TfhersFheIntDescription,
        ) -> usize {
            nounwind(|| {
                // we want to trigger a PBS on TFHErs side
                assert!(
                    desc.noise_level == NoiseLevel::UNKNOWN.get(),
                    "noise_level must be unknown"
                );
                // we want to use the max degree as we don't track it on Concrete side
                assert!(
                    desc.degree == desc.message_modulus - 1,
                    "degree must be the max value (msg_modulus - 1)"
                );

                let lwe_size = desc.lwe_size;
                let n_cts = desc.n_cts;
                // construct fheint from LWEs
                let lwe_vector: &[u64] =
                    slice::from_raw_parts(lwe_vec_buffer, n_elem * n_cts * lwe_size);
                let mut fheint_array: Vec<$type> = Vec::with_capacity(n_elem);
                for i in 0..n_elem {
                    let elem_offset = i * n_cts * lwe_size;
                    let mut blocks: Vec<Ciphertext> = Vec::with_capacity(n_cts);
                    for j in 0..n_cts {
                        let lwe_ct = LweCiphertext::<Vec<u64>>::from_container(
                            lwe_vector
                                [elem_offset + j * lwe_size..elem_offset + (j + 1) * lwe_size]
                                .to_vec(),
                            CiphertextModulus::new_native(),
                        );
                        blocks.push(desc.ct_from_lwe(lwe_ct));
                    }
                    let fheint = match <$type>::from_expanded_blocks(blocks, desc.data_kind()) {
                        Ok(value) => value,
                        Err(_) => {
                            return 0;
                        }
                    };
                    fheint_array.push(fheint);
                }
                super::utils::unsafe_serialize(&fheint_array, buffer, buffer_len)
            })
        }
    };
}

lwe_array_to_tfhers_array!(lwe_array_to_tfhers_uint2_array, FheUint2);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_int2_array, FheInt2);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_uint4_array, FheUint4);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_int4_array, FheInt4);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_uint6_array, FheUint6);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_int6_array, FheInt6);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_uint8_array, FheUint8);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_int8_array, FheInt8);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_uint10_array, FheUint10);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_int10_array, FheInt10);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_uint12_array, FheUint12);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_int12_array, FheInt12);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_uint14_array, FheUint14);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_int14_array, FheInt14);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_uint16_array, FheUint16);
lwe_array_to_tfhers_array!(lwe_array_to_tfhers_int16_array, FheInt16);

unsafe fn tfhers_uint_to_lwe_array(
    buffer: *const u8,
    buffer_len: usize,
    lwe_vec_buffer: *mut u64,
    n_elem: usize,
    desc: TfhersFheIntDescription,
    bitwidth: usize,
) -> i64 {
    assert!(n_elem > 0);
    match bitwidth {
        2 => {
            if n_elem == 1 {
                tfhers_uint2_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_uint2_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        4 => {
            if n_elem == 1 {
                tfhers_uint4_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_uint4_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        6 => {
            if n_elem == 1 {
                tfhers_uint6_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_uint6_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        8 => {
            if n_elem == 1 {
                tfhers_uint8_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_uint8_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        10 => {
            if n_elem == 1 {
                tfhers_uint10_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_uint10_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        12 => {
            if n_elem == 1 {
                tfhers_uint12_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_uint12_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        14 => {
            if n_elem == 1 {
                tfhers_uint14_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_uint14_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        16 => {
            if n_elem == 1 {
                tfhers_uint16_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_uint16_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        _ => panic!("Unsupported bitwidth"),
    }
}

macro_rules! concrete_cpu_tfhers_to_lwe_array {
    ($name:ident, $bitwidth:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            buffer: *const u8,
            buffer_len: usize,
            lwe_vec_buffer: *mut u64,
            n_elem: usize,
            desc: TfhersFheIntDescription,
        ) -> i64 {
            tfhers_uint_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, n_elem, desc, $bitwidth)
        }
    };
}

concrete_cpu_tfhers_to_lwe_array!(concrete_cpu_tfhers_uint2_to_lwe_array, 2);
concrete_cpu_tfhers_to_lwe_array!(concrete_cpu_tfhers_uint4_to_lwe_array, 4);
concrete_cpu_tfhers_to_lwe_array!(concrete_cpu_tfhers_uint6_to_lwe_array, 6);
concrete_cpu_tfhers_to_lwe_array!(concrete_cpu_tfhers_uint8_to_lwe_array, 8);
concrete_cpu_tfhers_to_lwe_array!(concrete_cpu_tfhers_uint10_to_lwe_array, 10);
concrete_cpu_tfhers_to_lwe_array!(concrete_cpu_tfhers_uint12_to_lwe_array, 12);
concrete_cpu_tfhers_to_lwe_array!(concrete_cpu_tfhers_uint14_to_lwe_array, 14);
concrete_cpu_tfhers_to_lwe_array!(concrete_cpu_tfhers_uint16_to_lwe_array, 16);

unsafe fn tfhers_int_to_lwe_array(
    buffer: *const u8,
    buffer_len: usize,
    lwe_vec_buffer: *mut u64,
    n_elem: usize,
    desc: TfhersFheIntDescription,
    bitwidth: usize,
) -> i64 {
    assert!(n_elem > 0);
    match bitwidth {
        2 => {
            if n_elem == 1 {
                tfhers_int2_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_int2_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        4 => {
            if n_elem == 1 {
                tfhers_int4_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_int4_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        6 => {
            if n_elem == 1 {
                tfhers_int6_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_int6_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        8 => {
            if n_elem == 1 {
                tfhers_int8_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_int8_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        10 => {
            if n_elem == 1 {
                tfhers_int10_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_int10_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        12 => {
            if n_elem == 1 {
                tfhers_int12_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_int12_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        14 => {
            if n_elem == 1 {
                tfhers_int14_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_int14_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        16 => {
            if n_elem == 1 {
                tfhers_int16_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            } else {
                tfhers_int16_array_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, desc)
            }
        }
        _ => panic!("Unsupported bitwidth"),
    }
}

macro_rules! concrete_cpu_tfhers_int_to_lwe_array {
    ($name:ident, $bitwidth:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            buffer: *const u8,
            buffer_len: usize,
            lwe_vec_buffer: *mut u64,
            n_elem: usize,
            desc: TfhersFheIntDescription,
        ) -> i64 {
            tfhers_int_to_lwe_array(buffer, buffer_len, lwe_vec_buffer, n_elem, desc, $bitwidth)
        }
    };
}

concrete_cpu_tfhers_int_to_lwe_array!(concrete_cpu_tfhers_int2_to_lwe_array, 2);
concrete_cpu_tfhers_int_to_lwe_array!(concrete_cpu_tfhers_int4_to_lwe_array, 4);
concrete_cpu_tfhers_int_to_lwe_array!(concrete_cpu_tfhers_int6_to_lwe_array, 6);
concrete_cpu_tfhers_int_to_lwe_array!(concrete_cpu_tfhers_int8_to_lwe_array, 8);
concrete_cpu_tfhers_int_to_lwe_array!(concrete_cpu_tfhers_int10_to_lwe_array, 10);
concrete_cpu_tfhers_int_to_lwe_array!(concrete_cpu_tfhers_int12_to_lwe_array, 12);
concrete_cpu_tfhers_int_to_lwe_array!(concrete_cpu_tfhers_int14_to_lwe_array, 14);
concrete_cpu_tfhers_int_to_lwe_array!(concrete_cpu_tfhers_int16_to_lwe_array, 16);

unsafe fn lwe_array_to_tfhers_uint(
    lwe_vec_buffer: *const u64,
    buffer: *mut u8,
    buffer_len: usize,
    n_elem: usize,
    desc: TfhersFheIntDescription,
    bitwidth: usize,
) -> usize {
    assert!(n_elem > 0);
    match bitwidth {
        2 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_uint2(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_uint2_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        4 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_uint4(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_uint4_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        6 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_uint6(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_uint6_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        8 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_uint8(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_uint8_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        10 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_uint10(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_uint10_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        12 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_uint12(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_uint12_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        14 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_uint14(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_uint14_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        16 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_uint16(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_uint16_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        _ => panic!("Unsupported bitwidth"),
    }
}

macro_rules! concrete_cpu_lwe_array_to_tfhers_uint {
    ($name:ident, $bitwidth:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            lwe_vec_buffer: *const u64,
            buffer: *mut u8,
            buffer_len: usize,
            n_elem: usize,
            desc: TfhersFheIntDescription,
        ) -> usize {
            lwe_array_to_tfhers_uint(lwe_vec_buffer, buffer, buffer_len, n_elem, desc, $bitwidth)
        }
    };
}

concrete_cpu_lwe_array_to_tfhers_uint!(concrete_cpu_lwe_array_to_tfhers_uint2, 2);
concrete_cpu_lwe_array_to_tfhers_uint!(concrete_cpu_lwe_array_to_tfhers_uint4, 4);
concrete_cpu_lwe_array_to_tfhers_uint!(concrete_cpu_lwe_array_to_tfhers_uint6, 6);
concrete_cpu_lwe_array_to_tfhers_uint!(concrete_cpu_lwe_array_to_tfhers_uint8, 8);
concrete_cpu_lwe_array_to_tfhers_uint!(concrete_cpu_lwe_array_to_tfhers_uint10, 10);
concrete_cpu_lwe_array_to_tfhers_uint!(concrete_cpu_lwe_array_to_tfhers_uint12, 12);
concrete_cpu_lwe_array_to_tfhers_uint!(concrete_cpu_lwe_array_to_tfhers_uint14, 14);
concrete_cpu_lwe_array_to_tfhers_uint!(concrete_cpu_lwe_array_to_tfhers_uint16, 16);

unsafe fn lwe_array_to_tfhers_int(
    lwe_vec_buffer: *const u64,
    buffer: *mut u8,
    buffer_len: usize,
    n_elem: usize,
    desc: TfhersFheIntDescription,
    bitwidth: usize,
) -> usize {
    assert!(n_elem > 0);
    match bitwidth {
        2 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_int2(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_int2_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        4 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_int4(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_int4_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        6 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_int6(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_int6_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        8 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_int8(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_int8_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        10 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_int10(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_int10_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        12 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_int12(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_int12_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        14 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_int14(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_int14_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        16 => {
            if n_elem == 1 {
                lwe_array_to_tfhers_int16(lwe_vec_buffer, buffer, buffer_len, desc)
            } else {
                lwe_array_to_tfhers_int16_array(lwe_vec_buffer, buffer, buffer_len, n_elem, desc)
            }
        }
        _ => panic!("Unsupported bitwidth"),
    }
}

macro_rules! concrete_cpu_lwe_array_to_tfhers_int {
    ($name:ident, $bitwidth:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            lwe_vec_buffer: *const u64,
            buffer: *mut u8,
            buffer_len: usize,
            n_elem: usize,
            desc: TfhersFheIntDescription,
        ) -> usize {
            lwe_array_to_tfhers_int(lwe_vec_buffer, buffer, buffer_len, n_elem, desc, $bitwidth)
        }
    };
}

concrete_cpu_lwe_array_to_tfhers_int!(concrete_cpu_lwe_array_to_tfhers_int2, 2);
concrete_cpu_lwe_array_to_tfhers_int!(concrete_cpu_lwe_array_to_tfhers_int4, 4);
concrete_cpu_lwe_array_to_tfhers_int!(concrete_cpu_lwe_array_to_tfhers_int6, 6);
concrete_cpu_lwe_array_to_tfhers_int!(concrete_cpu_lwe_array_to_tfhers_int8, 8);
concrete_cpu_lwe_array_to_tfhers_int!(concrete_cpu_lwe_array_to_tfhers_int10, 10);
concrete_cpu_lwe_array_to_tfhers_int!(concrete_cpu_lwe_array_to_tfhers_int12, 12);
concrete_cpu_lwe_array_to_tfhers_int!(concrete_cpu_lwe_array_to_tfhers_int14, 14);
concrete_cpu_lwe_array_to_tfhers_int!(concrete_cpu_lwe_array_to_tfhers_int16, 16);

#[no_mangle]
pub extern "C" fn concrete_cpu_tfhers_fheint_buffer_size_u64(
    lwe_size: usize,
    n_cts: usize,
    n_elem: usize,
) -> usize {
    // TODO - that is fragile
    // all FheUint should have the same size, but we use a big one to be safe
    let meta_fheint = core::mem::size_of::<FheUint128>();
    let meta_ct = core::mem::size_of::<Ciphertext>();

    if n_elem <= 1 {
        // FheUint[metadata, ciphertexts[ciphertext[metadata, lwe_buffer] * n_cts]] + headers
        (meta_fheint + (meta_ct + lwe_size * 8/*u64*/) * n_cts) + 201
    } else {
        let meta_vec: usize = core::mem::size_of::<Vec<FheUint128>>();
        // Vec[FheUint[metadata, ciphertexts[ciphertext[metadata, lwe_buffer] * n_cts]] + headers] * n_elem
        meta_vec + ((meta_fheint + (meta_ct + lwe_size * 8/*u64*/) * n_cts) + 201) * n_elem
    }
}
