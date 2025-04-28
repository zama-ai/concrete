use super::{EncryptionKeyChoice, IntegerType};
use crate::ffi::Value;
use crate::utils::from_value::FromValue;
use cxx::UniquePtr;
use tfhe::core_crypto::prelude::LweCiphertext;
use tfhe::integer::ciphertext::{DataKind, Expandable};
use tfhe::shortint::parameters::{Degree, NoiseLevel};
use tfhe::shortint::{CarryModulus, Ciphertext, CiphertextModulus, MessageModulus, PBSOrder, AtomicPatternKind};
use tfhe::{
    FheInt10, FheInt12, FheInt14, FheInt16, FheInt2, FheInt4, FheInt6, FheInt8, FheUint10,
    FheUint12, FheUint14, FheUint16, FheUint2, FheUint4, FheUint6, FheUint8,
};

macro_rules! impl_from_value_integer {
    ($ty:ty, $datakind:expr) => {
        impl FromValue for $ty {
            type Spec = IntegerType;

            fn from_value(s: Self::Spec, v: UniquePtr<Value>) -> Self {
                let lwe_size = s.params.polynomial_size + 1;
                let vals = v.get_tensor::<u64>().unwrap();
                let cts = (0..s.n_cts())
                    .map(|i| {
                        Ciphertext::new(
                            LweCiphertext::from_container(
                                vals.values()[i * lwe_size..(i + 1) * lwe_size].to_vec(),
                                CiphertextModulus::new_native(),
                            ),
                            Degree::new(2u64.pow(s.msg_width as u32) - 1),
                            NoiseLevel::UNKNOWN,
                            MessageModulus(2u64.pow(s.msg_width as u32)),
                            CarryModulus(2u64.pow(s.carry_width as u32)),
                            match s.params.encryption_key_choice {
                                EncryptionKeyChoice::BIG => {
                                    AtomicPatternKind::Standard(PBSOrder::KeyswitchBootstrap)
                                }
                                EncryptionKeyChoice::SMALL => {
                                    AtomicPatternKind::Standard(PBSOrder::BootstrapKeyswitch)
                                }
                            },
                        )
                    })
                    .collect();
                <$ty>::from_expanded_blocks(cts, $datakind).unwrap()
            }
        }
    };
}

impl_from_value_integer!(FheUint2, DataKind::Unsigned(2));
impl_from_value_integer!(FheUint4, DataKind::Unsigned(4));
impl_from_value_integer!(FheUint6, DataKind::Unsigned(6));
impl_from_value_integer!(FheUint8, DataKind::Unsigned(8));
impl_from_value_integer!(FheUint10, DataKind::Unsigned(10));
impl_from_value_integer!(FheUint12, DataKind::Unsigned(12));
impl_from_value_integer!(FheUint14, DataKind::Unsigned(14));
impl_from_value_integer!(FheUint16, DataKind::Unsigned(16));
impl_from_value_integer!(FheInt2, DataKind::Signed(2));
impl_from_value_integer!(FheInt4, DataKind::Signed(4));
impl_from_value_integer!(FheInt6, DataKind::Signed(6));
impl_from_value_integer!(FheInt8, DataKind::Signed(8));
impl_from_value_integer!(FheInt10, DataKind::Signed(10));
impl_from_value_integer!(FheInt12, DataKind::Signed(12));
impl_from_value_integer!(FheInt14, DataKind::Signed(14));
impl_from_value_integer!(FheInt16, DataKind::Signed(16));
