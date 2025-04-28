#![allow(non_camel_case_types, non_snake_case, unused)]
use crate::tfhe::ModuleSpec;
use serde::{Deserialize, Serialize};

/// A complete program can be described by the ensemble of circuit signatures, and the description
/// of the keyset that go with it. This structure regroup those informations.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ProgramInfo {
    /// The informations on the keyset of the program.
    pub keyset: KeysetInfo,
    /// The informations for the different circuits of the program.
    pub circuits: Vec<CircuitInfo>,
    /// The tfhers spec.
    pub tfhers_specs: Option<ModuleSpec>,
}

impl ProgramInfo {
    // Generates a `tfhers_specs` field from the circuits informations.
    //
    // If the `tfhers_specs` field is not available, it means that no tfhers interoperrability is needed.
    // We can generate a dummy `tfhers_specs` field to make further use of the program info object simpler.
    pub fn eventually_patch_tfhers_specs(&mut self) {
        if self.tfhers_specs.is_none() {
            self.tfhers_specs = Some(ModuleSpec {
                input_types_per_func: self
                    .circuits
                    .iter()
                    .map(|c| (c.name.clone(), vec![None; c.inputs.len()]))
                    .collect(),
                output_types_per_func: self
                    .circuits
                    .iter()
                    .map(|c| (c.name.clone(), vec![None; c.outputs.len()]))
                    .collect(),
                input_shapes_per_func: self
                    .circuits
                    .iter()
                    .map(|c| (c.name.clone(), vec![None; c.inputs.len()]))
                    .collect(),
                output_shapes_per_func: self
                    .circuits
                    .iter()
                    .map(|c| (c.name.clone(), vec![None; c.outputs.len()]))
                    .collect(),
            });
        }
    }
}

/// A circuit signature can be described completely by the type informations for its input and
/// outputs, as well as its name. This structure regroup those informations.
///
/// Note:
///   The order of the input and output lists matters. The order of values should be the same when
///   executing the circuit. Also, the name is expected to be unique in the program.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct CircuitInfo {
    /// The ordered list of input types.
    pub inputs: Vec<GateInfo>,
    /// The ordered list of output types.
    pub outputs: Vec<GateInfo>,
    /// The name of the circuit.
    pub name: String,
}

/// A value flowing in or out of a circuit is expected to be of a given type, according to the
/// signature of this circuit. This structure represents such a type in a circuit signature.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct GateInfo {
    /// The raw information that raw data must be possible to parse with.
    pub rawInfo: RawInfo,
    /// The type of the value expected at the gate.
    pub typeInfo: TypeInfo,
}

/// A value exchanged at the boundary between two parties of a computation will be transmitted as a
/// binary payload containing a tensor of integers. This payload will first have to be parsed to a
/// tensor of proper shape, signedness and precision before being pre-processed and passed to the
/// computation. This structure represents the informations needed to parse this payload into the
/// expected tensor.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RawInfo {
    /// The shape of the tensor.
    pub shape: Shape,
    /// The precision of the integers.
    pub integerPrecision: u32,
    /// The signedness of the integers.
    pub isSigned: bool,
}

/// Scalar and tensor values are represented by the same types. This structure contains a
/// description of the shape of value.
///
/// Note:
///   If the dimensions vector is empty, the message is interpreted as a scalar.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Shape {
    /// The dimensions of the value.
    pub dimensions: Vec<u32>,
}

/// The different possible type of values.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum TypeInfo {
    lweCiphertext(LweCiphertextTypeInfo),
    plaintext(PlaintextTypeInfo),
    index(IndexTypeInfo),
}

/// A plaintext value can flow in and out of a circuit. This structure represents the informations
/// needed to verify and pre-or-post process this value.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PlaintextTypeInfo {
    /// The shape of the value.
    pub shape: Shape,
    /// The precision of the integers.
    pub integerPrecision: u32,
    /// The signedness of the integers.
    pub isSigned: bool,
}

/// A plaintext value can flow in and out of a circuit. This structure represents the informations
/// needed to verify and pre-or-post process this value.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct IndexTypeInfo {
    /// The shape of the value.
    pub shape: Shape,
    /// The precision of the indexes.
    pub integerPrecision: u32,
    /// The signedness of the indexes.
    pub isSigned: bool,
}

/// A ciphertext value can flow in and out of a circuit. This structure represents the informations
/// needed to verify and pre-or-post process this value.
///
/// Note:
///   Two shape information are carried in this type. The abstract shape is the shape the tensor
///   would have if the values were cleartext. That is, it does not take into account the encryption
///   process. The concrete shape is the final shape of the object accounting for the encryption,
///   that usually add one or more dimension to the object.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LweCiphertextTypeInfo {
    /// The abstract shape of the value.
    pub abstractShape: Shape,
    /// The concrete shape of the value.
    pub concreteShape: Shape,
    /// The precision of the integers used for storage.
    pub integerPrecision: u32,
    /// The informations relative to the encryption.
    pub encryption: LweCiphertextEncryptionInfo,
    /// The compression used for this value.
    pub compression: Compression,
    /// The encoding of the value stored inside the ciphertext.
    pub encoding: LweCiphretextTypeInfo_Encoding,
}

/// The encryption of a cleartext value requires some parameters to operate. This structure
/// represents those parameters.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LweCiphertextEncryptionInfo {
    /// The identifier of the secret key used to perform the encryption.
    pub keyId: u32,
    /// The variance of the noise injected during encryption.
    pub variance: f64,
    /// The lwe dimension of the ciphertext.
    pub lweDimension: u32,
}

/// Evaluation keys and ciphertexts can be compressed when transported over the wire. This
/// enumeration encodes the different compressions that can be used to compress scheme objects.
///
/// Note:
///   Not all compressions are available for every types of evaluation keys or ciphertexts.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum Compression {
    none,
    seed,
    paillier,
}

/// The encoding of the value stored inside the ciphertext.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum LweCiphretextTypeInfo_Encoding {
    integer(IntegerCiphertextEncodingInfo),
    boolean(BooleanCiphertextEncodingInfo),
}

/// A ciphertext can be used to represent an integer value. This structure represents the
/// informations needed to encode such an integer.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct IntegerCiphertextEncodingInfo {
    /// The bitwidth of the encoded integer.
    pub width: u32,
    /// The signedness of the encoded integer.
    pub isSigned: bool,
    /// The mode used to encode the integer.
    pub mode: IntegerCiphertextEncodingInfo_Mode,
}

/// The mode used to encode the integer.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum IntegerCiphertextEncodingInfo_Mode {
    native(IntegerCiphertextEncodingInfo_Mode_NativeMode),
    chunked(IntegerCiphertextEncodingInfo_Mode_ChunkedMode),
    crt(IntegerCiphertextEncodingInfo_Mode_CrtMode),
}

/// An integer of width from 1 to 8 bits can be encoded in a single ciphertext natively, by
/// being shifted in the most significant bits. This structure represents this integer encoding
/// mode.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct IntegerCiphertextEncodingInfo_Mode_NativeMode {}

/// An integer of width from 1 to n can be encoded in a set of ciphertexts by chunking the bits
/// of the original integer. This structure represents this integer encoding mode.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct IntegerCiphertextEncodingInfo_Mode_ChunkedMode {
    /// The number of chunks to be used.
    pub size: u32,
    /// The number of bits encoded by each chunks.
    pub width: u32,
}

/// An integer of width 1 to 16 can be encoded in a set of ciphertexts, by decomposing a value
/// using a set of pairwise coprimes. This structure represents this integer encoding mode.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct IntegerCiphertextEncodingInfo_Mode_CrtMode {
    /// The coprimes used to decompose the original value.
    pub moduli: Vec<u32>,
}

/// A ciphertext can be used to represent a boolean value. This structure represents such an
/// encoding.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct BooleanCiphertextEncodingInfo {}

/// Secret Keys can be drawn from different ranges of values, using different distributions. This
/// enumeration encodes the different supported ways.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum KeyType {
    binary = 0,
    ternary = 1,
}

/// Ciphertext operations are performed using modular arithmetic. Depending on the use, different
/// modulus can be used for the operations. This structure encodes the different supported ways.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Modulus {
    /// The modulus expected to be used.
    pub modulus: Modulus_enum,
}

/// The modulus expected to be used.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum Modulus_enum {
    native(NativeModulus),
    powerOfTwo(PowerOfTwoModulus),
    integer(IntegerModulus),
}

/// Operations are performed using the modulus of the integers used to store the ciphertexts.
///
/// Note:
///   The bitwidth of the integer storage is represented implicitly here, and must be grabbed from
///   the rest of the description.
///
/// Example:
///   2^64 when the ciphertext is stored using 64 bits integers.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct NativeModulus {}

/// Operations are performed using a modulus that is a power of two.
///
/// Example:
///   2^n for any n between 0 and the bitwidth of the integer used to store the ciphertext.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PowerOfTwoModulus {
    /// The power used to raise 2.
    pub power: u32,
}

/// Operations are performed using a modulus that is an arbitrary integer.
///
/// Example:
///   n for any n between 0 and 2^N where N is the bitwidth of the integer used to store the
///   ciphertext.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct IntegerModulus {
    /// The value used as modulus.
    pub modulus: u32,
}

/// A secret key value is uniquely described by cryptographic parameters and an identifier. This
/// structure represents this description of a secret key.
///
/// Note:
///   Secret keys with same parameters are allowed to co-exist in a program, as long as they
///   have different ids.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LweSecretKeyInfo {
    /// The identifier of the key.
    pub id: u32,
    /// The cryptographic parameters of the keys.
    pub params: LweSecretKeyParams,
}

/// A secret key is parameterized by a few quantities of cryptographic importance. This structure
/// represents those parameters.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LweSecretKeyParams {
    /// The LWE dimension, e.g. the length of the key.
    pub lweDimension: u32,
    /// The bitwidth of the integers used for storage.
    pub integerPrecision: u32,
    /// The kind of distribution used to sample the key.
    pub keyType: KeyType,
}

/// A keyswitch key value is uniquely described by cryptographic parameters and a few application
/// related quantities. This structure represents this description of a keyswitch key.
///
/// Note:
///   Keyswitch keys with same parameters, compression, input and output id, are allowed to co-exist
///   in a program as long as they have different ids.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LweKeyswitchKeyInfo {
    /// The identifier of the keyswitch key.
    pub id: u32,
    /// The identifier of the input secret key.
    pub inputId: u32,
    /// The identifier of the output secret key.
    pub outputId: u32,
    /// The cryptographic parameters of the key.
    pub params: LweKeyswitchKeyParams,
    /// The compression used to store the key.
    pub compression: Compression,
}

/// A keyswitch key is parameterized by a few quantities of cryptographic importance. This structure
/// represents those parameters.
///
/// Note:
///   For now, only keys with the same input and output key types can be represented.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LweKeyswitchKeyParams {
    /// The number of levels of the ciphertexts.
    pub levelCount: u32,
    /// The logarithm of the base of ciphertexts.
    pub baseLog: u32,
    /// The variance used to encrypt the ciphertexts.
    pub variance: f64,
    /// The bitwidth of the integers used to store the ciphertexts.
    pub integerPrecision: u32,
    /// The dimension of the input secret key.
    pub inputLweDimension: u32,
    /// The dimension of the output secret key.
    pub outputLweDimension: u32,
    /// The modulus used to perform operations with this key.
    pub modulus: Modulus,
    /// The distribution of the input and output secret keys.
    pub keyType: KeyType,
}

/// A packing keyswitch key value is uniquely described by cryptographic parameters and a few
/// application related quantities. This structure represents this description of a packing
/// keyswitch key.
///
/// Note:
///   Packing keyswitch keys with same parameters, compression, input and output id, are allowed to
///   co-exist in a program as long as they have different ids.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PackingKeyswitchKeyInfo {
    /// The identifier of the packing keyswitch key.
    pub id: u32,
    /// The identifier of the input secret key.
    pub inputId: u32,
    /// The identifier of the output secret key.
    pub outputId: u32,
    /// The cryptographic parameters of the key.
    pub params: PackingKeyswitchKeyParams,
    /// The compression used to store the key.
    pub compression: Compression,
}

/// A packing keyswitch key is parameterized by a few quantities of cryptographic importance. This
/// structure represents those parameters.
///
/// Note:
///   For now, only keys with the same input and output key types can be represented.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PackingKeyswitchKeyParams {
    /// The number of levels of the ciphertexts.
    pub levelCount: u32,
    /// The logarithm of the base of the ciphertexts.
    pub baseLog: u32,
    /// The glwe dimension of the ciphertexts.
    pub glweDimension: u32,
    /// The polynomial size of the ciphertexts.
    pub polynomialSize: u32,
    /// The input lwe dimension.
    pub inputLweDimension: u32,
    /// The intermediate lwe dimension.
    pub innerLweDimension: u32,
    /// The variance used to encrypt the ciphertexts.
    pub variance: f64,
    /// The bitwidth of the integers used to store the ciphertexts.
    pub integerPrecision: u32,
    /// The modulus used to perform operations with this key.
    pub modulus: Modulus,
    /// The distribution of the input and output secret keys.
    pub keyType: KeyType,
}

/// A bootstrap key value is uniquely described by cryptographic parameters and a few application
/// related quantities. This structure represents this description of a bootstrap key.
///
/// Note:
///   Bootstrap keys with same parameters, compression, input and output id, are allowed to co-exist
///   in a program as long as they have different ids.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LweBootstrapKeyInfo {
    /// The identifier of the bootstrap key.
    pub id: u32,
    /// The identifier of the input secret key.
    pub inputId: u32,
    /// The identifier of the output secret key.
    pub outputId: u32,
    /// The cryptographic parameters of the key.
    pub params: LweBootstrapKeyParams,
    /// The compression used to store the key.
    pub compression: Compression,
}

/// A bootstrap key is parameterized by a few quantities of cryptographic importance. This structure
/// represents those parameters.
///
/// Note:
///   For now, only keys with the same input and output key types can be represented.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LweBootstrapKeyParams {
    /// The number of levels of the ciphertexts.
    pub levelCount: u32,
    /// The logarithm of the base of the ciphertext.
    pub baseLog: u32,
    /// The dimension of the ciphertexts.
    pub glweDimension: u32,
    /// The polynomial size of the ciphertexts.
    pub polynomialSize: u32,
    /// The dimension of the input lwe secret key.
    pub inputLweDimension: u32,
    /// The variance used to encrypt the ciphertexts.
    pub variance: f64,
    /// The bitwidth of the integers used to store the ciphertexts.
    pub integerPrecision: u32,
    /// The modulus used to perform operations with this key.
    pub modulus: Modulus,
    /// The distribution of the input and output secret keys.
    pub keyType: KeyType,
}

/// The keyset needed for an application can be described by an ensemble of descriptions of the
/// different keys used in the program. This structure represents such a description.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct KeysetInfo {
    /// The secret key descriptions.
    pub lweSecretKeys: Vec<LweSecretKeyInfo>,
    /// The bootstrap key descriptions
    pub lweBootstrapKeys: Vec<LweBootstrapKeyInfo>,
    /// The keyswitch key descriptions.
    pub lweKeyswitchKeys: Vec<LweKeyswitchKeyInfo>,
    /// The packing keyswitch key descriptions.
    pub packingKeyswitchKeys: Vec<PackingKeyswitchKeyInfo>,
}

#[cfg(feature = "compiler")]
mod to_tokens {
    //! This module contains `ToTokens` implementations for the protocol types. This allows protocol
    //! values to be interpolated in the `quote!` macro as constructors of the values.
    //! Useful to construct static protocol values.

    use super::*;
    use proc_macro2::TokenStream;
    use quote::{quote, ToTokens};

    impl ToTokens for ProgramInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let keyset = &self.keyset;
            let circuits = &self
                .circuits
                .iter()
                .map(|circuit| quote! { #circuit })
                .collect::<Vec<_>>();
            let tfhers_specs = match &self.tfhers_specs {
                Some(s) => quote! {Some(#s)},
                None => quote! {None},
            };
            tokens.extend(quote! {
                ::concrete::protocol::ProgramInfo {
                    keyset: #keyset,
                    circuits: vec![#(#circuits),*],
                    tfhers_specs: #tfhers_specs
                }
            });
        }
    }

    impl ToTokens for CircuitInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let inputs = self
                .inputs
                .iter()
                .map(|input| quote! { #input })
                .collect::<Vec<_>>();
            let outputs = self
                .outputs
                .iter()
                .map(|output| quote! { #output })
                .collect::<Vec<_>>();
            let name = &self.name;
            tokens.extend(quote! {
                ::concrete::protocol::CircuitInfo {
                    inputs: vec![#(#inputs),*],
                    outputs: vec![#(#outputs),*],
                    name: String::from(#name),
                }
            });
        }
    }
    impl ToTokens for GateInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let raw_info = &self.rawInfo;
            let type_info = &self.typeInfo;
            tokens.extend(quote! {
                ::concrete::protocol::GateInfo {
                    rawInfo: #raw_info,
                    typeInfo: #type_info,
                }
            });
        }
    }

    impl ToTokens for RawInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let shape = &self.shape;
            let integer_precision = &self.integerPrecision;
            let is_signed = &self.isSigned;
            tokens.extend(quote! {
                ::concrete::protocol::RawInfo {
                    shape: #shape,
                    integerPrecision: #integer_precision,
                    isSigned: #is_signed,
                }
            });
        }
    }

    impl ToTokens for Shape {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let dimensions = self
                .dimensions
                .iter()
                .map(|dim| quote! { #dim })
                .collect::<Vec<_>>();
            tokens.extend(quote! {
                ::concrete::protocol::Shape {
                    dimensions: vec![#(#dimensions),*],
                }
            });
        }
    }

    impl ToTokens for TypeInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                TypeInfo::lweCiphertext(info) => {
                    tokens.extend(quote! { ::concrete::protocol::TypeInfo::lweCiphertext(#info) })
                }
                TypeInfo::plaintext(info) => {
                    tokens.extend(quote! { ::concrete::protocol::TypeInfo::plaintext(#info) })
                }
                TypeInfo::index(info) => {
                    tokens.extend(quote! { ::concrete::procotol::TypeInfo::index(#info) })
                }
            }
        }
    }

    impl ToTokens for PlaintextTypeInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let shape = &self.shape;
            let integer_precision = &self.integerPrecision;
            let is_signed = &self.isSigned;
            tokens.extend(quote! {
                ::concrete::protocol::PlaintextTypeInfo {
                    shape: #shape,
                    integerPrecision: #integer_precision,
                    isSigned: #is_signed,
                }
            });
        }
    }

    impl ToTokens for IndexTypeInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let shape = &self.shape;
            let integer_precision = &self.integerPrecision;
            let is_signed = &self.isSigned;
            tokens.extend(quote! {
                ::concrete::protocol::IndexTypeInfo {
                    shape: #shape,
                    integerPrecision: #integer_precision,
                    isSigned: #is_signed,
                }
            });
        }
    }

    impl ToTokens for LweCiphertextTypeInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let abstract_shape = &self.abstractShape;
            let concrete_shape = &self.concreteShape;
            let integer_precision = &self.integerPrecision;
            let encryption = &self.encryption;
            let compression = &self.compression;
            let encoding = &self.encoding;
            tokens.extend(quote! {
                ::concrete::protocol::LweCiphertextTypeInfo {
                    abstractShape: #abstract_shape,
                    concreteShape: #concrete_shape,
                    integerPrecision: #integer_precision,
                    encryption: #encryption,
                    compression: #compression,
                    encoding: #encoding,
                }
            });
        }
    }

    impl ToTokens for LweCiphertextEncryptionInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let key_id = &self.keyId;
            let variance = &self.variance;
            let lwe_dimension = &self.lweDimension;
            tokens.extend(quote! {
                ::concrete::protocol::LweCiphertextEncryptionInfo {
                    keyId: #key_id,
                    variance: #variance,
                    lweDimension: #lwe_dimension,
                }
            });
        }
    }

    impl ToTokens for Compression {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                Compression::none => {
                    tokens.extend(quote! { ::concrete::protocol::Compression::none })
                }
                Compression::seed => {
                    tokens.extend(quote! { ::concrete::protocol::Compression::seed })
                }
                Compression::paillier => {
                    tokens.extend(quote! { ::concrete::protocol::Compression::paillier })
                }
            }
        }
    }

    impl ToTokens for LweCiphretextTypeInfo_Encoding {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                LweCiphretextTypeInfo_Encoding::integer(info) => tokens.extend(
                    quote! { ::concrete::protocol::LweCiphretextTypeInfo_Encoding::integer(#info) },
                ),
                LweCiphretextTypeInfo_Encoding::boolean(info) => tokens.extend(
                    quote! { ::concrete::protocol::LweCiphretextTypeInfo_Encoding::boolean(#info) },
                ),
            }
        }
    }

    impl ToTokens for IntegerCiphertextEncodingInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let width = &self.width;
            let is_signed = &self.isSigned;
            let mode = &self.mode;
            tokens.extend(quote! {
                ::concrete::protocol::IntegerCiphertextEncodingInfo {
                    width: #width,
                    isSigned: #is_signed,
                    mode: #mode,
                }
            });
        }
    }

    impl ToTokens for IntegerCiphertextEncodingInfo_Mode {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                IntegerCiphertextEncodingInfo_Mode::native(info) => tokens.extend(quote! { ::concrete::protocol::IntegerCiphertextEncodingInfo_Mode::native(#info) }),
                IntegerCiphertextEncodingInfo_Mode::chunked(info) => tokens.extend(quote! { ::concrete::protocol::IntegerCiphertextEncodingInfo_Mode::chunked(#info) }),
                IntegerCiphertextEncodingInfo_Mode::crt(info) => tokens.extend(quote! { ::concrete::protocol::IntegerCiphertextEncodingInfo_Mode::crt(#info) }),
            }
        }
    }

    impl ToTokens for IntegerCiphertextEncodingInfo_Mode_NativeMode {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.extend(quote! {
                ::concrete::protocol::IntegerCiphertextEncodingInfo_Mode_NativeMode {}
            });
        }
    }

    impl ToTokens for IntegerCiphertextEncodingInfo_Mode_ChunkedMode {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let size = &self.size;
            let width = &self.width;
            tokens.extend(quote! {
                ::concrete::protocol::IntegerCiphertextEncodingInfo_Mode_ChunkedMode {
                    size: #size,
                    width: #width,
                }
            });
        }
    }

    impl ToTokens for IntegerCiphertextEncodingInfo_Mode_CrtMode {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let moduli = self
                .moduli
                .iter()
                .map(|modulus| quote! { #modulus })
                .collect::<Vec<_>>();
            tokens.extend(quote! {
                ::concrete::protocol::IntegerCiphertextEncodingInfo_Mode_CrtMode {
                    moduli: vec![#(#moduli),*],
                }
            });
        }
    }
    impl ToTokens for BooleanCiphertextEncodingInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.extend(quote! {
                ::concrete::protocol::BooleanCiphertextEncodingInfo {}
            });
        }
    }

    impl ToTokens for KeyType {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                KeyType::binary => tokens.extend(quote! { ::concrete::protocol::KeyType::binary }),
                KeyType::ternary => {
                    tokens.extend(quote! { ::concrete::protocol::KeyType::ternary })
                }
            }
        }
    }

    impl ToTokens for Modulus {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let modulus = &self.modulus;
            tokens.extend(quote! {
                ::concrete::protocol::Modulus {
                    modulus: #modulus,
                }
            });
        }
    }

    impl ToTokens for Modulus_enum {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                Modulus_enum::native(info) => {
                    tokens.extend(quote! { ::concrete::protocol::Modulus_enum::native(#info) })
                }
                Modulus_enum::powerOfTwo(info) => {
                    tokens.extend(quote! { ::concrete::protocol::Modulus_enum::powerOfTwo(#info) })
                }
                Modulus_enum::integer(info) => {
                    tokens.extend(quote! { ::concrete::protocol::Modulus_enum::integer(#info) })
                }
            }
        }
    }

    impl ToTokens for NativeModulus {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.extend(quote! {
                ::concrete::protocol::NativeModulus {}
            });
        }
    }

    impl ToTokens for PowerOfTwoModulus {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let power = &self.power;
            tokens.extend(quote! {
                ::concrete::protocol::PowerOfTwoModulus {
                    power: #power,
                }
            });
        }
    }

    impl ToTokens for IntegerModulus {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let modulus = &self.modulus;
            tokens.extend(quote! {
                ::concrete::protocol::IntegerModulus {
                    modulus: #modulus,
                }
            });
        }
    }

    impl ToTokens for LweSecretKeyInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let id = &self.id;
            let params = &self.params;
            tokens.extend(quote! {
                ::concrete::protocol::LweSecretKeyInfo {
                    id: #id,
                    params: #params,
                }
            });
        }
    }

    impl ToTokens for LweSecretKeyParams {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let lwe_dimension = &self.lweDimension;
            let integer_precision = &self.integerPrecision;
            let key_type = &self.keyType;
            tokens.extend(quote! {
                ::concrete::protocol::LweSecretKeyParams {
                    lweDimension: #lwe_dimension,
                    integerPrecision: #integer_precision,
                    keyType: #key_type,
                }
            });
        }
    }

    impl ToTokens for LweKeyswitchKeyInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let id = &self.id;
            let input_id = &self.inputId;
            let output_id = &self.outputId;
            let params = &self.params;
            let compression = &self.compression;
            tokens.extend(quote! {
                ::concrete::protocol::LweKeyswitchKeyInfo {
                    id: #id,
                    inputId: #input_id,
                    outputId: #output_id,
                    params: #params,
                    compression: #compression,
                }
            });
        }
    }

    impl ToTokens for LweKeyswitchKeyParams {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let level_count = &self.levelCount;
            let base_log = &self.baseLog;
            let variance = &self.variance;
            let integer_precision = &self.integerPrecision;
            let input_lwe_dimension = &self.inputLweDimension;
            let output_lwe_dimension = &self.outputLweDimension;
            let modulus = &self.modulus;
            let key_type = &self.keyType;
            tokens.extend(quote! {
                ::concrete::protocol::LweKeyswitchKeyParams {
                    levelCount: #level_count,
                    baseLog: #base_log,
                    variance: #variance,
                    integerPrecision: #integer_precision,
                    inputLweDimension: #input_lwe_dimension,
                    outputLweDimension: #output_lwe_dimension,
                    modulus: #modulus,
                    keyType: #key_type,
                }
            });
        }
    }

    impl ToTokens for PackingKeyswitchKeyInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let id = &self.id;
            let input_id = &self.inputId;
            let output_id = &self.outputId;
            let params = &self.params;
            let compression = &self.compression;
            tokens.extend(quote! {
                ::concrete::protocol::PackingKeyswitchKeyInfo {
                    id: #id,
                    inputId: #input_id,
                    outputId: #output_id,
                    params: #params,
                    compression: #compression,
                }
            });
        }
    }

    impl ToTokens for PackingKeyswitchKeyParams {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let level_count = &self.levelCount;
            let base_log = &self.baseLog;
            let glwe_dimension = &self.glweDimension;
            let polynomial_size = &self.polynomialSize;
            let input_lwe_dimension = &self.inputLweDimension;
            let inner_lwe_dimension = &self.innerLweDimension;
            let variance = &self.variance;
            let integer_precision = &self.integerPrecision;
            let modulus = &self.modulus;
            let key_type = &self.keyType;
            tokens.extend(quote! {
                ::concrete::protocol::PackingKeyswitchKeyParams {
                    levelCount: #level_count,
                    baseLog: #base_log,
                    glweDimension: #glwe_dimension,
                    polynomialSize: #polynomial_size,
                    inputLweDimension: #input_lwe_dimension,
                    innerLweDimension: #inner_lwe_dimension,
                    variance: #variance,
                    integerPrecision: #integer_precision,
                    modulus: #modulus,
                    keyType: #key_type,
                }
            });
        }
    }

    impl ToTokens for LweBootstrapKeyInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let id = &self.id;
            let input_id = &self.inputId;
            let output_id = &self.outputId;
            let params = &self.params;
            let compression = &self.compression;
            tokens.extend(quote! {
                ::concrete::protocol::LweBootstrapKeyInfo {
                    id: #id,
                    inputId: #input_id,
                    outputId: #output_id,
                    params: #params,
                    compression: #compression,
                }
            });
        }
    }

    impl ToTokens for LweBootstrapKeyParams {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let level_count = &self.levelCount;
            let base_log = &self.baseLog;
            let glwe_dimension = &self.glweDimension;
            let polynomial_size = &self.polynomialSize;
            let input_lwe_dimension = &self.inputLweDimension;
            let variance = &self.variance;
            let integer_precision = &self.integerPrecision;
            let modulus = &self.modulus;
            let key_type = &self.keyType;
            tokens.extend(quote! {
                ::concrete::protocol::LweBootstrapKeyParams {
                    levelCount: #level_count,
                    baseLog: #base_log,
                    glweDimension: #glwe_dimension,
                    polynomialSize: #polynomial_size,
                    inputLweDimension: #input_lwe_dimension,
                    variance: #variance,
                    integerPrecision: #integer_precision,
                    modulus: #modulus,
                    keyType: #key_type,
                }
            });
        }
    }

    impl ToTokens for KeysetInfo {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let lwe_secret_keys = self
                .lweSecretKeys
                .iter()
                .map(|key| quote! { #key })
                .collect::<Vec<_>>();
            let lwe_bootstrap_keys = self
                .lweBootstrapKeys
                .iter()
                .map(|key| quote! { #key })
                .collect::<Vec<_>>();
            let lwe_keyswitch_keys = self
                .lweKeyswitchKeys
                .iter()
                .map(|key| quote! { #key })
                .collect::<Vec<_>>();
            let packing_keyswitch_keys = self
                .packingKeyswitchKeys
                .iter()
                .map(|key| quote! { #key })
                .collect::<Vec<_>>();
            tokens.extend(quote! {
                ::concrete::protocol::KeysetInfo {
                    lweSecretKeys: vec![#(#lwe_secret_keys),*],
                    lweBootstrapKeys: vec![#(#lwe_bootstrap_keys),*],
                    lweKeyswitchKeys: vec![#(#lwe_keyswitch_keys),*],
                    packingKeyswitchKeys: vec![#(#packing_keyswitch_keys),*],
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_program_info() {
        let string = r#"
            {"keyset": {"lweSecretKeys": [{"id": 0, "params": {"lweDimension": 2048, "integerPrecision": 64, "keyType": "binary"}}, {"id": 1, "params": {"lweDimension": 4096, "integerPrecision": 64, "keyType": "binary"}}, {"id": 2, "params": {"lweDimension": 776, "integerPrecision": 64, "keyType": "binary"}}, {"id": 3, "params": {"lweDimension": 626, "integerPrecision": 64, "keyType": "binary"}}, {"id": 4, "params": {"lweDimension": 2048, "integerPrecision": 64, "keyType": "binary"}}], "lweBootstrapKeys": [{"id": 0, "inputId": 2, "outputId": 0, "params": {"levelCount": 2, "baseLog": 15, "glweDimension": 2, "polynomialSize": 1024, "variance": 8.442253112932959e-31, "integerPrecision": 64, "modulus": {"modulus": {"native": {}}}, "keyType": "binary", "inputLweDimension": 776}, "compression": "none"}, {"id": 1, "inputId": 3, "outputId": 4, "params": {"levelCount": 11, "baseLog": 4, "glweDimension": 4, "polynomialSize": 512, "variance": 8.442253112932959e-31, "integerPrecision": 64, "modulus": {"modulus": {"native": {}}}, "keyType": "binary", "inputLweDimension": 626}, "compression": "none"}], "lweKeyswitchKeys": [{"id": 0, "inputId": 1, "outputId": 2, "params": {"levelCount": 5, "baseLog": 3, "variance": 4.0324907628621766e-11, "integerPrecision": 64, "modulus": {"modulus": {"native": {}}}, "keyType": "binary", "inputLweDimension": 4096, "outputLweDimension": 776}, "compression": "none"}, {"id": 1, "inputId": 0, "outputId": 1, "params": {"levelCount": 1, "baseLog": 31, "variance": 4.70197740328915e-38, "integerPrecision": 64, "modulus": {"modulus": {"native": {}}}, "keyType": "binary", "inputLweDimension": 2048, "outputLweDimension": 4096}, "compression": "none"}, {"id": 2, "inputId": 1, "outputId": 3, "params": {"levelCount": 5, "baseLog": 2, "variance": 8.437693323536307e-09, "integerPrecision": 64, "modulus": {"modulus": {"native": {}}}, "keyType": "binary", "inputLweDimension": 4096, "outputLweDimension": 626}, "compression": "none"}, {"id": 3, "inputId": 4, "outputId": 1, "params": {"levelCount": 2, "baseLog": 21, "variance": 4.70197740328915e-38, "integerPrecision": 64, "modulus": {"modulus": {"native": {}}}, "keyType": "binary", "inputLweDimension": 2048, "outputLweDimension": 4096}, "compression": "none"}], "packingKeyswitchKeys": []}, "circuits": [{"inputs": [{"rawInfo": {"shape": {"dimensions": [8, 4097]}, "integerPrecision": 64, "isSigned": false}, "typeInfo": {"lweCiphertext": {"abstractShape": {"dimensions": [8]}, "concreteShape": {"dimensions": [8, 4097]}, "integerPrecision": 64, "encryption": {"keyId": 1, "variance": 4.70197740328915e-38, "lweDimension": 4096, "modulus": {"modulus": {"native": {}}}}, "compression": "none", "encoding": {"integer": {"width": 4, "isSigned": false, "mode": {"native": {}}}}}}}, {"rawInfo": {"shape": {"dimensions": [8, 4097]}, "integerPrecision": 64, "isSigned": false}, "typeInfo": {"lweCiphertext": {"abstractShape": {"dimensions": [8]}, "concreteShape": {"dimensions": [8, 4097]}, "integerPrecision": 64, "encryption": {"keyId": 1, "variance": 4.70197740328915e-38, "lweDimension": 4096, "modulus": {"modulus": {"native": {}}}}, "compression": "none", "encoding": {"integer": {"width": 4, "isSigned": false, "mode": {"native": {}}}}}}}], "outputs": [{"rawInfo": {"shape": {"dimensions": [8, 4097]}, "integerPrecision": 64, "isSigned": false}, "typeInfo": {"lweCiphertext": {"abstractShape": {"dimensions": [8]}, "concreteShape": {"dimensions": [8, 4097]}, "integerPrecision": 64, "encryption": {"keyId": 1, "variance": 4.70197740328915e-38, "lweDimension": 4096, "modulus": {"modulus": {"native": {}}}}, "compression": "none", "encoding": {"integer": {"width": 4, "isSigned": false, "mode": {"native": {}}}}}}}], "name": "my_func"}], "tfhers_specs": {"input_types_per_func": {"my_func": [{"is_signed": false, "bit_width": 16, "carry_width": 2, "msg_width": 2, "params": {"lwe_dimension": 909, "glwe_dimension": 1, "polynomial_size": 4096, "pbs_base_log": 15, "pbs_level": 2, "lwe_noise_distribution": 0, "glwe_noise_distribution": 2.168404344971009e-19, "encryption_key_choice": 0}}, {"is_signed": false, "bit_width": 16, "carry_width": 2, "msg_width": 2, "params": {"lwe_dimension": 909, "glwe_dimension": 1, "polynomial_size": 4096, "pbs_base_log": 15, "pbs_level": 2, "lwe_noise_distribution": 0, "glwe_noise_distribution": 2.168404344971009e-19, "encryption_key_choice": 0}}]}, "output_types_per_func": {"my_func": [{"is_signed": false, "bit_width": 16, "carry_width": 2, "msg_width": 2, "params": {"lwe_dimension": 909, "glwe_dimension": 1, "polynomial_size": 4096, "pbs_base_log": 15, "pbs_level": 2, "lwe_noise_distribution": 0, "glwe_noise_distribution": 2.168404344971009e-19, "encryption_key_choice": 0}}]}, "input_shapes_per_func": {"my_func": [[], []]}, "output_shapes_per_func": {"my_func": [[]]}}}
        "#;
        let val: ProgramInfo = serde_json::from_str(string).unwrap();
    }
}
