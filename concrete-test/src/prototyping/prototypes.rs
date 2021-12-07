//! A module containing prototypical entities for different precisions.
use crate::prototyping::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::*;

macro_rules! prototype_traits {
    ($({$kind:literal, $name:ident}),+) => {
        $(
            #[doc = "A trait implemented by a "]
            #[doc = $kind]
            #[doc = " prototype."]
            pub trait $name {
                type Precision: IntegerPrecision;
            }
        )+
    };
}

prototype_traits! {
    {"cleartext", CleartextPrototype},
    {"cleartext vector", CleartextVectorPrototype},
    {"plaintext", PlaintextPrototype },
    {"plaintext vector", PlaintextVectorPrototype},
    {"binary lwe secret key", BinaryLweSecretKeyPrototype},
    {"binary lwe ciphertext", BinaryLweCiphertextPrototype },
    {"binary lwe ciphertext vector", BinaryLweCiphertextVectorPrototype },
    {"binary glwe secret key", BinaryGlweSecretKeyPrototype },
    {"binary glwe ciphertext", BinaryGlweCiphertextPrototype},
    {"binary glwe ciphertext vector", BinaryGlweCiphertextVectorPrototype},
    {"binary to binary lwe keyswitch key", BinaryBinaryLweKeyswitchKeyPrototype },
    {"binary to binary lwe bootstrap key", BinaryBinaryLweBootstrapKeyPrototype}
}

macro_rules! implement_prototype_traits {
    ($({$kind: literal, $name:ident, $original: ident, $prototrait:ident, $precision:ident}),*) => {
        $(
            #[doc = "A type representing the prototype of a "]
            #[doc = $kind]
            #[doc = " entity."]
            pub struct $name(pub(crate) $original);
            impl $prototrait for $name {
                type Precision = $precision;
            }
        )*
    };
}

implement_prototype_traits! {
    {"32 bit cleartext", ProtoCleartext32, Cleartext32, CleartextPrototype, Precision32},
    {"64 bit cleartext", ProtoCleartext64, Cleartext64, CleartextPrototype, Precision64},
    {"32 bit cleartext vector", ProtoCleartextVector32, CleartextVector32, CleartextVectorPrototype, Precision32},
    {"64 bit cleartext vector", ProtoCleartextVector64, CleartextVector64, CleartextVectorPrototype, Precision64},
    {"32 bit plaintext", ProtoPlaintext32, Plaintext32, PlaintextPrototype, Precision32},
    {"64 bit plaintext", ProtoPlaintext64, Plaintext64, PlaintextPrototype, Precision64},
    {"32 bit plaintext vector", ProtoPlaintextVector32, PlaintextVector32, PlaintextVectorPrototype, Precision32},
    {"64 bit plaintext vector", ProtoPlaintextVector64, PlaintextVector64, PlaintextVectorPrototype, Precision64},
    {"32 bit binary lwe secret key", ProtoBinaryLweSecretKey32, LweSecretKey32, BinaryLweSecretKeyPrototype, Precision32},
    {"64 bit binary lwe secret key", ProtoBinaryLweSecretKey64, LweSecretKey64, BinaryLweSecretKeyPrototype, Precision64},
    {"32 bit binary lwe ciphertext", ProtoBinaryLweCiphertext32, LweCiphertext32, BinaryLweCiphertextPrototype, Precision32},
    {"64 bit binary lwe ciphertext", ProtoBinaryLweCiphertext64, LweCiphertext64, BinaryLweCiphertextPrototype, Precision64},
    {"32 bit binary lwe ciphertext vector", ProtoBinaryLweCiphertextVector32, LweCiphertextVector32, BinaryLweCiphertextVectorPrototype, Precision32},
    {"64 bit binary lwe ciphertext vector", ProtoBinaryLweCiphertextVector64, LweCiphertextVector64, BinaryLweCiphertextVectorPrototype, Precision64},
    {"32 bit binary glwe secret key", ProtoBinaryGlweSecretKey32, GlweSecretKey32, BinaryGlweSecretKeyPrototype, Precision32},
    {"64 bit binary glwe secret key", ProtoBinaryGlweSecretKey64, GlweSecretKey64, BinaryGlweSecretKeyPrototype, Precision64},
    {"32 bit binary glwe ciphertext", ProtoBinaryGlweCiphertext32, GlweCiphertext32, BinaryGlweCiphertextPrototype, Precision32},
    {"64 bit binary glwe ciphertext", ProtoBinaryGlweCiphertext64, GlweCiphertext64, BinaryGlweCiphertextPrototype, Precision64},
    {"32 bit binary glwe ciphertext vector", ProtoBinaryGlweCiphertextVector32, GlweCiphertextVector32, BinaryGlweCiphertextVectorPrototype, Precision32},
    {"64 bit binary glwe ciphertext vector", ProtoBinaryGlweCiphertextVector64, GlweCiphertextVector64, BinaryGlweCiphertextVectorPrototype, Precision64},
    {"32 bit binary to binary lwe keyswitch key", ProtoBinaryBinaryLweKeyswitchKey32, LweKeyswitchKey32, BinaryBinaryLweKeyswitchKeyPrototype, Precision32},
    {"64 bit binary to binary lwe keyswitch key", ProtoBinaryBinaryLweKeyswitchKey64, LweKeyswitchKey64, BinaryBinaryLweKeyswitchKeyPrototype, Precision64},
    {"32 bit binary to binary lwe bootstrap key", ProtoBinaryBinaryLweBootstrapKey32, LweBootstrapKey32, BinaryBinaryLweBootstrapKeyPrototype, Precision32},
    {"64 bit binary to binary lwe bootstrap key", ProtoBinaryBinaryLweBootstrapKey64, LweBootstrapKey64, BinaryBinaryLweBootstrapKeyPrototype, Precision64}
}
