//! A module to synthesize actual entities from prototypical entities.
//!
//! This module allows to convert back and forth, prototypical entities to the actual entity types
//! used for tests.
use crate::prototyping::prototyper::Prototyper;
use crate::prototyping::prototypes::*;
use crate::prototyping::*;
use crate::Maker;
use concrete_core::prelude::markers::*;
use concrete_core::prelude::*;

macro_rules! synthesizable_traits {
    ($({$kind: literal, $name:ident, $entity:ident, $prototype:ident}),+) => {
        $(
            #[doc = "A trait to synthesize "]
            #[doc = $kind]
            #[doc = " from prototypical entities."]
            pub trait $name<Prec: IntegerPrecision>: $entity where Maker: Prototyper<Prec>{
                fn from_prototype(synth: &mut Maker, prototype: &<Maker as Prototyper<Prec>>::$prototype) -> Self;
                fn into_prototype(synth: &mut Maker, entity: &Self) -> <Maker as Prototyper<Prec>>::$prototype;
            }
        )*
    };
    ($({$kind: literal, $name:ident, $entity:ident<$($left:ident=$right:ident),+>, $prototype:ident}),+) => {
        $(
            #[doc = "A trait to synthesize "]
            #[doc = $kind]
            #[doc = " from prototypical entities."]
            pub trait $name<Prec: IntegerPrecision>: $entity < $($left=$right),* > where Maker: Prototyper<Prec>{
                fn from_prototype(synth: &mut Maker, prototype: &<Maker as Prototyper<Prec>>::$prototype) -> Self;
                fn into_prototype(synth: &mut Maker, entity: &Self) -> <Maker as Prototyper<Prec>>::$prototype;
            }
        )*
    };
}

synthesizable_traits! {
    {"cleartext", SynthesizableCleartext, CleartextEntity, CleartextPrototype},
    {"cleartext vector", SynthesizableCleartextVector, CleartextVectorEntity, CleartextVectorPrototype},
    {"plaintext", SynthesizablePlaintext, PlaintextEntity, PlaintextPrototype},
    {"plaintext vector", SynthesizablePlaintextVector, PlaintextVectorEntity, PlaintextVectorPrototype}
}

synthesizable_traits! {
    {"binary lwe secret key", SynthesizableBinaryLweSecretKey, LweSecretKeyEntity<KeyDistribution=BinaryKeyDistribution>, BinaryLweSecretKeyPrototype},
    {"binary lwe ciphertext", SynthesizableBinaryLweCiphertext, LweCiphertextEntity<KeyDistribution=BinaryKeyDistribution>, BinaryLweCiphertextPrototype},
    {"binary lwe ciphertext vector", SynthesizableBinaryLweCiphertextVector, LweCiphertextVectorEntity<KeyDistribution=BinaryKeyDistribution>, BinaryLweCiphertextVectorPrototype},
    {"binary glwe secret key", SynthesizableBinaryGlweSecretKey, GlweSecretKeyEntity<KeyDistribution=BinaryKeyDistribution>, BinaryGlweSecretKeyPrototype},
    {"binary glwe ciphertext", SynthesizableBinaryGlweCiphertext, GlweCiphertextEntity<KeyDistribution=BinaryKeyDistribution>, BinaryGlweCiphertextPrototype},
    {"binary glwe ciphertext vector", SynthesizableBinaryGlweCiphertextVector, GlweCiphertextVectorEntity<KeyDistribution=BinaryKeyDistribution>, BinaryGlweCiphertextVectorPrototype},
    {"binary to binary lwe keyswitch key", SynthesizableBinaryBinaryLweKeyswitchKey, LweKeyswitchKeyEntity<InputKeyDistribution=BinaryKeyDistribution, OutputKeyDistribution=BinaryKeyDistribution>, BinaryBinaryLweKeyswitchKeyPrototype},
    {"binary to binary lwe bootstrap key", SynthesizableBinaryBinaryLweBootstrapKey, LweBootstrapKeyEntity<InputKeyDistribution=BinaryKeyDistribution, OutputKeyDistribution=BinaryKeyDistribution>, BinaryBinaryLweBootstrapKeyPrototype}
}

macro_rules! impl_synthesizable_for_core {
    ($({$traitname:ident, $precision:ident, $entity:ident, $proto:ident}),*) => {
        $(
            impl $traitname<$precision> for $entity {
                fn from_prototype(_synth: &mut Maker, prototype: &$proto) -> Self {
                    prototype.0.to_owned()
                }

                fn into_prototype(_synth: &mut Maker, entity: &Self) -> $proto {
                    $proto(entity.to_owned())
                }
            }
        )*
    };
}

impl_synthesizable_for_core! {
    {SynthesizablePlaintext, Precision32, Plaintext32, ProtoPlaintext32},
    {SynthesizablePlaintext, Precision64, Plaintext64, ProtoPlaintext64},
    {SynthesizablePlaintextVector, Precision32, PlaintextVector32, ProtoPlaintextVector32},
    {SynthesizablePlaintextVector, Precision64, PlaintextVector64, ProtoPlaintextVector64},
    {SynthesizableCleartext, Precision32, Cleartext32, ProtoCleartext32},
    {SynthesizableCleartext, Precision64, Cleartext64, ProtoCleartext64},
    {SynthesizableCleartextVector, Precision32, CleartextVector32, ProtoCleartextVector32},
    {SynthesizableCleartextVector, Precision64, CleartextVector64, ProtoCleartextVector64},
    {SynthesizableBinaryLweSecretKey, Precision32, LweSecretKey32, ProtoBinaryLweSecretKey32},
    {SynthesizableBinaryLweSecretKey, Precision64, LweSecretKey64, ProtoBinaryLweSecretKey64},
    {SynthesizableBinaryLweCiphertext, Precision32, LweCiphertext32, ProtoBinaryLweCiphertext32},
    {SynthesizableBinaryLweCiphertext, Precision64, LweCiphertext64, ProtoBinaryLweCiphertext64},
    {SynthesizableBinaryLweCiphertextVector, Precision32, LweCiphertextVector32, ProtoBinaryLweCiphertextVector32},
    {SynthesizableBinaryLweCiphertextVector, Precision64, LweCiphertextVector64, ProtoBinaryLweCiphertextVector64},
    {SynthesizableBinaryGlweSecretKey, Precision32, GlweSecretKey32, ProtoBinaryGlweSecretKey32},
    {SynthesizableBinaryGlweSecretKey, Precision64, GlweSecretKey64, ProtoBinaryGlweSecretKey64},
    {SynthesizableBinaryGlweCiphertext, Precision32, GlweCiphertext32, ProtoBinaryGlweCiphertext32},
    {SynthesizableBinaryGlweCiphertext, Precision64, GlweCiphertext64, ProtoBinaryGlweCiphertext64},
    {SynthesizableBinaryGlweCiphertextVector, Precision32, GlweCiphertextVector32, ProtoBinaryGlweCiphertextVector32},
    {SynthesizableBinaryGlweCiphertextVector, Precision64, GlweCiphertextVector64, ProtoBinaryGlweCiphertextVector64},
    {SynthesizableBinaryBinaryLweKeyswitchKey, Precision32, LweKeyswitchKey32, ProtoBinaryBinaryLweKeyswitchKey32},
    {SynthesizableBinaryBinaryLweKeyswitchKey, Precision64, LweKeyswitchKey64, ProtoBinaryBinaryLweKeyswitchKey64},
    {SynthesizableBinaryBinaryLweBootstrapKey, Precision32, LweBootstrapKey32, ProtoBinaryBinaryLweBootstrapKey32},
    {SynthesizableBinaryBinaryLweBootstrapKey, Precision64, LweBootstrapKey64, ProtoBinaryBinaryLweBootstrapKey64}
}
