//! A module containing various marker traits used for entities.
use std::fmt::Debug;

/// A trait implemented by marker types encoding the __kind__ of an FHE entity in
/// the type system.
///
/// By _kind_ here, we mean the _what_, the abstract nature of an FHE entity.
///
/// # Note
///
/// [`EntityKindMarker`] types are only defined in the specification part of the library, and
/// can not be defined by a backend.
pub trait EntityKindMarker: seal::EntityKindMarkerSealed {}
macro_rules! entity_kind_marker {
        (@ $name: ident => $doc: literal)=>{
            #[doc=$doc]
            #[derive(Debug, Clone, Copy)]
            pub struct $name{}
            impl seal::EntityKindMarkerSealed for $name{}
            impl EntityKindMarker for $name{}
        };
        ($($name: ident => $doc: literal),+) =>{
            $(
                entity_kind_marker!(@ $name => $doc);
            )+
        }
}
entity_kind_marker! {
        PlaintextKind
            => "An empty type representing the plaintext kind in the type system.",
        PlaintextVectorKind
            => "An empty type representing the plaintext vector kind in the type system",
        CleartextKind
            => "An empty type representing the cleartext kind in the type system.",
        CleartextVectorKind
            => "An empty type representing the cleartext vector kind in the type system.",
        LweCiphertextKind
            => "An empty type representing the LWE ciphertext kind in the type system.",
        LweCiphertextVectorKind
            => "An empty type representing the LWE ciphertext vector kind in the type system.",
        GlweCiphertextKind
            => "An empty type representing the GLWE ciphertext kind in the type system.",
        GlweCiphertextVectorKind
            => "An empty type representing the GLWE ciphertext vector kind in the type system.",
        GgswCiphertextKind
            => "An empty type representing the GGSW ciphertext kind in the type system.",
        GgswCiphertextVectorKind
            => "An empty type representing the GGSW ciphertext vector kind in the type system.",
        GswCiphertextKind
            => "An empty type representing the GSW ciphertext kind in the type system.",
        GswCiphertextVectorKind
            => "An empty type representing the GSW ciphertext vector kind in the type system.",
        LweSecretKeyKind
            => "An empty type representing the LWE secret key kind in the type system.",
        GlweSecretKeyKind
            => "An empty type representing the GLWE secret key kind in the type system.",
        LweKeyswitchKeyKind
            => "An empty type representing the LWE keyswitch key kind in the type system.",
        LweBootstrapKeyKind
            => "An empty type representing the LWE bootstrap key kind in the type system.",
        EncoderKind
            => "An empty type representing the encoder kind in the type system.",
        EncoderVectorKind
            => "An empty type representing the encoder vector kind in the type system"
}

/// A trait implemented by marker types encoding a _distribution_ of secret key in the type system.
///
/// By _distribution_ here, we mean the different types of secret key that can exist such as binary,
/// ternary, uniform or gaussian key.
///
/// # Note
///
/// [`KeyDistributionMarker`] types are only defined in the specification part of the library, and
/// can not be defined by a backend.
pub trait KeyDistributionMarker: seal::KeyDistributionMarkerSealed + 'static {}
macro_rules! key_distribution_marker {
        (@ $name: ident => $doc: literal)=>{
            #[doc=$doc]
            #[derive(Debug, Clone, Copy)]
            pub struct $name{}
            impl seal::KeyDistributionMarkerSealed for $name{}
            impl KeyDistributionMarker for $name{}
        };
        ($($name: ident => $doc: literal),+) =>{
            $(
                key_distribution_marker!(@ $name => $doc);
            )+
        }
    }
key_distribution_marker! {
    BinaryKeyDistribution => "An empty type encoding the binary key distribution in the type system.",
    TernaryKeyDistribution => "An empty type encoding the ternary key distribution in the type system.",
    GaussianKeyDistribution => "An empty type encoding the gaussian key distribution in the type system."
}

pub(crate) mod seal {
    pub trait EntityKindMarkerSealed {}
    pub trait KeyDistributionMarkerSealed {}
}
