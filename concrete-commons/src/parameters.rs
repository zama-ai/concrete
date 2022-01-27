#![allow(deprecated)]
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// The number plaintexts in a plaintext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct PlaintextCount(pub usize);

/// The number encoder in an encoder list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct EncoderCount(pub usize);

/// The number messages in a messages list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct CleartextCount(pub usize);

/// The number of ciphertexts in a ciphertext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct CiphertextCount(pub usize);

/// The number of ciphertexts in an lwe ciphertext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct LweCiphertextCount(pub usize);

/// The index of a ciphertext in an lwe ciphertext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct LweCiphertextIndex(pub usize);

/// The range of indices of multiple contiguous ciphertexts in an lwe ciphertext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct LweCiphertextRange(pub usize, pub usize);

impl LweCiphertextRange {
    pub fn is_ordered(&self) -> bool {
        self.1 <= self.0
    }
}

/// The number of ciphertexts in a glwe ciphertext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct GlweCiphertextCount(pub usize);

/// The number of ciphertexts in a gsw ciphertext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct GswCiphertextCount(pub usize);

/// The number of ciphertexts in a ggsw ciphertext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct GgswCiphertextCount(pub usize);

/// The number of scalars in an LWE ciphertext, i.e. the number of scalar in an LWE mask plus one.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct LweSize(pub usize);

impl LweSize {
    /// Returns the associated [`LweDimension`].
    pub fn to_lwe_dimension(&self) -> LweDimension {
        LweDimension(self.0 - 1)
    }
}

/// The number of scalar in an LWE mask, or the length of an LWE secret key.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct LweDimension(pub usize);

impl LweDimension {
    /// Returns the associated [`LweSize`].
    pub fn to_lwe_size(&self) -> LweSize {
        LweSize(self.0 + 1)
    }
}

/// The number of polynomials in a GLWE ciphertext, i.e. the number of polynomials in a GLWE mask
/// plus one.
#[derive(Debug, PartialEq, Eq, Ord, PartialOrd, Copy, Clone)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct GlweSize(pub usize);

impl GlweSize {
    /// Returns the associated [`GlweDimension`].
    pub fn to_glwe_dimension(&self) -> GlweDimension {
        GlweDimension(self.0 - 1)
    }
}

/// The number of polynomials of an GLWE mask, or the size of an GLWE secret key.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct GlweDimension(pub usize);

impl GlweDimension {
    /// Returns the associated [`GlweSize`].
    pub fn to_glwe_size(&self) -> GlweSize {
        GlweSize(self.0 + 1)
    }
}
/// The number of coefficients of a polynomial.
///
/// Assuming a polynomial $a_0 + a_1X + /dots + a_nX^N$, this returns $N+1$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct PolynomialSize(pub usize);

/// The number of polynomials in a polynomial list.
///
/// Assuming a polynomial list, this return the number of polynomials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct PolynomialCount(pub usize);

/// The degree of a monomial.
///
/// Assuming a monomial $aX^N$, this returns the $N$ value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[deprecated(note = "MonomialDegree is not used anymore in the API. You should not use it.")]
pub struct MonomialDegree(pub usize);

/// The index of a monomial in a polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct MonomialIndex(pub usize);

/// The logarithm of the base used in a decomposition.
///
/// When decomposing an integer over powers of the $2^B$ basis, this type represents the $B$ value.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct DecompositionBaseLog(pub usize);

/// The number of levels used in a decomposition.
///
/// When decomposing an integer over the $l$ largest powers of the basis, this type represents
/// the $l$ value.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
pub struct DecompositionLevelCount(pub usize);
