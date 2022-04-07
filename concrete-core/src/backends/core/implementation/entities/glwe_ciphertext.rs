use concrete_fftw::array::AlignedVec;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::parameters::{GlweDimension, PolynomialSize};

use crate::backends::core::private::math::fft::Complex64;
use crate::specification::entities::markers::{BinaryKeyDistribution, GlweCiphertextKind};
use crate::specification::entities::{AbstractEntity, GlweCiphertextEntity};

use super::super::super::private::crypto::glwe::{
    FourierGlweCiphertext as ImplFourierGlweCiphertext, GlweCiphertext as ImplGlweCiphertext,
};

/// A structure representing a GLWE ciphertext with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweCiphertext32(pub(crate) ImplGlweCiphertext<Vec<u32>>);

impl AbstractEntity for GlweCiphertext32 {
    type Kind = GlweCiphertextKind;
}

impl GlweCiphertextEntity for GlweCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a GLWE ciphertext with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweCiphertext64(pub(crate) ImplGlweCiphertext<Vec<u64>>);

impl AbstractEntity for GlweCiphertext64 {
    type Kind = GlweCiphertextKind;
}

impl GlweCiphertextEntity for GlweCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a Fourier GLWE ciphertext with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]

pub struct FourierGlweCiphertext32(
    pub(crate) ImplFourierGlweCiphertext<AlignedVec<Complex64>, u32>,
);
impl AbstractEntity for FourierGlweCiphertext32 {
    type Kind = GlweCiphertextKind;
}
impl GlweCiphertextEntity for FourierGlweCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a Fourier GLWE ciphertext with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FourierGlweCiphertext64(
    pub(crate) ImplFourierGlweCiphertext<AlignedVec<Complex64>, u64>,
);
impl AbstractEntity for FourierGlweCiphertext64 {
    type Kind = GlweCiphertextKind;
}
impl GlweCiphertextEntity for FourierGlweCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.glwe_size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

// GlweCiphertextViews are just GlweCiphertext entities that do not own their memory, they use a
// slice as a container as opposed to Vec for the standard GlweCiphertext

/// A structure representing a GLWE ciphertext view, with 32 bits of precision.
///
/// By _view_ here, we mean that the entity does not own the data, but immutably borrows it.
///
/// Notes:
/// ------
/// This view is not Clone as Clone for a slice is not defined. It is not Deserialize either,
/// as Deserialize of a slice is not defined. Immutable variant.
#[cfg_attr(feature = "serde_serialize", derive(Serialize))]
#[derive(Debug, PartialEq)]
pub struct GlweCiphertextView32<'a>(pub(crate) ImplGlweCiphertext<&'a [u32]>);
impl AbstractEntity for GlweCiphertextView32<'_> {
    type Kind = GlweCiphertextKind;
}

impl GlweCiphertextEntity for GlweCiphertextView32<'_> {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a GLWE ciphertext view, with 32 bits of precision.
///
/// By _view_ here, we mean that the entity does not own the data, but mutably borrows it.
///
/// Notes:
/// ------
/// This view is not Clone as Clone for a slice is not defined. It is not Deserialize either,
/// as Deserialize of a slice is not defined. Mutable variant.
#[cfg_attr(feature = "serde_serialize", derive(Serialize))]
#[derive(Debug, PartialEq)]
pub struct GlweCiphertextMutView32<'a>(pub(crate) ImplGlweCiphertext<&'a mut [u32]>);
impl AbstractEntity for GlweCiphertextMutView32<'_> {
    type Kind = GlweCiphertextKind;
}

impl GlweCiphertextEntity for GlweCiphertextMutView32<'_> {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a GLWE ciphertext view, with 32 bits of precision.
///
/// By _view_ here, we mean that the entity does not own the data, but immutably borrows it.
///
/// Notes:
/// ------
/// This view is not Clone as Clone for a slice is not defined. It is not Deserialize either,
/// as Deserialize of a slice is not defined. Immutable variant.
#[cfg_attr(feature = "serde_serialize", derive(Serialize))]
#[derive(Debug, PartialEq)]
pub struct GlweCiphertextView64<'a>(pub(crate) ImplGlweCiphertext<&'a [u64]>);

impl AbstractEntity for GlweCiphertextView64<'_> {
    type Kind = GlweCiphertextKind;
}

impl GlweCiphertextEntity for GlweCiphertextView64<'_> {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a GLWE ciphertext view, with 64 bits of precision.
///
/// By _view_ here, we mean that the entity does not own the data, but mutably borrows it.
///
/// Notes:
/// ------
/// This view is not Clone as Clone for a slice is not defined. It is not Deserialize either,
/// as Deserialize of a slice is not defined. Mutable variant.
#[cfg_attr(feature = "serde_serialize", derive(Serialize))]
#[derive(Debug, PartialEq)]
pub struct GlweCiphertextMutView64<'a>(pub(crate) ImplGlweCiphertext<&'a mut [u64]>);

impl AbstractEntity for GlweCiphertextMutView64<'_> {
    type Kind = GlweCiphertextKind;
}

impl GlweCiphertextEntity for GlweCiphertextMutView64<'_> {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.size().to_glwe_dimension()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}
