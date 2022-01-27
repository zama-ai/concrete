use crate::backends::core::private::crypto::secret::GlweSecretKey as ImpGlweSecretKey;
use crate::specification::entities::markers::{BinaryKeyDistribution, GlweSecretKeyKind};
use crate::specification::entities::{AbstractEntity, GlweSecretKeyEntity};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A structure representing a GLWE secret key with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweSecretKey32(pub(crate) ImpGlweSecretKey<BinaryKeyKind, Vec<u32>>);
impl AbstractEntity for GlweSecretKey32 {
    type Kind = GlweSecretKeyKind;
}
impl GlweSecretKeyEntity for GlweSecretKey32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.key_size()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}

/// A structure representing a GLWE secret key with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweSecretKey64(pub(crate) ImpGlweSecretKey<BinaryKeyKind, Vec<u64>>);
impl AbstractEntity for GlweSecretKey64 {
    type Kind = GlweSecretKeyKind;
}
impl GlweSecretKeyEntity for GlweSecretKey64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn glwe_dimension(&self) -> GlweDimension {
        self.0.key_size()
    }

    fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }
}
