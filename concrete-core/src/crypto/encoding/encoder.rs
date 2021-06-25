use crate::math::tensor::{AsMutTensor, AsRefTensor};
use crate::math::torus::{FromTorus, IntoTorus, UnsignedTorus};

use super::{Cleartext, CleartextList, Plaintext, PlaintextList};
use concrete_commons::numeric::{FloatingPoint, Numeric};

/// A trait for types that encode cleartext to plaintext.
///
/// Examples use the [`RealEncoder'] type.
pub trait Encoder<Enc: Numeric> {
    /// The type of the cleartexts.
    type Raw: Numeric;

    /// Encodes a single cleartext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::encoding::*;
    /// let encoder = RealEncoder {
    ///     offset: 1. as f32,
    ///     delta: 10.,
    /// };
    /// let cleartext = Cleartext(7. as f32);
    /// let encoded: Plaintext<u32> = encoder.encode(cleartext.clone());
    /// let decoded = encoder.decode(encoded);
    /// assert!((cleartext.0 - decoded.0).abs() < 0.1);
    /// ```
    fn encode(&self, raw: Cleartext<Self::Raw>) -> Plaintext<Enc>;

    /// Decodes a single encoded value.
    ///
    /// See [`Encoder::encode`] for an example.
    fn decode(&self, encoded: Plaintext<Enc>) -> Cleartext<Self::Raw>;

    /// Encodes a list of cleartexts to a list of plaintexts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::encoding::*;
    /// let encoder = RealEncoder {
    ///     offset: 1. as f32,
    ///     delta: 10.,
    /// };
    /// let clear_values = CleartextList::from_container(vec![7. as f32; 100]);
    /// let mut plain_values = PlaintextList::from_container(vec![0 as u32; 100]);
    /// encoder.encode_list(&mut plain_values, &clear_values);
    /// let mut decoded_values = CleartextList::from_container(vec![0. as f32; 100]);
    /// encoder.decode_list(&mut decoded_values, &plain_values);
    /// for (clear, decoded) in clear_values
    ///     .cleartext_iter()
    ///     .zip(decoded_values.cleartext_iter())
    /// {
    ///     assert!((clear.0 - decoded.0).abs() < 0.1);
    /// }
    /// ```
    fn encode_list<RawCont, EncCont>(
        &self,
        encoded: &mut PlaintextList<EncCont>,
        raw: &CleartextList<RawCont>,
    ) where
        CleartextList<RawCont>: AsRefTensor<Element = Self::Raw>,
        PlaintextList<EncCont>: AsMutTensor<Element = Enc>;

    /// Decodes a list of plaintexts into a list of cleartexts.
    ///
    /// See [`Encoder::encode_list`] for an example.
    fn decode_list<RawCont, EncCont>(
        &self,
        raw: &mut CleartextList<RawCont>,
        encoded: &PlaintextList<EncCont>,
    ) where
        CleartextList<RawCont>: AsMutTensor<Element = Self::Raw>,
        PlaintextList<EncCont>: AsRefTensor<Element = Enc>;
}

/// An encoder for real cleartexts
pub struct RealEncoder<T: FloatingPoint> {
    /// The offset of the encoding
    pub offset: T,
    /// The delta of the encoding
    pub delta: T,
}

impl<RawScalar, EncScalar> Encoder<EncScalar> for RealEncoder<RawScalar>
where
    EncScalar: UnsignedTorus + FromTorus<RawScalar> + IntoTorus<RawScalar>,
    RawScalar: FloatingPoint,
{
    type Raw = RawScalar;
    fn encode(&self, raw: Cleartext<RawScalar>) -> Plaintext<EncScalar> {
        Plaintext(<EncScalar as FromTorus<RawScalar>>::from_torus(
            (raw.0 - self.offset) / self.delta,
        ))
    }
    fn decode(&self, encoded: Plaintext<EncScalar>) -> Cleartext<RawScalar> {
        let mut e: RawScalar = encoded.0.into_torus();
        e *= self.delta;
        e += self.offset;
        Cleartext(e)
    }
    fn encode_list<RawCont, EncCont>(
        &self,
        encoded: &mut PlaintextList<EncCont>,
        raw: &CleartextList<RawCont>,
    ) where
        CleartextList<RawCont>: AsRefTensor<Element = RawScalar>,
        PlaintextList<EncCont>: AsMutTensor<Element = EncScalar>,
    {
        encoded
            .as_mut_tensor()
            .fill_with_one(raw.as_tensor(), |r| self.encode(Cleartext(*r)).0);
    }
    fn decode_list<RawCont, EncCont>(
        &self,
        raw: &mut CleartextList<RawCont>,
        encoded: &PlaintextList<EncCont>,
    ) where
        CleartextList<RawCont>: AsMutTensor<Element = RawScalar>,
        PlaintextList<EncCont>: AsRefTensor<Element = EncScalar>,
    {
        raw.as_mut_tensor()
            .fill_with_one(encoded.as_tensor(), |e| self.decode(Plaintext(*e)).0);
    }
}
