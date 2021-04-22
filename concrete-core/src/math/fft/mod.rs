//! Fourier transform for polynomials.
//!
//! This module provides the tools to perform a fast product of two polynomials, reduced modulo
//! $X^N+1$, using the fast fourier transform.
use serde::ser::{Serialize, Serializer, SerializeTuple};
use serde::de::{self, Deserialize, Deserializer, Visitor, SeqAccess};
use std::fmt;



#[cfg(test)]
mod tests;

mod twiddles;
use twiddles::*;

mod polynomial;
pub use polynomial::*;

mod transform;
pub use transform::*;

/// A complex number encoded over two `f64`.
pub type Complex64 = fftw::types::c64;

#[derive(PartialEq, Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct SerializableComplex64(Complex64);

impl Serialize for SerializableComplex64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_tuple(2)?;
        s.serialize_element(&self.0.re)?;
        s.serialize_element(&self.0.im)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for SerializableComplex64 {
    fn deserialize<D>(deserializer: D) -> Result<SerializableComplex64, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct VisitorImpl;

        impl<'de> Visitor<'de> for VisitorImpl {
            type Value = SerializableComplex64;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("Complex")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<SerializableComplex64, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let re = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let im = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(SerializableComplex64(Complex64{re, im}))
            }

        }

        deserializer.deserialize_tuple(2, VisitorImpl)
    }
}