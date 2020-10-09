//! Encoding Tensor Operations
//! * Contains functions dealing with encodings and decodings at a tensor level

#[cfg(test)]
mod tests;

use crate::Types;

pub trait Encoding: Sized {
    fn encode(message: f64, o: f64, delta: f64) -> Self;
    fn several_encode(encoded: &mut [Self], message: &[f64], o: &[f64], delta: &[f64]);
    fn several_encode_with_same_parameters(
        encoded: &mut [Self],
        message: &[f64],
        o: f64,
        delta: f64,
    );
    fn decode(encoding: Self, o: f64, delta: f64) -> f64;
    fn several_decode(decoded: &mut [f64], encoding: &[Self], os: &[f64], delta: &[f64]);
    fn several_decode_with_same_parameters(
        decoded: &mut [f64],
        encoding: &[Self],
        o: f64,
        delta: f64,
    );
}

macro_rules! impl_trait_encoding {
    ($T:ty,$DOC:expr) => {
        impl Encoding for $T {
            /// Encode a message with the real encoding
            /// * `message` - a message seen a real value in a f64
            /// * `o` - the offset of the encoding
            /// * `delta` - the delta of the encoding
            /// # Output
            /// * a Torus element encoding the message
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::Encoding;
            #[doc = $DOC]
            ///
            /// // we want to work in [-10,30[
            /// let o: f64 = -10.;
            /// let delta: f64 = 40.;
            ///
            /// // creation of a message
            /// let m: f64 = -2.5;
            ///
            /// // encoding of our message
            /// let enc: Torus = Encoding::encode(m, o, delta);
            /// ```
            fn encode(message: f64, o: f64, delta: f64) -> $T {
                return (((message - o) / delta) * f64::powi(2.0, <$T as Types>::TORUS_BIT as i32)) as $T;
            }

            /// Encode several messages with different real encodings
            /// * `encoded` - a slice of Torus element encoding each message (output)
            /// * `messages` - a slice of messages seen as real values
            /// * `os` - offsets of the encoding
            /// * `deltas` - deltas of the encoding
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::Encoding;
            #[doc = $DOC]
            ///
            /// // we want to work in [-10,30[ and [-20, 20]
            /// let os: Vec<f64> = vec![-10., -20.];
            /// let deltas: Vec<f64> = vec![40., 40.];
            ///
            /// // creation of the messages
            /// let messages: Vec<f64> = vec![-2.5, 3.];
            ///
            /// // encoding of our message
            /// let mut enc: Vec<Torus> = vec![0 as Torus ; 2] ;
            /// Encoding::several_encode(&mut enc, &messages, &os, &deltas);
            /// ```
            fn several_encode(encoded: &mut [Self], messages: &[f64], os: &[f64], deltas: &[f64]){
                for (enc, m, o, delta) in izip!(encoded.iter_mut(), messages.iter(), os.iter(), deltas.iter()) {
                    *enc = (((m - o) / delta) * f64::powi(2.0, <$T as Types>::TORUS_BIT as i32)) as $T;
                }
            }

            /// Encode several messages with a real encoding
            /// * `encoded` - a slice of Torus element encoding each message (output)
            /// * `messages` - a slice of messages seen as real values
            /// * `o` - offset of the encoding
            /// * `delta` - delta of the encoding
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::Encoding;
            #[doc = $DOC]
            ///
            /// // we want to work in [-10,30[ and [-20, 20]
            /// let o: f64 = -10. ;
            /// let delta: f64 = 40. ;
            ///
            /// // creation of the messages
            /// let messages: Vec<f64> = vec![-2.5, 3.];
            ///
            /// // encoding of our message
            /// let mut enc: Vec<Torus> = vec![0 as Torus ; 2] ;
            /// Encoding::several_encode_with_same_parameters(&mut enc, &messages, o, delta);
            /// ```
            fn several_encode_with_same_parameters(encoded: &mut [Self], messages: &[f64], o: f64, delta: f64){
                for (enc, m,) in izip!(encoded.iter_mut(), messages.iter()) {
                    *enc = (((m - o) / delta) * f64::powi(2.0, <$T as Types>::TORUS_BIT as i32)) as $T;
                }
            }

            /// Decode a message with the real encoding
            /// # Arguments
            /// * `encoding` - a Torus element which encoding a message
            /// * `o` - the offset of the encoding
            /// * `delta` - the delta of the encoding
            /// # Output
            /// * a f64 message that was encoded in encoding
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::Encoding;
            #[doc = $DOC]
            ///
            /// // we want to work in [-10,30[
            /// let o: f64 = -10.;
            /// let delta: f64 = 40.;
            ///
            /// // we have an encoding
            /// let enc: Torus = 2 << 31;
            ///
            /// // encoding of our message
            /// let dec: f64 = Encoding::decode(enc, o, delta);
            /// ```
            fn decode(encoding: $T, o: f64, delta: f64) -> f64 {
                let mut e: f64 = (encoding as f64) / f64::powi(2.0, <$T as Types>::TORUS_BIT as i32);
                e = e * delta + o;
                return e;
            }

            /// Decode several messages with the real encoding
            /// # Arguments
            /// * `messages` - a slice of messages seen as real values (output)
            /// * `encoding` - a slice of Torus element encoding each message
            /// * `os` - offsets of the encoding
            /// * `deltas` - deltas of the encoding
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::Encoding;
            #[doc = $DOC]
            ///
            /// // we want to work in [-10,30[ and [-20., 20.]
            /// let os: Vec<f64> = vec![-10. , -20.];
            /// let deltas: Vec<f64> = vec![40., 40.];
            ///
            /// // we have two encodings
            /// let encs: Vec<Torus> = vec![2 << 31, 2 << 30];
            ///
            /// // decoding of our messages
            /// let mut decs: Vec<f64> = vec![0.; 2];
            /// Encoding::several_decode(&mut decs, &encs,  &os, &deltas);
            /// ```
            fn several_decode(decodings: &mut [f64], encodings: &[$T], os: &[f64], deltas: &[f64])  {
                for (dec, enc, o, delta) in izip!(decodings.iter_mut(), encodings.iter(), os.iter(), deltas.iter()) {
                    let e: f64 = (*enc as f64) / f64::powi(2.0, <$T as Types>::TORUS_BIT as i32);
                    *dec  = e * *delta + *o;
                }

            }

                        /// Decode several messages with the real encoding
            /// # Arguments
            /// * `messages` - a slice of messages seen as real values (output)
            /// * `encoding` - a slice of Torus element encoding each message
            /// * `os` - offsets of the encoding
            /// * `deltas` - deltas of the encoding
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::Encoding;
            #[doc = $DOC]
            ///
            /// // we want to work in [-10,30[ and [-20., 20.]
            /// let o: f64 = -10. ;
            /// let delta: f64 = 40.;
            ///
            /// // we have two encodings
            /// let encs: Vec<Torus> = vec![2 << 31, 2 << 30];
            ///
            /// // decoding of our messages
            /// let mut decs: Vec<f64> = vec![0.; 2];
            /// Encoding::several_decode_with_same_parameters(&mut decs, &encs,  o, delta);
            /// ```
            fn several_decode_with_same_parameters(decodings: &mut [f64], encodings: &[$T], o: f64, delta: f64)  {
                for (dec, enc) in izip!(decodings.iter_mut(), encodings.iter()) {
                    let e: f64 = (*enc as f64) / f64::powi(2.0, <$T as Types>::TORUS_BIT as i32);
                    *dec  = e * delta + o;
                }
            }
        }
    };
}

impl_trait_encoding!(u32, "type Torus = u32;");
impl_trait_encoding!(u64, "type Torus = u64;");
