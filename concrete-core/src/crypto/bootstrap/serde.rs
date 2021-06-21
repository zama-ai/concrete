use std::fmt;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use serde::Deserialize;

use crate::math::decomposition::{DecompositionBaseLog, DecompositionLevelCount};
use crate::math::fft::{Complex64, SerializableComplex64};
use crate::math::polynomial::PolynomialSize;
use crate::math::tensor::Tensor;

use super::{BootstrapKey, GlweSize};

impl Serialize for BootstrapKey<Vec<Complex64>> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("BSKv0", 5)?;
        state.serialize_field("v", &(0u8))?;
        state.serialize_field("c", unsafe {
            std::slice::from_raw_parts(
                self.tensor.as_container().as_ptr() as *const SerializableComplex64,
                self.tensor.len(),
            )
        })?;
        state.serialize_field("p", &self.poly_size.0)?;
        state.serialize_field("r", &self.rlwe_size.0)?;
        state.serialize_field("l", &self.decomp_level.0)?;
        state.serialize_field("b", &self.decomp_base_log.0)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for BootstrapKey<Vec<Complex64>> {
    fn deserialize<D>(deserializer: D) -> Result<BootstrapKey<Vec<Complex64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            V,
            C,
            P,
            R,
            L,
            B,
        }

        const FIELDS: &'static [&'static str] = &["v", "c", "p", "r", "l", "b"];

        struct VisitorImpl;

        fn serializable_to_vec_complex64(x: Vec<SerializableComplex64>) -> Vec<Complex64> {
            let len = x.len();
            let cap = x.capacity();
            let ptr = x.leak();
            unsafe { Vec::from_raw_parts(ptr.as_mut_ptr() as *mut Complex64, len, cap) }
        }

        impl<'de> Visitor<'de> for VisitorImpl {
            type Value = BootstrapKey<Vec<Complex64>>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("BSKv0")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<BootstrapKey<Vec<Complex64>>, V::Error>
            where
                V: SeqAccess<'de>,
            {
                seq.next_element()?.map_or_else(
                    || Err(de::Error::invalid_length(0, &self)),
                    |v: u8| {
                        if v == 0 {
                            Ok(())
                        } else {
                            Err(de::Error::invalid_length(0, &self))
                        }
                    },
                )?;
                let c: Vec<SerializableComplex64> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let p = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let r = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;
                let l = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &self))?;
                let b = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(5, &self))?;
                Ok(BootstrapKey {
                    tensor: Tensor::from_container(serializable_to_vec_complex64(c)),
                    poly_size: PolynomialSize(p),
                    rlwe_size: GlweSize(r),
                    decomp_level: DecompositionLevelCount(l),
                    decomp_base_log: DecompositionBaseLog(b),
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<BootstrapKey<Vec<Complex64>>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut c: std::option::Option<Vec<SerializableComplex64>> = None;
                let mut p = None;
                let mut r = None;
                let mut l = None;
                let mut b = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::V => {
                            let v: u8 = map.next_value()?;
                            if v != 0 {
                                return Err(de::Error::invalid_value(
                                    de::Unexpected::Unsigned(v as u64),
                                    &"version number equal to 0",
                                ));
                            }
                        }
                        Field::C => {
                            if c.is_some() {
                                return Err(de::Error::duplicate_field("c"));
                            }
                            c = Some(map.next_value()?);
                        }
                        Field::P => {
                            if p.is_some() {
                                return Err(de::Error::duplicate_field("p"));
                            }
                            p = Some(map.next_value()?);
                        }
                        Field::R => {
                            if r.is_some() {
                                return Err(de::Error::duplicate_field("r"));
                            }
                            r = Some(map.next_value()?);
                        }
                        Field::L => {
                            if l.is_some() {
                                return Err(de::Error::duplicate_field("l"));
                            }
                            l = Some(map.next_value()?);
                        }
                        Field::B => {
                            if b.is_some() {
                                return Err(de::Error::duplicate_field("b"));
                            }
                            b = Some(map.next_value()?);
                        }
                    }
                }
                let c = c.ok_or_else(|| de::Error::missing_field("c"))?;
                let p = p.ok_or_else(|| de::Error::missing_field("p"))?;
                let r = r.ok_or_else(|| de::Error::missing_field("r"))?;
                let l = l.ok_or_else(|| de::Error::missing_field("l"))?;
                let b = b.ok_or_else(|| de::Error::missing_field("b"))?;
                Ok(BootstrapKey {
                    tensor: Tensor::from_container(serializable_to_vec_complex64(c)),
                    poly_size: PolynomialSize(p),
                    rlwe_size: GlweSize(r),
                    decomp_level: DecompositionLevelCount(l),
                    decomp_base_log: DecompositionBaseLog(b),
                })
            }
        }

        deserializer.deserialize_struct("BSKv0", FIELDS, VisitorImpl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::LweDimension;
    use crate::math::random::RandomGenerator;
    use crate::math::tensor::AsMutTensor;
    use crate::test_tools;
    use serde_test::{assert_de_tokens_error, assert_tokens, Token};

    #[test]
    fn test_ser_de() {
        let bsk = BootstrapKey::allocate(
            Complex64::new(9., 8.),
            GlweSize(1),
            PolynomialSize(1),
            DecompositionLevelCount(1),
            DecompositionBaseLog(1),
            LweDimension(1),
        );
        assert_tokens(
            &bsk,
            &[
                Token::Struct {
                    name: "BSKv0",
                    len: 5,
                },
                Token::Str("v"),
                Token::U8(0),
                Token::Str("c"),
                Token::Seq { len: Some(1) },
                Token::Tuple { len: 2 },
                Token::F64(9.0),
                Token::F64(8.0),
                Token::TupleEnd,
                Token::SeqEnd,
                Token::Str("p"),
                Token::U64(1),
                Token::Str("r"),
                Token::U64(1),
                Token::Str("l"),
                Token::U64(1),
                Token::Str("b"),
                Token::U64(1),
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn test_de_version_error() {
        assert_de_tokens_error::<BootstrapKey<Vec<Complex64>>>(
            &[
                Token::Struct {
                    name: "BSKv0",
                    len: 5,
                },
                Token::Str("v"),
                Token::U8(1),
            ],
            "invalid value: integer `1`, expected version number equal to 0",
        )
    }

    #[test]
    fn test_randomized_bsk_gen() {
        let mut generator = RandomGenerator::new(None);
        for _ in 0..100 {
            let mut bsk = BootstrapKey::allocate(
                Complex64::new(0., 0.),
                test_tools::random_glwe_dimension(5).to_glwe_size(),
                test_tools::random_polynomial_size(1024),
                test_tools::random_level_count(5),
                test_tools::random_base_log(4),
                test_tools::random_lwe_dimension(5),
            );
            for mut val in bsk.as_mut_tensor().iter_mut() {
                let (a, b) = generator.random_gaussian(0., 1.);
                val.re = a;
                val.im = b;
            }
            let serialized = bincode::serialize(&bsk).unwrap();
            let deserialized = bincode::deserialize(serialized.as_slice()).unwrap();
            assert_eq!(bsk, deserialized);
        }
    }
}
