use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Copy, Debug)]
pub enum EncryptionKeyChoice {
    BIG = 0,
    SMALL = 1,
}

impl TryFrom<i32> for EncryptionKeyChoice {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(EncryptionKeyChoice::BIG),
            1 => Ok(EncryptionKeyChoice::SMALL),
            _ => Err("Invalid value for EncryptionKeyChoice"),
        }
    }
}

impl Serialize for EncryptionKeyChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (*self as i32).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EncryptionKeyChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = i32::deserialize(deserializer)?;
        EncryptionKeyChoice::try_from(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CryptoParams {
    pub lwe_dimension: usize,
    pub glwe_dimension: usize,
    pub polynomial_size: usize,
    pub pbs_base_log: usize,
    pub pbs_level: usize,
    pub lwe_noise_distribution: f64,
    pub glwe_noise_distribution: f64,
    pub encryption_key_choice: EncryptionKeyChoice,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegerType {
    pub carry_width: usize,
    pub msg_width: usize,
    pub is_signed: bool,
    pub bit_width: usize,
    pub params: CryptoParams,
}

impl IntegerType {
    pub fn n_cts(&self) -> usize {
        self.bit_width / self.msg_width
    }
}

#[cfg(feature = "compiler")]
mod to_tokens {
    //! This module contains `ToTokens` implementations. This allows protocol
    //! values to be interpolated in the `quote!` macro as constructors of the values.
    //! Useful to construct static protocol values.

    use super::*;
    use proc_macro2::TokenStream;
    use quote::{quote, ToTokens};

    impl ToTokens for EncryptionKeyChoice {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                EncryptionKeyChoice::BIG => {
                    tokens.extend(quote! {::concrete::tfhe::EncryptionKeyChoice::BIG})
                }
                EncryptionKeyChoice::SMALL => {
                    tokens.extend(quote! {::concrete::tfhe::EncryptionKeyChoice::SMALL})
                }
            }
        }
    }

    impl ToTokens for CryptoParams {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let lwe_dimension = self.lwe_dimension;
            let glwe_dimension = self.glwe_dimension;
            let polynomial_size = self.polynomial_size;
            let pbs_base_log = self.pbs_base_log;
            let pbs_level = self.pbs_level;
            let lwe_noise_distribution = self.lwe_noise_distribution;
            let glwe_noise_distribution = self.glwe_noise_distribution;
            let encryption_key_choice = &self.encryption_key_choice;
            tokens.extend(quote! {
                ::concrete::tfhe::CryptoParams {
                    lwe_dimension: #lwe_dimension,
                    glwe_dimension: #glwe_dimension,
                    polynomial_size: #polynomial_size,
                    pbs_base_log: #pbs_base_log,
                    pbs_level: #pbs_level,
                    lwe_noise_distribution: #lwe_noise_distribution,
                    glwe_noise_distribution: #glwe_noise_distribution,
                    encryption_key_choice: #encryption_key_choice,
                }
            });
        }
    }

    impl ToTokens for IntegerType {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            let carry_width = self.carry_width;
            let msg_width = self.msg_width;
            let params = &self.params;
            let bit_width = &self.bit_width;
            let is_signed = &self.is_signed;
            tokens.extend(quote! {
                ::concrete::tfhe::IntegerType {
                    carry_width: #carry_width,
                    msg_width: #msg_width,
                    params: #params,
                    bit_width: #bit_width,
                    is_signed: #is_signed
                }
            });
        }
    }
}
