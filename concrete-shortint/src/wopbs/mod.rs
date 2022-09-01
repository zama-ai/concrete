//! Module with the definition of the WopbsKey (WithOut padding PBS Key).
//!
//! This module implements the generation of another server public key, which allows to compute
//! an alternative version of the programmable bootstrapping. This does not require the use of a
//! bit of padding.
//!
//! # WARNING: this module is experimental.

use crate::engine::ShortintEngine;
use crate::{Ciphertext, ClientKey, Parameters, ServerKey};
use concrete_core::backends::fftw::private::crypto::circuit_bootstrap::DeltaLog;
use concrete_core::commons::crypto::glwe::FunctionalPackingKeyswitchKey;
use concrete_core::commons::crypto::lwe::LweCiphertext;
use concrete_core::prelude::{
    AbstractEngine, EntityDeserializationEngine, EntitySerializationEngine,
    FftwFourierLweBootstrapKey64, FftwSerializationEngine,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(test)]
mod test;

// Struct for WoPBS based on the private functional packing keyswitch.
#[derive(Clone, Debug)]
pub struct WopbsKey {
    //Key for the private functional keyswitch
    pub(crate) vec_pfks_key: Vec<FunctionalPackingKeyswitchKey<Vec<u64>>>,
    pub(crate) small_bsk: FftwFourierLweBootstrapKey64,
    pub param: Parameters,
}

impl WopbsKey {
    /// Generates the server key required to compute a WoPBS from the client and the server keys.
    /// # Example
    ///
    /// ```rust
    /// use concrete_shortint::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_1_CARRY_1;
    /// use concrete_shortint::wopbs::*;
    /// use concrete_shortint::ciphertext::Ciphertext;
    /// use rand::Rng;
    ///
    /// // Generate the client key and the server key:
    /// let (mut cks, mut sks) = gen_keys(WOPBS_PARAM_MESSAGE_1_CARRY_1);
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// ```
    pub fn new_wopbs_key(cks: &ClientKey, sks: &ServerKey) -> WopbsKey {
        ShortintEngine::with_thread_local_mut(|engine| engine.new_wopbs_key_v0(cks, sks).unwrap())
    }

    /// Applies the Look-Up Table homomorphically using the WoPBS approach.
    ///
    /// #Warning: this assumes one bit of padding.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_shortint::ciphertext::Ciphertext;
    /// use concrete_shortint::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs::WOPBS_PARAM_MESSAGE_2_NORM2_2;
    /// use concrete_shortint::wopbs::*;
    /// use rand::Rng;
    ///
    /// // Generate the client key and the server key:
    /// let (mut cks, mut sks) = gen_keys(WOPBS_PARAM_MESSAGE_2_NORM2_2);
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut rng = rand::thread_rng();
    /// let mut ct = cks.encrypt(rng.gen::<u64>() % 2);
    /// let lut = vec![(1_u64 << 61); wopbs_key.param.polynomial_size.0];
    /// let ct_res = wopbs_key.circuit_bootstrap_vertical_packing(&mut sks, &mut ct, &vec![lut; 1]);
    /// let res = cks.decrypt_message_and_carry(&ct_res[0]);
    /// assert_eq!(res, 1);
    /// ```
    pub fn circuit_bootstrap_vertical_packing(
        &self,
        sks: &ServerKey,
        ct_in: &mut Ciphertext,
        lut: &[Vec<u64>],
    ) -> Vec<Ciphertext> {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine
                .circuit_bootstrap_vertical_packing(self, sks, ct_in, lut)
                .unwrap()
        })
    }

    /// Applies the Look-Up Table homomorphically using the WoPBS approach.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_shortint::ciphertext::Ciphertext;
    /// use concrete_shortint::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs::WOPBS_PARAM_MESSAGE_1_NORM2_2;
    /// use concrete_shortint::wopbs::*;
    /// use rand::Rng;
    ///
    /// let (mut cks, mut sks) = gen_keys(WOPBS_PARAM_MESSAGE_1_NORM2_2);
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut rng = rand::thread_rng();
    /// let mut ct = cks.encrypt_without_padding(rng.gen::<u64>() % 2);
    /// let lut = vec![(1_u64 << 63); wopbs_key.param.polynomial_size.0];
    /// let ct_res = wopbs_key.circuit_bootstrap_vertical_packing_without_padding(
    ///     &mut sks,
    ///     &mut ct,
    ///     &vec![lut; 1],
    /// );
    /// let res = cks.decrypt_message_and_carry_without_padding(&ct_res[0]);
    /// assert_eq!(res, 1);
    /// ```
    pub fn circuit_bootstrap_vertical_packing_without_padding(
        &self,
        sks: &ServerKey,
        ct_in: &mut Ciphertext,
        lut: &[Vec<u64>],
    ) -> Vec<Ciphertext> {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine
                .circuit_bootstrap_vertical_packing_without_padding(self, sks, ct_in, lut)
                .unwrap()
        })
    }

    /// Extracts the given number of bits from a ciphertext.
    ///
    /// # Warning Experimental
    pub fn extract_bit(
        &self,
        delta_log: DeltaLog,
        lwe_in: &mut LweCiphertext<Vec<u64>>,
        server_key: &ServerKey,
        number_values_to_extract: usize,
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine.extract_bit(delta_log, lwe_in, server_key, number_values_to_extract)
        })
    }

    /// Temporary wrapper used in `concrete-integer`.
    ///
    /// # Warning Experimental
    pub fn vertical_packing_cbs_binary_v0(
        &self,
        server_key: &ServerKey,
        vec_lut: Vec<Vec<u64>>,
        vec_lwe_in: &[LweCiphertext<Vec<u64>>],
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine.vertical_packing_cbs_binary_v0(self, server_key, vec_lut, vec_lwe_in)
        })
    }
}

#[derive(Serialize)]
struct SerializableWopbsKey<'a> {
    vec_pfks_key: &'a Vec<FunctionalPackingKeyswitchKey<Vec<u64>>>,
    small_bsk: Vec<u8>,
    param: Parameters,
}

#[derive(Deserialize)]
struct DeserializableWopbsKey {
    vec_pfks_key: Vec<FunctionalPackingKeyswitchKey<Vec<u64>>>,
    small_bsk: Vec<u8>,
    param: Parameters,
}

impl Serialize for WopbsKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fftw_ser_eng =
            FftwSerializationEngine::new(()).map_err(serde::ser::Error::custom)?;

        let small_bsk = fftw_ser_eng
            .serialize(&self.small_bsk)
            .map_err(serde::ser::Error::custom)?;

        SerializableWopbsKey {
            vec_pfks_key: &self.vec_pfks_key,
            small_bsk,
            param: self.param,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WopbsKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let thing =
            DeserializableWopbsKey::deserialize(deserializer).map_err(serde::de::Error::custom)?;
        let mut fftw_ser_eng =
            FftwSerializationEngine::new(()).map_err(serde::de::Error::custom)?;

        let small_bsk = fftw_ser_eng
            .deserialize(thing.small_bsk.as_slice())
            .map_err(serde::de::Error::custom)?;

        Ok(Self {
            vec_pfks_key: thing.vec_pfks_key,
            small_bsk,
            param: thing.param,
        })
    }
}
