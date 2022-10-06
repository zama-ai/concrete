//! Module with the definition of the WopbsKey (WithOut padding PBS Key).
//!
//! This module implements the generation of another server public key, which allows to compute
//! an alternative version of the programmable bootstrapping. This does not require the use of a
//! bit of padding.
//!
//! # WARNING: this module is experimental.

use crate::engine::ShortintEngine;
use crate::{Ciphertext, ClientKey, Parameters, ServerKey};

use concrete_core::prelude::*;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(test)]
mod test;

// Struct for WoPBS based on the private functional packing keyswitch.
#[derive(Clone, Debug)]
pub struct WopbsKey {
    //Key for the private functional keyswitch
    pub(crate) small_bsk: FftFourierLweBootstrapKey64,
    pub(crate) cbs_pfpksk: LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64,
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
    ///
    /// // Generate the client key and the server key:
    /// let (mut cks, mut sks) = gen_keys(WOPBS_PARAM_MESSAGE_1_CARRY_1);
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// ```
    pub fn new_wopbs_key(cks: &ClientKey, sks: &ServerKey) -> WopbsKey {
        ShortintEngine::with_thread_local_mut(|engine| engine.new_wopbs_key(cks, sks).unwrap())
    }

    /// Generates the Look-Up Table homomorphically using the WoPBS approach.
    ///
    /// # Warning: this assumes one bit of padding.
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
    /// let message_modulus = WOPBS_PARAM_MESSAGE_2_NORM2_2.message_modulus.0 as u64;
    /// let m = 2;
    /// let mut ct = cks.encrypt(m);
    /// let lut = wopbs_key.generate_lut(&ct, |x| x * x % message_modulus);
    /// let ct_res = wopbs_key.programmable_bootstrapping(&mut sks, &mut ct, &lut);
    /// let res = cks.decrypt(&ct_res[0]);
    /// assert_eq!(res, (m * m) % message_modulus);
    /// ```
    pub fn generate_lut<F>(&self, ct: &Ciphertext, f: F) -> Vec<u64>
    where
        F: Fn(u64) -> u64,
    {
        // The function is applied only on the message modulus bits
        let basis = ct.message_modulus.0 * ct.carry_modulus.0;
        let delta = 64 - f64::log2((basis) as f64).ceil() as u64 - 1;
        let poly_size = self.small_bsk.polynomial_size().0;
        let mut vec_lut = vec![0; poly_size];
        for i in 0..basis {
            vec_lut[i] = f((i % ct.message_modulus.0) as u64) << delta;
        }
        vec_lut
    }

    /// Generates the Look-Up Table homomorphically using the WoPBS approach.
    ///
    /// # Warning: this assumes no bit of padding.
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
    /// let message_modulus = WOPBS_PARAM_MESSAGE_2_NORM2_2.message_modulus.0 as u64;
    /// let m = 2;
    /// let mut ct = cks.encrypt_without_padding(m);
    /// let lut = wopbs_key.generate_lut(&ct, |x| x * x % message_modulus);
    /// let ct_res = wopbs_key.programmable_bootstrapping_without_padding(&mut sks, &mut ct, &lut);
    /// let res = cks.decrypt_without_padding(&ct_res[0]);
    /// assert_eq!(res, (m * m) % message_modulus);
    /// ```
    pub fn generate_lut_without_padding<F>(&self, ct: &Ciphertext, f: F) -> Vec<u64>
    where
        F: Fn(u64) -> u64,
    {
        // The function is applied only on the message modulus bits
        let basis = ct.message_modulus.0 * ct.carry_modulus.0;
        let delta = 64 - f64::log2((basis) as f64).ceil() as u64;
        let poly_size = self.small_bsk.polynomial_size().0;
        let mut vec_lut = vec![0; poly_size];
        for i in 0..basis {
            vec_lut[i] = f((i % ct.message_modulus.0) as u64) << delta;
        }
        vec_lut
    }

    /// Generates the Look-Up Table homomorphically using the WoPBS approach.
    ///
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_shortint::ciphertext::Ciphertext;
    /// use concrete_shortint::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs::WOPBS_PARAM_MESSAGE_3_NORM2_2;
    /// use concrete_shortint::wopbs::*;
    /// use rand::Rng;
    ///
    /// // Generate the client key and the server key:
    /// let (mut cks, mut sks) = gen_keys(WOPBS_PARAM_MESSAGE_3_NORM2_2);
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let message_modulus = 5;
    /// let m = 2;
    /// let mut ct = cks.encrypt_with_message_modulus_not_power_of_two(m, message_modulus);
    /// let lut = wopbs_key.generate_lut_without_padding_crt(&ct, |x| x * x % message_modulus as u64);
    /// let ct_res = wopbs_key.programmable_bootstrapping_without_padding_crt(&mut sks, &mut ct, &lut);
    /// let res = cks.decrypt_message_and_carry_not_power_of_two(&ct_res[0], message_modulus);
    /// assert_eq!(res, (m * m) % message_modulus as u64);
    /// ```
    pub fn generate_lut_native_crt<F>(&self, ct: &Ciphertext, f: F) -> Vec<u64>
    where
        F: Fn(u64) -> u64,
    {
        // The function is applied only on the message modulus bits
        let basis = ct.message_modulus.0 * ct.carry_modulus.0;
        let nb_bit = f64::log2((basis) as f64).ceil() as u64;
        let poly_size = self.small_bsk.polynomial_size().0;
        let mut vec_lut = vec![0; poly_size];
        for i in 0..basis {
            let index_lut = (((i as u64 % basis as u64) << nb_bit) / basis as u64) as usize;
            //println!("val = {}, lut index = {}", i, index_lut);
            vec_lut[index_lut] =
                (((f(i as u64) % basis as u64) as u128 * (1 << 64)) / basis as u128) as u64;
        }
        vec_lut
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
    /// let (cks, sks) = gen_keys(WOPBS_PARAM_MESSAGE_2_NORM2_2);
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut rng = rand::thread_rng();
    /// let message_modulus = WOPBS_PARAM_MESSAGE_2_NORM2_2.message_modulus.0;
    /// let ct = cks.encrypt(rng.gen::<u64>() % message_modulus);
    /// let lut = vec![(1_u64 << 61); wopbs_key.param.polynomial_size.0];
    /// let ct_res = wopbs_key.programmable_bootstrapping(&sks, &ct, &lut);
    /// let res = cks.decrypt_message_and_carry(&ct_res);
    /// assert_eq!(res, 1);
    /// ```
    pub fn programmable_bootstrapping(
        &self,
        sks: &ServerKey,
        ct_in: &Ciphertext,
        lut: &Vec<u64>,
    ) -> Ciphertext {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine
                .programmable_bootstrapping(self, sks, ct_in, &lut)
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
    /// let (cks, sks) = gen_keys(WOPBS_PARAM_MESSAGE_1_NORM2_2);
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut rng = rand::thread_rng();
    /// let ct = cks.encrypt_without_padding(rng.gen::<u64>() % 2);
    /// let lut = vec![(1_u64 << 63); wopbs_key.param.polynomial_size.0];
    /// let ct_res = wopbs_key.programmable_bootstrapping_without_padding(&sks, &ct, &lut);
    /// let res = cks.decrypt_message_and_carry_without_padding(&ct_res);
    /// assert_eq!(res, 1);
    /// ```
    pub fn programmable_bootstrapping_without_padding(
        &self,
        sks: &ServerKey,
        ct_in: &Ciphertext,
        lut: &Vec<u64>,
    ) -> Ciphertext {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine
                .programmable_bootstrapping_without_padding(self, sks, ct_in, lut)
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
    /// use concrete_shortint::parameters::parameters_wopbs::WOPBS_PARAM_MESSAGE_3_NORM2_2;
    /// use concrete_shortint::wopbs::*;
    ///
    /// let (cks, sks) = gen_keys(WOPBS_PARAM_MESSAGE_3_NORM2_2);
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let msg = 2;
    /// let modulus = 5;
    /// let mut ct = cks.encrypt_with_message_modulus_not_power_of_two(msg, modulus);
    /// let lut = wopbs_key.generate_lut_without_padding_crt(&ct, |x| x);
    /// let ct_res = wopbs_key.programmable_bootstrapping_without_padding_crt(&sks, &mut ct, &lut);
    /// let res = cks.decrypt_message_and_carry_not_power_of_two(&ct_res, modulus);
    /// assert_eq!(res, msg);
    /// ```
    pub fn programmable_bootstrapping_native_crt(
        &self,
        sks: &ServerKey,
        ct_in: &mut Ciphertext,
        lut: &Vec<u64>,
    ) -> Ciphertext {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine
                .programmable_bootstrapping_native_crt(self, sks, ct_in, lut)
                .unwrap()
        })
    }

    /// Extracts the given number of bits from a ciphertext.
    ///
    /// # Warning Experimental
    pub fn extract_bits(
        &self,
        delta_log: DeltaLog,
        ciphertext: &Ciphertext,
        server_key: &ServerKey,
        num_bits_to_extract: usize,
    ) -> LweCiphertextVector64 {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine
                .extract_bits(
                    delta_log,
                    &ciphertext.ct,
                    server_key,
                    ExtractedBitsCount(num_bits_to_extract),
                )
                .unwrap()
        })
    }

    /// Temporary wrapper used in `concrete-integer`.
    ///
    /// # Warning Experimental
    pub fn circuit_bootstrapping_vertical_packing(
        &self,
        server_key: &ServerKey,
        vec_lut: Vec<Vec<u64>>,
        extracted_bits_blocks: Vec<LweCiphertextVector64>,
    ) -> Vec<LweCiphertext64> {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine.circuit_bootstrapping_vertical_packing(
                self,
                server_key,
                vec_lut,
                extracted_bits_blocks,
            )
        })
    }
}

#[derive(Serialize, Deserialize)]
struct SerializableWopbsKey {
    small_bsk: Vec<u8>,
    cbs_pfpksk: Vec<u8>,
    param: Parameters,
}

impl Serialize for WopbsKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fft_ser_eng =
            FftSerializationEngine::new(()).map_err(serde::ser::Error::custom)?;

        let mut default_ser_eng =
            DefaultSerializationEngine::new(()).map_err(serde::ser::Error::custom)?;

        let small_bsk = fft_ser_eng
            .serialize(&self.small_bsk)
            .map_err(serde::ser::Error::custom)?;

        let cbs_pfpksk = default_ser_eng
            .serialize(&self.cbs_pfpksk)
            .map_err(serde::ser::Error::custom)?;

        SerializableWopbsKey {
            small_bsk,
            cbs_pfpksk,
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
            SerializableWopbsKey::deserialize(deserializer).map_err(serde::de::Error::custom)?;

        let mut fft_ser_eng =
            FftSerializationEngine::new(()).map_err(serde::de::Error::custom)?;

        let mut default_ser_eng =
            DefaultSerializationEngine::new(()).map_err(serde::de::Error::custom)?;

        let small_bsk = fft_ser_eng
            .deserialize(thing.small_bsk.as_slice())
            .map_err(serde::de::Error::custom)?;

        let cbs_pfpksk = default_ser_eng
            .deserialize(thing.cbs_pfpksk.as_slice())
            .map_err(serde::de::Error::custom)?;

        Ok(Self {
            small_bsk,
            cbs_pfpksk,
            param: thing.param,
        })
    }
}
