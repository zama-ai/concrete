//! Module with the definition of the WopbsKey (WithOut padding PBS Key).
//!
//! This module implements the generation of another server public key, which allows to compute
//! an alternative version of the programmable bootstrapping. This does not require the use of a
//! bit of padding.
//!
//! # WARNING: this module is experimental.

#[cfg(test)]
mod test;

use crate::{ClientKey, CrtCiphertext, IntegerCiphertext, RadixCiphertext, ServerKey};
use concrete_core::backends::fftw::private::crypto::circuit_bootstrap::DeltaLog;
use concrete_core::backends::fftw::private::crypto::wop_pbs_vp::extract_bit_v0_v1;
use concrete_core::commons::crypto::lwe::LweCiphertext;
use concrete_core::prelude::LweCiphertext64;
use concrete_shortint::Ciphertext;
use concrete_shortint::ciphertext::Degree;

pub struct WopbsKey {
    wopbs_key: concrete_shortint::wopbs::WopbsKey,
}

impl WopbsKey {
    /// Generates the server key required to compute a WoPBS from the client and the server keys.
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry
    /// ::WOPBS_PARAM_MESSAGE_1_CARRY_1;
    /// use concrete_integer::wopbs::*;
    ///
    /// // Generate the client key and the server key:
    /// let block = 2;
    /// let (mut cks, mut sks) = gen_keys(&WOPBS_PARAM_MESSAGE_1_CARRY_1);
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// ```
    pub fn new_wopbs_key(cks: &ClientKey, sks: &ServerKey) -> WopbsKey {
        WopbsKey {
            wopbs_key: concrete_shortint::wopbs::WopbsKey::new_wopbs_key(&cks.key, &sks.key),
        }
    }

    pub fn new_from_shortint(wopbskey: &concrete_shortint::wopbs::WopbsKey) -> WopbsKey {
        let key = wopbskey.clone();
        WopbsKey { wopbs_key: key }
    }

    /// # Example
    ///
    /// ```rust
    ///   //TODO
    /// ```
    pub fn wopbs(
        &self,
        sks: &ServerKey,
        ct_in: &mut RadixCiphertext,
        lut: &[Vec<u64>],
    ) -> RadixCiphertext {
        let mut vec_lwe: Vec<LweCiphertext<Vec<u64>>> = vec![];
        let mut ct_in = ct_in.clone();
        // Extraction of each bit for each block
        for block in ct_in.blocks.iter_mut() {
            let delta = (1_usize << 63) / (block.message_modulus.0 * block.carry_modulus.0);
            let delta_log = DeltaLog(f64::log2(delta as f64) as usize);
            let nb_bit_to_extract =
                f64::log2((block.message_modulus.0 * block.carry_modulus.0) as f64) as usize;

            let mut tmp =
                self.wopbs_key
                    .extract_bit(delta_log, &mut block.ct.0, &sks.key, nb_bit_to_extract);
            vec_lwe.append(&mut tmp);
        }

        vec_lwe.reverse();

        let vec_ct_out = self.wopbs_key.vertical_packing_cbs_binary_v0(
            &sks.key,
            lut.to_vec(),
            vec_lwe.as_slice(),
        );

        let mut ct_vec_out: Vec<concrete_shortint::Ciphertext> = vec![];
        for (block, block_out) in ct_in.blocks.iter().zip(vec_ct_out.into_iter()) {
            ct_vec_out.push(concrete_shortint::Ciphertext {
                ct: LweCiphertext64(block_out),
                degree: Degree(block.message_modulus.0 - 1),
                message_modulus: block.message_modulus,
                carry_modulus: block.carry_modulus,
            });
        }

        RadixCiphertext { blocks: ct_vec_out }
    }


    pub fn wopbs_without_padding<T>(
        &self,
        sks: &ServerKey,
        ct_in: &T,
        lut: &[Vec<u64>],
    ) -> T where
        T:IntegerCiphertext
    {
        let mut vec_lwe: Vec<LweCiphertext<Vec<u64>>> = vec![];
        let mut ct_in = ct_in.clone();
        // Extraction of each bit for each block
        for block in ct_in.blocks_mut().iter_mut() {
            let delta = (1_usize << 63) / (block.message_modulus.0 * block.carry_modulus.0 / 2);
            let delta_log = DeltaLog(f64::log2(delta as f64) as usize);
            let nb_bit_to_extract =
                f64::log2((block.message_modulus.0 * block.carry_modulus.0) as f64) as usize;

            let mut tmp =
                self.wopbs_key
                    .extract_bit(delta_log, &mut block.ct.0, &sks.key, nb_bit_to_extract);
            vec_lwe.append(&mut tmp);
        }

        vec_lwe.reverse();

        let vec_ct_out = self.wopbs_key.vertical_packing_cbs_binary_v0(
            &sks.key,
            lut.to_vec(),
            vec_lwe.as_slice(),
        );

        let mut ct_vec_out: Vec<concrete_shortint::Ciphertext> = vec![];
        for (block, block_out) in ct_in.blocks().iter().zip(vec_ct_out.into_iter()) {
            ct_vec_out.push(concrete_shortint::Ciphertext {
                ct: LweCiphertext64(block_out),
                degree: Degree(block.message_modulus.0 - 1),
                message_modulus: block.message_modulus,
                carry_modulus: block.carry_modulus,
            });
        }
        T::from_blocks(ct_vec_out)
    }

    /// # Example
    ///
    /// ```rust
    ///
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// use concrete_integer::wopbs::*;
    ///
    /// let param = WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// let nb_block = 3;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&param);
    /// //Generate wopbs_v0 key
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    ///
    /// let clear = 15;
    /// let mut ct = cks.encrypt_radix(clear as u64, nb_block);
    /// let lut = wopbs_key.generate_lut(&ct, |x| 2 * x);
    /// let ct_res = wopbs_key.wopbs(&sks, &mut ct, &lut);
    ///  let res = cks.decrypt_radix(&ct_res);
    ///
    /// assert_eq!(res, clear * 2)
    /// ```
    pub fn generate_lut<F, T>(&self, ct: &T, f: F) -> Vec<Vec<u64>>
        where
            F: Fn(u64) -> u64,
            T: IntegerCiphertext,
    {
        let log_message_modulus = f64::log2((self.wopbs_key.param.message_modulus.0) as f64) as u64;
        let log_carry_modulus = f64::log2((self.wopbs_key.param.carry_modulus.0) as f64) as u64;
        let log_basis = log_message_modulus + log_carry_modulus;
        let delta = 64 - log_basis - 1;
        let nb_block = ct.blocks().len();
        let poly_size  = self.wopbs_key.param.polynomial_size.0;

        let mut lut_size = 1 << (nb_block * log_basis as usize);
        if lut_size < poly_size {
            lut_size = poly_size;
        }
        let mut vec_lut = vec![vec![0; lut_size]; nb_block];

        for index in 0..lut_size{

            // find the value represented by the index
            let mut value = 0;
            let mut tmp_index = index;
            for i in 0..nb_block as u64{
                let tmp = tmp_index % (1<<(log_basis * (i+1)));
                tmp_index -= tmp;
                value += tmp >> (log_carry_modulus * i);
            }

            // fill the LUTs
            for block in 0..nb_block{
                vec_lut[block][index] = ((f(value as u64) >> (log_carry_modulus * block as u64)) %
                    (1<<log_message_modulus)) << delta
            }
        }
        vec_lut
    }

    /// # Example
    ///
    /// ```rust
    ///
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// use concrete_integer::wopbs::*;
    ///
    /// let param = WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// let nb_block = 3;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&param);
    /// //Generate wopbs_v0 key
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    ///
    /// let clear = 15;
    /// let mut ct = cks.encrypt_without_padding(clear as u64);
    /// let lut = wopbs_key.generate_lut_without_padding(&ct, |x| 2 * x);
    /// let ct_res = wopbs_key.wopbs_without_padding(&sks, &mut ct, &lut);
    ///  let res = cks.decrypt_without_padding(&ct_res);
    ///
    /// assert_eq!(res, clear * 2)
    /// ```
    pub fn generate_lut_without_padding<F, T>(&self, ct: &T, f: F) -> Vec<Vec<u64>>
        where
            F: Fn(u64) -> u64,
            T: IntegerCiphertext
    {
        let log_message_modulus = f64::log2((self.wopbs_key.param.message_modulus.0) as f64) as u64;
        let log_carry_modulus = f64::log2((self.wopbs_key.param.carry_modulus.0) as f64) as u64;
        let log_basis = log_message_modulus + log_carry_modulus;
        let delta = 64 - log_basis;
        let nb_block = ct.blocks().len();
        let poly_size  = self.wopbs_key.param.polynomial_size.0;

        let mut lut_size = 1 << (nb_block * log_basis as usize);
        if lut_size < poly_size {
            lut_size = poly_size;
        }
        let mut vec_lut = vec![vec![0; lut_size]; nb_block];

        for index in 0..lut_size{

            // find the value represented by the index
            let mut value = 0;
            let mut tmp_index = index;
            for i in 0..nb_block as u64{
                let tmp = tmp_index % (1<<(log_basis * (i+1)));
                tmp_index -= tmp;
                value += tmp >> (log_carry_modulus * i);
            }

            // fill the LUTs
            for block in 0..nb_block{
                vec_lut[block][index] = ((f(value as u64) >> (log_carry_modulus * block as u64)) %
                    (1<<log_message_modulus)) << delta
            }
        }
        vec_lut
    }

    /// # Example
    ///
    /// ```rust
    ///
    /// use concrete_integer::gen_keys;
    /// use concrete_integer::parameters::PARAM_4_BITS_5_BLOCKS;
    /// use concrete_integer::wopbs::*;
    ///
    /// let basis : Vec<u64> = vec![9,11,13];
    /// let nb_block = basis.len();
    ///
    /// let param = PARAM_4_BITS_5_BLOCKS;
    /// //Generate the client key and the server key:
    /// let (mut cks, sks) = gen_keys(&param);
    /// let wopbs_key =  WopbsKey::new_wopbs_key(&cks, &sks);
    ///
    /// let mut msg_space = 1;
    /// for modulus in basis.iter() {
    ///     msg_space *= modulus;
    /// }
    /// let clear = 42;    // Encrypt the integers
    /// let mut ct = cks.encrypt_crt_not_power_of_two(clear, basis.clone());
    /// let lut = wopbs_key.generate_lut_crt_without_padding(&ct, |x| x);
    /// let ct_res = wopbs_key.circuit_bootstrap_vertical_packing_v0_without_padding(&sks, &mut ct, &lut);
    /// let res = cks.decrypt_crt_not_power_of_two(&ct_res);
    /// assert_eq!(res, clear);
    /// ```
    pub fn generate_lut_crt_without_padding<F>(&self, ct: &CrtCiphertext, f:F) -> Vec<Vec<u64>>
    where
    F: Fn(u64) -> u64,
    {
        let mut bit = vec![];
        let mut total_bit = 0;
        let mut modulus = 1;
        let mut basis: Vec<_> = ct.blocks.iter().map(|x| x.message_modulus.0 as u64).collect();
        basis.reverse();
        for i in basis.iter() {
            modulus *= i;
            let b = f64::log2(*i as f64).ceil() as u64;
            total_bit += b;
            bit.push(b);
        }
        let mut lut_size = 1 << total_bit;
        if 1 << total_bit < self.wopbs_key.param.polynomial_size.0 as u64 {
            lut_size = self.wopbs_key.param.polynomial_size.0;
        }
        let mut vec_lut = vec![vec![0; lut_size]; basis.len()];

        for value in 0..modulus {
            let mut index_lut = 0;
            let mut tmp = 1 << total_bit;
            for (base, bit) in basis.iter().zip(bit.iter()) {
                tmp >>= bit;
                index_lut += (((value % base) << bit) / base) * tmp;
            }
            for (j, b) in basis.iter().enumerate() {
                vec_lut[basis.len() - 1 - j][index_lut as usize] = (((f(value) % b) as u128 *
                    (1 << 64)) /
                    *b as
                        u128) as u64
            }
        }
        vec_lut
    }

    pub fn circuit_bootstrap_vertical_packing_v0_without_padding<T>(
        &self,
        sks: &ServerKey,
        ct_in: &T,
        lut: &[Vec<u64>],
    ) -> T where T:IntegerCiphertext{
        let mut vec_lwe: Vec<LweCiphertext<Vec<u64>>> = vec![];
        let mut ct_in = ct_in.clone();

        // Extraction of each bit for each block
        for block in ct_in.blocks_mut().iter_mut() {
            let nb_bit_to_extract =
                f64::log2((block.message_modulus.0 * block.carry_modulus.0) as f64).ceil() as
                    usize;
            let delta_log = DeltaLog(64 - nb_bit_to_extract);


            // trick ( ct - delta/2 + delta/2^4  )
            let lwe_size = block.ct.0.lwe_size().0;
            let mut cont = vec![0; lwe_size];
            cont[lwe_size - 1] = (1 << (64 - nb_bit_to_extract - 1)) - (1 << (64 -
                nb_bit_to_extract - 5));
            let tmp = LweCiphertext::from_container(cont);
            block.ct.0.update_with_sub(&tmp);

            concrete_shortint::engine::ShortintEngine::with_thread_local_mut
                (|engine| {
                    let (buffers, _, _) = engine.buffers_for_key(&sks.key);
                    let mut vec_lwe_tmp = extract_bit_v0_v1(
                        delta_log,
                        &mut block.ct.0,
                        &sks.key.key_switching_key.0,
                        &sks.key.bootstrapping_key.0,
                        &mut buffers.fourier,
                        nb_bit_to_extract,
                    );
                    vec_lwe.append(&mut vec_lwe_tmp);
                });
        }

        vec_lwe.reverse();

        let vec_ct_out = self.wopbs_key.vertical_packing_cbs_binary_v0(
            &sks.key,
            lut.to_vec(),
            vec_lwe.as_slice(),
        );

        let mut ct_vec_out: Vec<concrete_shortint::Ciphertext> = vec![];
        for (block, block_out) in ct_in.blocks().iter().zip(vec_ct_out.into_iter()) {
            ct_vec_out.push(concrete_shortint::Ciphertext {
                ct: LweCiphertext64(block_out),
                degree: Degree(block.message_modulus.0 - 1),
                message_modulus: block.message_modulus,
                carry_modulus: block.carry_modulus,
            });
        }
        T::from_blocks(ct_vec_out)
    }
}
