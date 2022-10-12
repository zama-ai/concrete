//! Module with the definition of the WopbsKey (WithOut padding PBS Key).
//!
//! This module implements the generation of another server public key, which allows to compute
//! an alternative version of the programmable bootstrapping. This does not require the use of a
//! bit of padding.
//!
//! # WARNING: this module is experimental.

#[cfg(test)]
mod test;

use crate::client_key::utils::i_crt;
use crate::{ClientKey, CrtCiphertext, IntegerCiphertext, RadixCiphertext, ServerKey};
use concrete_core::prelude::*;
use concrete_shortint::ciphertext::Degree;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct WopbsKey {
    wopbs_key: concrete_shortint::wopbs::WopbsKey,
}

//TODO: Move to a different file?
// Update to pub(crate)
/// ```rust
/// use concrete_integer::wopbs::{decode_radix, encode_radix};
///
/// let val = 11;
/// let basis = 2;
/// let nb_block = 5;
/// let radix = encode_radix(val, basis, nb_block);
///
/// assert_eq!(val, decode_radix(radix, basis));
/// ```
pub fn encode_radix(val: u64, basis: u64, nb_block: u64) -> Vec<u64> {
    let mut output = vec![];
    //Bits of message put to 1éfé
    let mask = (basis - 1) as u64;

    let mut power = 1_u64;
    //Put each decomposition into a new ciphertext
    for _ in 0..nb_block {
        let mut decomp = val & (mask * power);
        decomp /= power;

        // fill the vector with the message moduli
        output.push(decomp);

        //modulus to the power i
        power *= basis;
    }
    output
}

pub fn encode_crt(val: u64, basis: &Vec<u64>) -> Vec<u64> {
    let mut output = vec![];
    //Put each decomposition into a new ciphertext
    for i in 0..basis.len() {
        output.push(val % basis[i]);
    }
    output
}

//Concatenate two ciphertexts in one
//Used to compute bivariate wopbs
fn ciphertext_concatenation<T>(ct1: &T, ct2: &T) -> T
where
    T: IntegerCiphertext,
{
    let mut new_blocks = ct1.blocks().to_vec();
    new_blocks.extend_from_slice(ct2.blocks());
    T::from_blocks(new_blocks)
}

//TODO: Move to a different file?
// Update to pub(crate)
// split the value following the degree and the modulus
/// ```rust
/// ```
pub fn encode_mix_radix(mut val: u64, basis: &Vec<u64>, modulus: u64) -> Vec<u64> {
    let mut output = vec![];
    for basis in basis.iter() {
        output.push(val % modulus);
        val -= val % modulus;
        let tmp = (val % (1 << basis)) >> (f64::log2(modulus as f64) as u64);
        val >>= basis;
        val += tmp;
    }
    output
}

// Example: val = 5 = 0b101 , basis = [1,2] -> output = [1, 1]
/// ```rust
/// use concrete_integer::wopbs::split_value_according_to_bit_basis;
/// // Generate the client key and the server key:
/// let val = 5;
/// let basis = vec![1, 2];
/// assert_eq!(vec![1, 2], split_value_according_to_bit_basis(val, &basis));
/// ```
pub fn split_value_according_to_bit_basis(value: u64, basis: &Vec<u64>) -> Vec<u64> {
    let mut output = vec![];
    let mut tmp = value.clone();
    let mask = 1;

    for i in 0..basis.len() {
        let mut tmp_output = 0;
        for j in 0..basis[i] {
            let val = tmp & mask;
            tmp_output += val << j;
            tmp >>= 1;
        }
        output.push(tmp_output);
    }
    output
}

//TODO: Move to a different file?
// Update to pub(crate)
/// ```rust
/// use concrete_integer::wopbs::{decode_radix, encode_radix};
///
/// let val = 11;
/// let basis = 2;
/// let nb_block = 5;
/// assert_eq!(val, decode_radix(encode_radix(val, basis, nb_block), basis));
/// ```
pub fn decode_radix(val: Vec<u64>, basis: u64) -> u64 {
    let mut result = 0_u64;
    let mut shift = 1_u64;
    for v_i in val.iter() {
        //decrypt the component i of the integer and multiply it by the radix product
        let tmp = v_i.wrapping_mul(shift);

        // update the result
        result = result.wrapping_add(tmp);

        // update the shift for the next iteration
        shift = shift.wrapping_mul(basis);
    }
    result
}


impl From<concrete_shortint::wopbs::WopbsKey> for WopbsKey {
    fn from(wopbs_key: concrete_shortint::wopbs::WopbsKey) -> Self {
        Self {
            wopbs_key
        }
    }
}

impl WopbsKey {
    /// Generates the server key required to compute a WoPBS from the client and the server keys.
    /// # Example
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_1_CARRY_1;
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

    /// Computes the WoP-PBS given the luts.
    ///
    /// This works for both RadixCiphertext and CrtCiphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// use concrete_integer::wopbs::*;
    ///
    /// let nb_block = 3;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&WOPBS_PARAM_MESSAGE_2_CARRY_2);
    ///
    /// //Generate wopbs_v0 key
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut moduli = 1_u64;
    /// for _ in 0..nb_block{
    ///     moduli *= cks.parameters().message_modulus.0 as u64;
    /// }
    /// let clear = 42 % moduli;
    /// let mut ct = cks.encrypt_radix(clear as u64, nb_block);
    /// let lut = wopbs_key.generate_lut(&ct, |x|x);
    /// let ct_res = wopbs_key.wopbs_with_degree(&sks, &mut ct, &lut);
    /// let res = cks.decrypt_radix(&ct_res);
    ///
    ///  assert_eq!(res, clear);
    /// ```
    pub fn wopbs<T>(&self, sks: &ServerKey, ct_in: &T, lut: &[Vec<u64>]) -> T
    where
        T: IntegerCiphertext,
    {
        let mut extracted_bits_blocks = vec![];
        // Extraction of each bit for each block
        for block in ct_in.blocks().iter() {
            let delta = (1_usize << 63)
                / (self.wopbs_key.param.message_modulus.0 * self.wopbs_key.param.carry_modulus.0);
            let delta_log = DeltaLog(f64::log2(delta as f64) as usize);
            let nb_bit_to_extract = f64::log2((block.degree.0 + 1) as f64).ceil() as usize;
            let extracted_bits =
                self.wopbs_key
                    .extract_bits(delta_log, block, &sks.key, nb_bit_to_extract);

            extracted_bits_blocks.push(extracted_bits);
        }

        extracted_bits_blocks.reverse();
        let vec_ct_out = self.wopbs_key.circuit_bootstrapping_vertical_packing(
            &sks.key,
            lut.to_vec(),
            extracted_bits_blocks,
        );

        let mut ct_vec_out = vec![];
        for (block, block_out) in ct_in.blocks().iter().zip(vec_ct_out.into_iter()) {
            ct_vec_out.push(concrete_shortint::Ciphertext {
                ct: block_out,
                degree: Degree(block.message_modulus.0 - 1),
                message_modulus: block.message_modulus,
                carry_modulus: block.carry_modulus,
            });
        }
        T::from_blocks(ct_vec_out)
    }

    /// # Example
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_integer::wopbs::WopbsKey;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2;
    ///
    /// let param = WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// let nb_block = 3;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&param);
    /// //Generate wopbs_v0 key
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut moduli = 1_u64;
    /// for _ in 0..nb_block{
    ///     moduli *= cks.parameters().message_modulus.0 as u64;
    /// }
    /// let clear = 15 % moduli;
    /// let mut ct = cks.encrypt_radix_without_padding(clear as u64, nb_block);
    /// let lut = wopbs_key.generate_lut_without_padding(&ct, |x| 2 * x);
    /// let ct_res = wopbs_key.wopbs_without_padding(&sks, &mut ct, &lut);
    /// let res = cks.decrypt_radix_without_padding(&ct_res);
    ///
    /// assert_eq!(res, (clear * 2) % moduli)
    /// ```
    pub fn wopbs_without_padding<T>(&self, sks: &ServerKey, ct_in: &T, lut: &[Vec<u64>]) -> T
    where
        T: IntegerCiphertext,
    {
        let mut extracted_bits_blocks = vec![];
        let mut ct_in = ct_in.clone();
        // Extraction of each bit for each block
        for block in ct_in.blocks_mut().iter_mut() {
            let delta = (1_usize << 63) / (block.message_modulus.0 * block.carry_modulus.0 / 2);
            let delta_log = DeltaLog(f64::log2(delta as f64) as usize);
            let nb_bit_to_extract =
                f64::log2((block.message_modulus.0 * block.carry_modulus.0) as f64) as usize;

            let extracted_bits =
                self.wopbs_key
                    .extract_bits(delta_log, block, &sks.key, nb_bit_to_extract);
            extracted_bits_blocks.push(extracted_bits);
        }

        extracted_bits_blocks.reverse();

        let vec_ct_out = self.wopbs_key.circuit_bootstrapping_vertical_packing(
            &sks.key,
            lut.to_vec(),
            extracted_bits_blocks,
        );

        let mut ct_vec_out = vec![];
        for (block, block_out) in ct_in.blocks().iter().zip(vec_ct_out.into_iter()) {
            ct_vec_out.push(concrete_shortint::Ciphertext {
                ct: block_out,
                degree: Degree(block.message_modulus.0 - 1),
                message_modulus: block.message_modulus,
                carry_modulus: block.carry_modulus,
            });
        }
        T::from_blocks(ct_vec_out)
    }

    /// WOPBS for 'true' CRT
    /// # Example
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_integer::parameters::PARAM_4_BITS_5_BLOCKS;
    /// use concrete_integer::wopbs::WopbsKey;
    ///
    /// let basis: Vec<u64> = vec![9, 11];
    ///
    /// let param = PARAM_4_BITS_5_BLOCKS;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&param);
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    ///
    /// let mut msg_space = 1;
    /// for modulus in basis.iter() {
    ///     msg_space *= modulus;
    /// }
    /// let clear = 42 % msg_space; // Encrypt the integers
    /// let mut ct = cks.encrypt_crt_not_power_of_two(clear, basis.clone());
    /// let lut = wopbs_key.generate_lut_crt_without_padding(&ct, |x| x);
    /// let ct_res = wopbs_key.wopbs_not_power_of_two(&sks, &mut ct, &lut);
    /// let res = cks.decrypt_crt_not_power_of_two(&ct_res);
    /// assert_eq!(res, clear);
    /// ```
    pub fn wopbs_not_power_of_two(
        &self,
        sks: &ServerKey,
        ct1: &CrtCiphertext,
        lut: &[Vec<u64>],
    ) -> CrtCiphertext {
        self.circuit_bootstrap_vertical_packing_native_crt(sks, &[ct1.clone()], lut)
    }

    /// # Example
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// use concrete_integer::wopbs::*;
    ///
    /// let nb_block = 3;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&WOPBS_PARAM_MESSAGE_2_CARRY_2);
    ///
    /// //Generate wopbs_v0 key    ///
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut moduli = 1_u64;
    /// for _ in 0..nb_block{
    ///     moduli *= cks.parameters().message_modulus.0 as u64;
    /// }
    /// let clear1 = 42 % moduli;
    /// let clear2 = 24 % moduli;
    /// let mut ct1 = cks.encrypt_radix(clear1 as u64, nb_block);
    /// let mut ct2 = cks.encrypt_radix(clear2 as u64, nb_block);
    ///
    /// let lut = wopbs_key.generate_lut_bivariate_radix(&ct1, &ct2, |x,y| 2 * x * y);
    /// let ct_res = wopbs_key.bivariate_wopbs_with_degree(&sks, &mut ct1, &mut ct2, &lut);
    /// let res = cks.decrypt_radix(&ct_res);
    ///
    ///  assert_eq!(res, (2 * clear1 * clear2) % moduli);
    /// ```
    pub fn bivariate_wopbs_with_degree<T>(
        &self,
        sks: &ServerKey,
        ct1: &T,
        ct2: &T,
        lut: &[Vec<u64>],
    ) -> T
    where
        T: IntegerCiphertext,
    {
        let ct = ciphertext_concatenation(ct1, ct2);
        self.wopbs(sks, &ct, lut)
    }

    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// use concrete_integer::wopbs::*;
    ///
    /// let nb_block = 3;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&WOPBS_PARAM_MESSAGE_2_CARRY_2);
    ///
    /// //Generate wopbs_v0 key    ///
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut moduli = 1_u64;
    /// for _ in 0..nb_block{
    ///     moduli *= cks.parameters().message_modulus.0 as u64;
    /// }
    /// let clear = 42 % moduli;
    /// let mut ct = cks.encrypt_radix(clear as u64, nb_block);
    /// let lut = wopbs_key.generate_lut(&ct, |x| 2 * x);
    /// let ct_res = wopbs_key.wopbs_with_degree(&sks, &mut ct, &lut);
    /// let res = cks.decrypt_radix(&ct_res);
    ///
    ///  assert_eq!(res, (2 * clear) % moduli);
    /// ```
    pub fn generate_lut_radix<F, T>(&self, ct: &T, f: F) -> Vec<Vec<u64>>
    where
        F: Fn(u64) -> u64,
        T: IntegerCiphertext,
    {
        let mut total_bit = 0;
        let block_nb = ct.blocks().len();
        let mut modulus = 1;

        //This contains the basis of each block depending on the degree
        let mut vec_deg_basis = vec![];

        for (i, deg) in ct.moduli().iter().zip(ct.blocks().iter()) {
            modulus *= i;
            let b = f64::log2((deg.degree.0 + 1) as f64).ceil() as u64;
            vec_deg_basis.push(b);
            total_bit += b;
        }

        let mut lut_size = 1 << total_bit;
        if 1 << total_bit < self.wopbs_key.param.polynomial_size.0 as u64 {
            lut_size = self.wopbs_key.param.polynomial_size.0;
        }
        let mut vec_lut = vec![vec![0; lut_size]; ct.blocks().len()];

        let basis = ct.moduli()[0];
        let delta: u64 = (1 << 63)
            / (self.wopbs_key.param.message_modulus.0 * self.wopbs_key.param.carry_modulus.0)
                as u64;

        for lut_index_val in 0..(1 << total_bit) {
            let encoded_with_deg_val = encode_mix_radix(lut_index_val, &vec_deg_basis, basis);
            let decoded_val = decode_radix(encoded_with_deg_val.clone(), basis);
            let f_val = f(decoded_val % modulus) % modulus;
            let encoded_f_val = encode_radix(f_val, basis, block_nb as u64);
            for lut_number in 0..block_nb {
                vec_lut[lut_number as usize][lut_index_val as usize] =
                    encoded_f_val[lut_number] * delta;
            }
        }
        vec_lut
    }

    /// # Example
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_integer::wopbs::WopbsKey;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2;
    ///
    /// let param = WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// let nb_block = 3;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&param);
    /// //Generate wopbs_v0 key
    /// let mut wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut moduli = 1_u64;
    /// for _ in 0..nb_block{
    ///     moduli *= cks.parameters().message_modulus.0 as u64;
    /// }
    /// let clear = 15 % moduli;
    /// let mut ct = cks.encrypt_radix_without_padding(clear as u64, nb_block);
    /// let lut = wopbs_key.generate_lut_without_padding(&ct, |x| 2 * x);
    /// let ct_res = wopbs_key.wopbs_without_padding(&sks, &mut ct, &lut);
    /// let res = cks.decrypt_radix_without_padding(&ct_res);
    ///
    /// assert_eq!(res, (clear * 2) % moduli)
    /// ```
    pub fn generate_lut_radix_without_padding<F, T>(&self, ct: &T, f: F) -> Vec<Vec<u64>>
    where
        F: Fn(u64) -> u64,
        T: IntegerCiphertext,
    {
        let log_message_modulus = f64::log2((self.wopbs_key.param.message_modulus.0) as f64) as u64;
        let log_carry_modulus = f64::log2((self.wopbs_key.param.carry_modulus.0) as f64) as u64;
        let log_basis = log_message_modulus + log_carry_modulus;
        let delta = 64 - log_basis;
        let nb_block = ct.blocks().len();
        let poly_size = self.wopbs_key.param.polynomial_size.0;
        let mut lut_size = 1 << (nb_block * log_basis as usize);
        if lut_size < poly_size {
            lut_size = poly_size;
        }
        let mut vec_lut = vec![vec![0; lut_size]; nb_block];

        for index in 0..lut_size {
            // find the value represented by the index
            let mut value = 0;
            let mut tmp_index = index;
            for i in 0..nb_block as u64 {
                let tmp = tmp_index % (1 << (log_basis * (i + 1)));
                tmp_index -= tmp;
                value += tmp >> (log_carry_modulus * i);
            }

            // fill the LUTs
            for block in 0..nb_block {
                vec_lut[block][index] = ((f(value as u64) >> (log_carry_modulus * block as u64))
                    % (1 << log_message_modulus))
                    << delta
            }
        }
        vec_lut
    }

    /// generate lut for 'true' CRT
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_integer::parameters::PARAM_4_BITS_5_BLOCKS;
    /// use concrete_integer::wopbs::WopbsKey;
    ///
    /// let basis: Vec<u64> = vec![9, 11];
    ///
    /// let param = PARAM_4_BITS_5_BLOCKS;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&param);
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    ///
    /// let mut msg_space = 1;
    /// for modulus in basis.iter() {
    ///     msg_space *= modulus;
    /// }
    /// let clear = 42 % msg_space; // Encrypt the integers
    /// let mut ct = cks.encrypt_crt_not_power_of_two(clear, basis.clone());
    /// let lut = wopbs_key.generate_lut_crt_without_padding(&ct, |x| x);
    /// let ct_res = wopbs_key.wopbs_not_power_of_two(&sks, &mut ct, &lut);
    /// let res = cks.decrypt_crt_not_power_of_two(&ct_res);
    /// assert_eq!(res, clear);
    /// ```
    pub fn generate_lut_native_crt<F>(&self, ct: &CrtCiphertext, f: F) -> Vec<Vec<u64>>
    where
        F: Fn(u64) -> u64,
    {
        let mut bit = vec![];
        let mut total_bit = 0;
        let mut modulus = 1;
        let basis: Vec<_> = ct.moduli();

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
            let mut tmp = 1;
            for (base, bit) in basis.iter().zip(bit.iter()) {
                index_lut += (((value % base) << bit) / base) * tmp;
                tmp <<= bit;
            }
            for (j, b) in basis.iter().enumerate() {
                vec_lut[j][index_lut as usize] =
                    (((f(value) % b) as u128 * (1 << 64)) / *b as u128) as u64
            }
        }
        vec_lut
    }

    /// generate LUt for 'fake' crt
    /// # Example
    /// ```rust
    /// 
    /// use concrete_integer::gen_keys;
    /// use concrete_integer::wopbs::*;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_3_CARRY_3;
    ///
    /// let basis : Vec<u64> = vec![5,7];
    /// let nb_block = basis.len();
    ///
    /// let param = WOPBS_PARAM_MESSAGE_3_CARRY_3;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&param);
    /// let wopbs_key =  WopbsKey::new_wopbs_key(&cks, &sks);
    ///
    /// let mut msg_space = 1;
    /// for modulus in basis.iter() {
    ///     msg_space *= modulus;
    /// }
    /// let clear = 42 % msg_space;
    /// let mut ct = cks.encrypt_crt(clear, basis.clone());
    /// let lut = wopbs_key.generate_lut_fake_crt(&ct, |x| x);
    /// let ct_res = wopbs_key.wopbs_with_degree(&sks, &mut ct, &lut);
    /// let res = cks.decrypt_crt(&ct_res);
    /// assert_eq!(res, clear);
    /// ```
    pub fn generate_lut_crt<F>(&self, ct: &CrtCiphertext, f: F) -> Vec<Vec<u64>>
    where
        F: Fn(u64) -> u64,
    {
        let mut bit = vec![];
        let mut total_bit = 0;
        let mut modulus = 1;
        let basis = ct.moduli();

        for (i, deg) in basis.iter().zip(ct.blocks.iter()) {
            modulus *= i;
            let b = f64::log2((deg.degree.0 + 1) as f64).ceil() as u64;
            total_bit += b;
            bit.push(b);
        }
        let mut lut_size = 1 << total_bit;
        if 1 << total_bit < self.wopbs_key.param.polynomial_size.0 as u64 {
            lut_size = self.wopbs_key.param.polynomial_size.0;
        }
        let mut vec_lut = vec![vec![0; lut_size]; basis.len()];

        for i in 0..(1 << total_bit) {
            let mut value = i.clone();
            for (j, block) in ct.blocks.iter().enumerate() {
                let deg = f64::log2((block.degree.0 + 1) as f64).ceil() as u64;
                let delta: u64 = (1 << 63)
                    / (self.wopbs_key.param.message_modulus.0
                        * self.wopbs_key.param.carry_modulus.0) as u64;
                vec_lut[j][i as usize] =
                    ((f((value % (1 << deg)) % block.message_modulus.0 as u64))
                        % block.message_modulus.0 as u64)
                        * delta;
                value >>= deg;
            }
        }
        vec_lut
    }

    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2;
    /// use concrete_integer::wopbs::*;
    ///
    /// let nb_block = 3;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&WOPBS_PARAM_MESSAGE_2_CARRY_2);
    ///
    /// //Generate wopbs_v0 key    ///
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    /// let mut moduli = 1_u64;
    /// for _ in 0..nb_block{
    ///     moduli *= cks.parameters().message_modulus.0 as u64;
    /// }
    /// let clear1 = 42 % moduli;
    /// let clear2 = 24 % moduli;
    /// let mut ct1 = cks.encrypt_radix(clear1 as u64, nb_block);
    /// let mut ct2 = cks.encrypt_radix(clear2 as u64, nb_block);
    ///
    /// let lut = wopbs_key.generate_lut_bivariate_radix(&ct1, &ct2, |x,y| 2 * x * y);
    /// let ct_res = wopbs_key.bivariate_wopbs_with_degree(&sks, &mut ct1, &mut ct2, &lut);
    /// let res = cks.decrypt_radix(&ct_res);
    ///
    ///  assert_eq!(res, (2 * clear1 * clear2) % moduli);
    /// ```
    pub fn generate_lut_bivariate_radix<F>(
        &self,
        ct1: &RadixCiphertext,
        ct2: &RadixCiphertext,
        f: F,
    ) -> Vec<Vec<u64>>
    where
        F: Fn(u64, u64) -> u64,
    {
        let mut nb_bit_to_extract = vec![0; 2];
        let block_nb = ct1.blocks.len();
        //ct2 & ct1 should have the same basis
        let basis = ct1.moduli();

        //This contains the basis of each block depending on the degree
        let mut vec_deg_basis = vec![vec![]; 2];

        let mut modulus = 1;
        for (ct_num, ct) in [ct1, ct2].iter().enumerate() {
            modulus = 1;
            for deg in ct.blocks.iter() {
                modulus *= self.wopbs_key.param.message_modulus.0 as u64;
                let b = f64::log2((deg.degree.0 + 1) as f64).ceil() as u64;
                vec_deg_basis[ct_num].push(b);
                nb_bit_to_extract[ct_num] += b;
            }
        }

        let total_bit: u64 = nb_bit_to_extract.iter().sum();

        let mut lut_size = 1 << total_bit;
        if 1 << total_bit < self.wopbs_key.param.polynomial_size.0 as u64 {
            lut_size = self.wopbs_key.param.polynomial_size.0;
        }
        let mut vec_lut = vec![vec![0; lut_size]; basis.len()];
        let basis = ct1.moduli()[0];

        let delta: u64 = (1 << 63)
            / (self.wopbs_key.param.message_modulus.0 * self.wopbs_key.param.carry_modulus.0)
                as u64;

        for lut_index_val in 0..(1 << total_bit) {
            let split = vec![
                lut_index_val % (1 << nb_bit_to_extract[0]),
                lut_index_val >> nb_bit_to_extract[0],
            ];
            let mut decoded_val = vec![0; 2];
            for i in 0..2 {
                let encoded_with_deg_val = encode_mix_radix(split[i], &vec_deg_basis[i], basis);
                decoded_val[i] = decode_radix(encoded_with_deg_val.clone(), basis);
            }
            let f_val = f(decoded_val[0] % modulus, decoded_val[1] % modulus) % modulus;
            let encoded_f_val = encode_radix(f_val, basis, block_nb as u64);
            for lut_number in 0..block_nb {
                vec_lut[lut_number as usize][lut_index_val as usize] =
                    encoded_f_val[lut_number] * delta;
            }
        }
        vec_lut
    }

    /// generate bivariate LUT for 'fake' CRT
    ///
    /// # Example
    ///
    /// ```rust
    /// 
    /// use concrete_integer::gen_keys;
    /// use concrete_integer::wopbs::*;
    /// use concrete_shortint::parameters::parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_3_CARRY_3;
    ///
    /// let basis : Vec<u64> = vec![5,7];
    /// let param = WOPBS_PARAM_MESSAGE_3_CARRY_3;
    /// //Generate the client key and the server key:
    /// let ( cks, sks) = gen_keys(&param);
    /// let wopbs_key =  WopbsKey::new_wopbs_key(&cks, &sks);
    ///
    /// let mut msg_space = 1;
    /// for modulus in basis.iter() {
    ///     msg_space *= modulus;
    /// }
    /// let clear1 = 42 % msg_space;    // Encrypt the integers
    /// let clear2 = 24 % msg_space;    // Encrypt the integers
    /// let mut ct1 = cks.encrypt_crt(clear1, basis.clone());
    /// let mut ct2 = cks.encrypt_crt(clear2, basis.clone());
    /// let lut = wopbs_key.generate_lut_bivariate_fake_crt(&ct1, &ct2, |x,y| x * y * 2);
    /// let ct_res = wopbs_key.bivariate_wopbs_with_degree(&sks, &mut ct1, &mut ct2, &lut);
    /// let res = cks.decrypt_crt(&ct_res);
    /// assert_eq!(res, (clear1 * clear2 * 2) % msg_space );
    /// ```
    pub fn generate_lut_bivariate_crt<F>(
        &self,
        ct1: &CrtCiphertext,
        ct2: &CrtCiphertext,
        f: F,
    ) -> Vec<Vec<u64>>
    where
        F: Fn(u64, u64) -> u64,
    {
        let mut bit = vec![];
        let mut nb_bit_to_extract = vec![0; 2];
        let mut modulus = 1;

        //ct2 & ct1 should have the same basis
        let basis = ct1.moduli();

        for (ct_num, ct) in [ct1, ct2].iter().enumerate() {
            for (i, deg) in basis.iter().zip(ct.blocks.iter()) {
                modulus *= i;
                let b = f64::log2((deg.degree.0 + 1) as f64).ceil() as u64;
                nb_bit_to_extract[ct_num] += b;
                bit.push(b);
            }
        }

        let total_bit: u64 = nb_bit_to_extract.iter().sum();

        let mut lut_size = 1 << total_bit;
        if 1 << total_bit < self.wopbs_key.param.polynomial_size.0 as u64 {
            lut_size = self.wopbs_key.param.polynomial_size.0;
        }
        let mut vec_lut = vec![vec![0; lut_size]; basis.len()];

        let delta: u64 = (1 << 63)
            / (self.wopbs_key.param.message_modulus.0 * self.wopbs_key.param.carry_modulus.0)
                as u64;

        for index in 0..(1 << total_bit) {
            let mut split = encode_radix(index, 1 << nb_bit_to_extract[0], 2);
            let mut crt_value = vec![vec![0; ct1.blocks.len()]; 2];
            for j in 0..ct1.blocks.len() {
                let deg_1 = f64::log2((ct1.blocks[j].degree.0 + 1) as f64).ceil() as u64;
                let deg_2 = f64::log2((ct2.blocks[j].degree.0 + 1) as f64).ceil() as u64;
                crt_value[0][j] = (split[0] % (1 << deg_1)) % basis[j];
                crt_value[1][j] = (split[1] % (1 << deg_2)) % basis[j];
                split[0] >>= deg_1;
                split[1] >>= deg_2;
            }
            let value_1 = i_crt(&ct1.moduli(), &crt_value[0]);
            let value_2 = i_crt(&ct2.moduli(), &crt_value[1]);
            for (j, current_mod) in basis.iter().enumerate() {
                let value = f(value_1, value_2) % current_mod;
                vec_lut[j][index as usize] = (value % current_mod) * delta;
            }
        }

        vec_lut
    }

    /// generate bivariate LUT for 'true' CRT
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_integer::parameters::PARAM_4_BITS_5_BLOCKS;
    /// use concrete_integer::wopbs::WopbsKey;
    ///
    /// let basis: Vec<u64> = vec![9, 11];
    ///
    /// let param = PARAM_4_BITS_5_BLOCKS;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&param);
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    ///
    /// let mut msg_space = 1;
    /// for modulus in basis.iter() {
    ///     msg_space *= modulus;
    /// }
    /// let clear1 = 42 % msg_space;
    /// let clear2 = 24 % msg_space;
    /// let mut ct1 = cks.encrypt_crt_not_power_of_two(clear1, basis.clone());
    /// let mut ct2 = cks.encrypt_crt_not_power_of_two(clear2, basis.clone());
    /// let lut = wopbs_key.generate_lut_bivariate_crt_without_padding(&ct1, |x, y| x * y * 2);
    /// let ct_res = wopbs_key.bivariate_wopbs_not_power_of_two(&sks, &mut ct1, &mut ct2, &lut);
    /// let res = cks.decrypt_crt_not_power_of_two(&ct_res);
    /// assert_eq!(res, (clear1 * clear2 * 2) % msg_space);
    /// ```
    pub fn generate_lut_bivariate_native_crt<F>(&self, ct_1: &CrtCiphertext, f: F) -> Vec<Vec<u64>>
    where
        F: Fn(u64, u64) -> u64,
    {
        let mut bit = vec![];
        let mut total_bit = 0;
        let mut modulus = 1;
        let basis = ct_1.moduli();
        for i in basis.iter() {
            modulus *= i;
            let b = f64::log2(*i as f64).ceil() as u64;
            total_bit += b;
            bit.push(b);
        }
        let mut lut_size = 1 << (2 * total_bit);
        if 1 << (2 * total_bit) < self.wopbs_key.param.polynomial_size.0 as u64 {
            lut_size = self.wopbs_key.param.polynomial_size.0;
        }
        let mut vec_lut = vec![vec![0; lut_size]; basis.len()];

        for value in 0..1 << (2 * total_bit) {
            let value_1 = value % (1 << total_bit);
            let value_2 = value >> total_bit;
            let mut index_lut_1 = 0;
            let mut index_lut_2 = 0;
            let mut tmp = 1;
            for (base, bit) in basis.iter().zip(bit.iter()) {
                index_lut_1 += (((value_1 % base) << bit) / base) * tmp;
                index_lut_2 += (((value_2 % base) << bit) / base) * tmp;
                tmp <<= bit;
            }
            let index = (index_lut_2 << total_bit) + (index_lut_1);
            for (j, b) in basis.iter().enumerate() {
                vec_lut[j][index as usize] =
                    (((f(value_1, value_2) % b) as u128 * (1 << 64)) / *b as u128) as u64
            }
        }
        vec_lut
    }

    /// bivariate WOPBS for 'true' CRT
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_integer::parameters::PARAM_4_BITS_5_BLOCKS;
    /// use concrete_integer::wopbs::WopbsKey;
    ///
    /// let basis: Vec<u64> = vec![9, 11];
    ///
    /// let param = PARAM_4_BITS_5_BLOCKS;
    /// //Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&param);
    /// let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks);
    ///
    /// let mut msg_space = 1;
    /// for modulus in basis.iter() {
    ///     msg_space *= modulus;
    /// }
    /// let clear1 = 42 % msg_space;
    /// let clear2 = 24 % msg_space;
    /// let mut ct1 = cks.encrypt_crt_not_power_of_two(clear1, basis.clone());
    /// let mut ct2 = cks.encrypt_crt_not_power_of_two(clear2, basis.clone());
    /// let lut = wopbs_key.generate_lut_bivariate_crt_without_padding(&ct1, |x, y| x * y * 2);
    /// let ct_res = wopbs_key.bivariate_wopbs_not_power_of_two(&sks, &mut ct1, &mut ct2, &lut);
    /// let res = cks.decrypt_crt_not_power_of_two(&ct_res);
    /// assert_eq!(res, (clear1 * clear2 * 2) % msg_space);
    /// ```
    pub fn bivariate_wopbs_native_crt(
        &self,
        sks: &ServerKey,
        ct1: &CrtCiphertext,
        ct2: &CrtCiphertext,
        lut: &[Vec<u64>],
    ) -> CrtCiphertext {
        self.circuit_bootstrap_vertical_packing_native_crt(sks, &[ct1.clone(), ct2.clone()], lut)
    }

    fn circuit_bootstrap_vertical_packing_native_crt<T>(
        &self,
        sks: &ServerKey,
        vec_ct_in: &[T],
        lut: &[Vec<u64>],
    ) -> T
    where
        T: IntegerCiphertext,
    {
        let mut extracted_bits_blocks = vec![];
        for ct_in in vec_ct_in.iter() {
            let mut ct_in = ct_in.clone();
            // Extraction of each bit for each block
            for block in ct_in.blocks_mut().iter_mut() {
                let nb_bit_to_extract =
                    f64::log2((block.message_modulus.0 * block.carry_modulus.0) as f64).ceil()
                        as usize;
                let delta_log = DeltaLog(64 - nb_bit_to_extract);

                // trick ( ct - delta/2 + delta/2^4  )
                let lwe_size = block.ct.lwe_dimension().to_lwe_size().0;
                let mut cont = vec![0u64; lwe_size];
                cont[lwe_size - 1] =
                    (1 << (64 - nb_bit_to_extract - 1)) - (1 << (64 - nb_bit_to_extract - 5));
                let mut engine = DefaultEngine::new(Box::new(UnixSeeder::new(0))).unwrap();
                let tmp = engine.create_lwe_ciphertext_from(cont).unwrap();
                engine.fuse_sub_lwe_ciphertext(&mut block.ct, &tmp).unwrap();

                let extracted_bits =
                    self.wopbs_key
                        .extract_bits(delta_log, block, &sks.key, nb_bit_to_extract);
                extracted_bits_blocks.push(extracted_bits);
            }
        }

        extracted_bits_blocks.reverse();

        let vec_ct_out = self.wopbs_key.circuit_bootstrapping_vertical_packing(
            &sks.key,
            lut.to_vec(),
            extracted_bits_blocks,
        );

        let mut ct_vec_out: Vec<concrete_shortint::Ciphertext> = vec![];
        for (block, block_out) in vec_ct_in[0].blocks().iter().zip(vec_ct_out.into_iter()) {
            ct_vec_out.push(concrete_shortint::Ciphertext {
                ct: block_out,
                degree: Degree(block.message_modulus.0 - 1),
                message_modulus: block.message_modulus,
                carry_modulus: block.carry_modulus,
            });
        }
        T::from_blocks(ct_vec_out)
    }
}
