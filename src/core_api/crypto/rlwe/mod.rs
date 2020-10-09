//! RLWE Tensor Operations
//! * Contains every function only related to tensors of RLWE samples

#[cfg(test)]
mod tests;

use crate::core_api::math::{PolynomialTensor, Tensor};
use crate::Types;

pub trait RLWE: Sized {
    fn sk_encrypt(
        t_res: &mut [Self],
        sk: &[Self],
        t_mu: &[Self],
        dimension: usize,
        polynomial_size: usize,
        std_dev: f64,
    );
    fn zero_encryption(
        t_res: &mut [Self],
        sk: &[Self],
        dimension: usize,
        polynomial_size: usize,
        std_dev: f64,
    );
    fn compute_phase(
        t_res: &mut [Self],
        sk: &[Self],
        t_ct: &[Self],
        dimension: usize,
        polynomial_size: usize,
    );
    fn add_gadgetmatrix(
        t_res: &mut [Self],
        mu: Self,
        dimension: usize,
        polynomial_size: usize,
        base_log: usize,
        level: usize,
    );
    fn add_gadgetmatrix_generic(
        t_res: &mut [Self],
        mu: Self,
        dimension: usize,
        polynomial_size: usize,
        base_log: usize,
        level: usize,
    );
}

macro_rules! impl_trait_rlwe {
    ($T:ty,$DOC:expr) => {
        impl RLWE for $T {
            /// Encrypts a bunch of messages with the same secret key into several RLWE samples, and fill each slot of each sample
            /// # Arguments
            /// * `t_res` -the computed RLWE (output)
            /// * `sk` - the secret key to encrypt the message
            /// * `t_mu` - the encoded messages to be encrypted
            /// * `dimension` - size of the rlwe mask
            /// * `polynomial_size: usize` - number of coefficients in polynomials
            /// * `std_dev` - standard deviation of the encryption noise
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{RLWE, SecretKey};
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let mut nb_ct: usize = 10;
            /// let mut dimension: usize = 20;
            /// let mut polynomial_size: usize = 128;
            ///
            /// // generates a secret key
            /// let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            /// let mut sk: Vec<Torus> = vec![0; sk_len];
            /// Tensor::uniform_random_default(&mut sk);
            ///
            /// // allocation for messages
            /// let mut messages: Vec<Torus> = vec![0; nb_ct * polynomial_size];
            ///
            /// // ... (fill the messages)
            ///
            /// // allocation for the ciphertexts
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1) * polynomial_size];
            ///
            /// // encryption
            /// RLWE::sk_encrypt(
            ///     &mut ciphertexts,
            ///     &sk,
            ///     &messages,
            ///     dimension as usize,
            ///     polynomial_size,
            ///     0.00003,
            /// );
            /// ```
            fn sk_encrypt(
                t_res: &mut [$T],
                sk: &[$T],
                t_mu: &[$T],
                dimension: usize,
                polynomial_size: usize,
                std_dev: f64,
            ) {
                let rlwe_size = (dimension + 1) * polynomial_size ;
                debug_assert!(t_res.len() / rlwe_size == t_mu.len() / polynomial_size,
                    "There is room for encrypting {} polynomials and there is {} input polynomials.",
                    t_res.len() / rlwe_size, t_mu.len() / polynomial_size ) ;
                for (res_i, m) in t_res.chunks_mut(rlwe_size).zip(t_mu.chunks(polynomial_size)) {
                    let (t_mask_res, t_body_res) = res_i.split_at_mut(dimension * polynomial_size) ;
                    Tensor::normal_random_default( t_body_res, 0., std_dev);
                    Tensor::uniform_random_default( t_mask_res);
                    PolynomialTensor::add_binary_multisum(t_body_res, t_mask_res, sk, polynomial_size);
                    Tensor::add_inplace(t_body_res, m) ;
                }
            }

            /// Writes encryptions of zeros in a list of ciphertexts
            /// # Arguments
            /// * `t_res` -  the computed RLWE samples (output)
            /// * `sk` - the secret key to encrypt the message
            /// * `dimension` - size of the rlwe mask
            /// * `polynomial_size` - number of coefficients in polynomials
            /// * `std_dev` - standard deviation of the encryption noise
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{RLWE, SecretKey};
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let mut nb_ct: usize = 10;
            /// let mut dimension: usize = 20;
            /// let mut polynomial_size: usize = 128;
            ///
            /// // generates a secret key
            /// let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            /// let mut sk: Vec<Torus> = vec![0; sk_len];
            /// Tensor::uniform_random_default(&mut sk);
            ///
            /// // allocation for the ciphertexts
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1) * polynomial_size];
            ///
            /// // encryption
            /// RLWE::zero_encryption(
            ///     &mut ciphertexts,
            ///     &sk,
            ///     dimension as usize,
            ///     polynomial_size,
            ///     0.00003,
            /// );
            /// ```
            fn zero_encryption(
                t_res: &mut [$T],
                sk: &[$T],
                dimension: usize,
                polynomial_size: usize,
                std_dev: f64,
            ) {
                let rlwe_size = (dimension + 1) * polynomial_size ;
                for res_i in t_res.chunks_mut(rlwe_size) {
                    let (t_mask_res, t_body_res) = res_i.split_at_mut(dimension * polynomial_size) ;
                    Tensor::normal_random_default( t_body_res, 0., std_dev);
                    Tensor::uniform_random_default( t_mask_res);
                    PolynomialTensor::add_binary_multisum(t_body_res, t_mask_res, sk, polynomial_size);
                }
            }

            /// Decrypt a bunch of ciphertext encrypted with the same key
            /// # Arguments
            /// * `t_res` - Torus slice containing the decryption of the LWE (output)
            /// * `sk` - torus slice representing a boolean slice, the secret key for encryption
            /// * `t_in` - Torus slice containing the input ciphertexts
            /// * `dimension` - size of the rlwe mask
            /// * `polynomial_size` - number of coefficients in polynomials
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{RLWE, SecretKey};
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let mut nb_ct: usize = 10;
            /// let mut dimension: usize = 20;
            /// let mut polynomial_size: usize = 128;
            ///
            /// // generates a secret key
            /// let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            /// let mut sk: Vec<Torus> = vec![0; sk_len];
            /// Tensor::uniform_random_default(&mut sk);
            ///
            /// // allocation for the ciphertexts
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1) * polynomial_size];
            ///
            /// // ... (fill the ciphertexts)
            ///
            /// // allocation for the decrypted messages
            /// let mut decryptions: Vec<Torus> = vec![0; nb_ct * polynomial_size];
            ///
            /// // encryption
            /// RLWE::compute_phase(
            ///     &mut decryptions,
            ///     &sk,
            ///     &ciphertexts,
            ///     dimension,
            ///     polynomial_size,
            /// );
            /// ```
            fn compute_phase(
                t_res: &mut [$T],
                sk: &[$T],
                t_in: &[$T],
                dimension: usize,
                polynomial_size: usize,
            ) {
                let rlwe_size = (dimension + 1) * polynomial_size ;
                debug_assert!(t_res.len() / polynomial_size == t_in.len() / rlwe_size,
                    "There is room for decrypting {} polynomials and there is {} input rlwes.",
                    t_res.len() / polynomial_size, t_in.len() / rlwe_size ) ;
                for (ct_i, res_i) in t_in.chunks(rlwe_size).zip(t_res.chunks_mut(polynomial_size)) {
                    let (ct_mask, ct_body) = ct_i.split_at(dimension * polynomial_size) ;
                    res_i.copy_from_slice(ct_body);
                    PolynomialTensor::sub_binary_multisum(res_i, ct_mask, sk, polynomial_size);
                }
            }

            /// Add one gadget matrix to a well sized set of RLWE samples
            /// # Arguments
            /// * `t_res` - the RLWE samples (output)
            /// * `mu` - a message
            /// * `dimension` - size of the mask
            /// * `polynomial_size` - max degree of the polynomials + 1
            /// * `base_log` - number of bits for the base B (B=2^base_log)
            /// * `level` - number of blocks of the gadget matrix
            /// Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{rgsw, RLWE};
            #[doc = $DOC]
            ///
            /// // settings
            /// let dimension: usize = 128;
            /// let polynomial_size: usize = 256;
            /// let level: usize = 3;
            /// let base_log: usize = 4;
            ///
            /// // compute the sizes for the allocation of the list or rlwe
            /// let trgsw_size: usize = rgsw::get_trgsw_size(dimension, polynomial_size, 1, level) ;
            ///
            /// // allocation of the list of rlwe samples
            /// let mut trgsw: Vec<Torus> = vec![0; trgsw_size];
            ///
            /// // ... (fill the rlwe samples)
            ///
            /// let bit = 1;
            ///
            /// // add the gadget matrix
            /// RLWE::add_gadgetmatrix(
            ///     &mut trgsw,
            ///     bit as Torus,
            ///     dimension,
            ///     polynomial_size,
            ///     base_log,
            ///     level,
            /// );
            /// ```
            fn add_gadgetmatrix(
                t_res: &mut [$T],
                mu: $T,
                dimension: usize,
                polynomial_size: usize,
                base_log: usize,
                level: usize,
            ) {
                let mut index_poly: usize;
                let mut index: usize;
                let mut elt: $T;

                // mask part
                for i in 0..(dimension + 1) {
                    // columns of H
                    for j in 0..level {
                        // precision of the gadget
                        index_poly = level * i * (dimension +1) + i + j * (dimension + 1);
                        index = index_poly * polynomial_size;
                        elt = 1 << (<$T as Types>::TORUS_BIT - (base_log * (j + 1)));
                        elt *= mu;
                        t_res[index] = t_res[index].wrapping_add(elt);
                    }
                }

                // // body part
                // for j in 0..level {
                //     // precision of the gadget
                //     let index = (dimension * level + j) * polynomial_size;
                //     let mut elt: $T = 1 << (<$T as Types>::TORUS_BIT - (base_log * (j + 1)));
                //     elt = elt * mu;
                //     t_body_res[index] = t_body_res[index].wrapping_add(elt);
                // }
            }

            fn add_gadgetmatrix_generic(
                t_res: &mut [$T],
                mu: $T,
                dimension: usize,
                polynomial_size: usize,
                base_log: usize,
                level: usize,
            ) {
                let mut index: usize;
                let mut elt: $T;
                let matrix_size = (dimension + 1) * (dimension + 1) * polynomial_size;
                for j in 0..level {
                    elt = 1 << (<$T as Types>::TORUS_BIT - (base_log * (j + 1)));
                    elt *= mu;
                    for i in 0..(dimension+1) {
                        index = matrix_size * j + i * polynomial_size * (dimension + 2);
                        t_res[index] = t_res[index].wrapping_add(elt);
                    }
                }
            }
        }
    };
}

impl_trait_rlwe!(u32, "type Torus = u32;");
impl_trait_rlwe!(u64, "type Torus = u64;");

/// Encrypts a message with a public key (to be implemented)
/// # Arguments
/// * `result` a slice where the ciphertext will be stored
/// * `public_key` the public key under which to encrypt the message
/// * `mu` the encoded message to be encrypted
// pub fn pk_encrypt(result: &mut [Torus], public_key: &[Torus], mu: &[Torus]) {}

/// return the needed size for a vector of i32 to contain the mask of a signed decomposition of an RLWE
/// # Arguments
/// * `dimension` - size of the mask
/// * `polynomial_size` - max degree of the polynomials + 1
/// * `level` - number of blocks of the gadget matrix
/// # Output
/// * the desired size as a usize
/// # Example
/// ```rust
/// use concrete_lib::core_api::crypto::rlwe;
///
/// // settings
/// let dimension: usize = 128;
/// let polynomial_size: usize = 256;
/// let level: usize = 3;
///
/// let mask_size: usize = rlwe::get_sign_decompose_mask_size(dimension, polynomial_size, level);
/// ```
pub fn get_sign_decompose_mask_size(
    dimension: usize,
    polynomial_size: usize,
    level: usize,
) -> usize {
    dimension * polynomial_size * level
}

/// Return the needed size for a vector of i32 to contain the body of a signed decomposition of an RLWE
/// # Arguments
/// * `polynomial_size` - max degree of the polynomials + 1
/// * `level` - number of blocks of the gadget matrix
/// # Output
/// * the desired size as a usize
/// # Example
/// ```rust
/// use concrete_lib::core_api::crypto::rlwe;
///
/// // settings
/// let polynomial_size: usize = 256;
/// let level: usize = 3;
///
/// let body_size: usize = rlwe::get_sign_decompose_body_size(polynomial_size, level);
/// ```
pub fn get_sign_decompose_body_size(polynomial_size: usize, level: usize) -> usize {
    polynomial_size * level
}
