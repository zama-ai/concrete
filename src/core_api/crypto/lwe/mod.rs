//! LWE Tensor Operations
//! * Contains functions only related to LWE samples
//! * By default, homomorphic operations are implemented to handle several ciphertexts
//! * The prefix "mono_" means that the operation is only implemented for a unique ciphertext

#[cfg(test)]
mod tests;

use crate::core_api::crypto::SecretKey;
use crate::core_api::math::Tensor;
use crate::Types;
use itertools::enumerate;
use itertools::izip;
use std::debug_assert;

pub trait LWE: Sized {
    fn key_switch(
        ct_res: &mut [Self],
        ct_in: &[Self],
        ksk: &[Self],
        base_log: usize,
        level: usize,
        dimension_before: usize,
        dimension_after: usize,
    );
    fn mono_key_switch(
        ct_res: &mut [Self],
        ct_in: &[Self],
        ksk: &[Self],
        base_log: usize,
        level: usize,
        dimension_before: usize,
        dimension_after: usize,
    );
    #[no_mangle]
    fn sk_encrypt(res: &mut [Self], sk: &[Self], mu: &[Self], dimension: usize, std_dev: f64);
    fn mono_sk_encrypt(res: &mut [Self], sk: &[Self], mu: &Self, std_dev: f64);
    fn trivial_sk_encrypt(res: &mut [Self], mu: &[Self], dimension: usize, std_dev: f64);
    fn compute_phase(result: &mut [Self], sk: &[Self], ciphertexts: &[Self], dimension: usize);
    fn scalar_mul(t_res: &mut [Self], t_in: &[Self], t_w: &[Self], dimension: usize);
    fn mono_scalar_mul(ct_res: &mut [Self], ct_in: &[Self], w: Self);
    fn mono_multisum_with_bias(
        ct_res: &mut [Self],
        ct_in: &[Self],
        weights: &[Self],
        bias: Self,
        dimension: usize,
    );
    fn multisum_with_bias(
        t_res: &mut [Self],
        t_in: &[Self],
        t_weights: &[Self],
        t_bias: &[Self],
        dimension: usize,
    );
    fn create_key_switching_key(
        ksk: &mut [Self],
        base_log: usize,
        level: usize,
        dimension_after: usize,
        std: f64,
        sk_before: &[Self],
        sk_after: &[Self],
    );
    fn create_trivial_key_switching_key(
        ksk: &mut [Self],
        base_log: usize,
        level: usize,
        dimension_after: usize,
        std: f64,
        sk_before: &[Self],
        sk_after: &[Self],
    );
}

macro_rules! impl_trait_lwe {
    ($T:ty,$DOC:expr) => {
        impl LWE for $T {
            /// Keyswitch several LWE ciphertexts encrypted under the same key
            /// it implies that the KSK is the same for all the input ciphertexts
            /// # Comments
            /// * warning: panic when base_log*level>=TORUS_BIT
            /// * warning: mask_res has to be filled with zeros when we call this function!
            /// * warning: naive implementation calling mono_key_switch() without any optimization!
            /// # Arguments
            /// * `ct_res` - Torus slice containing the output LWEs (output)
            /// * `ct_in` - Torus slice containing the input LWEs
            /// * `ksk` - Torus slice containing the keyswitching key
            /// * `base_log` - number of bits for the base B (B=2^base_log)
            /// * `level` - number of blocks of the gadget matrix
            /// * `dimension_before` - size of the LWE masks before key switching (typical value: n=1024)
            /// * `dimension_after` - size of the LWE masks after key switching (typical value: n=630)
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{LWE, lwe};
            #[doc = $DOC]
            ///
            /// // parameters
            /// let nb_ct: usize = 10;
            /// let base_log: usize = 4;
            /// let level: usize = 3;
            /// let dimension_before = 1024;
            /// let dimension_after = 600;
            ///
            /// // ciphertexts before key switching
            /// let mut ciphertexts_before: Vec<Torus> = vec![0; nb_ct * (dimension_before + 1)];
            ///
            /// // ... (fill the ciphertexts)
            ///
            /// // key switching key
            /// let mut ksk: Vec<Torus> =
            ///     vec![0; lwe::get_ksk_size(dimension_before, dimension_after, level)];

            /// // ... (create the key switching key)
            ///
            /// // allocate the after key switching ciphertexts
            /// let mut ciphertexts_after: Vec<Torus> = vec![0; nb_ct * (dimension_after + 1)];
            ///
            /// // key switch before -> after
            /// LWE::key_switch(
            ///     &mut ciphertexts_after,
            ///     &ciphertexts_before,
            ///     &ksk,
            ///     base_log,
            ///     level,
            ///     dimension_before,
            ///     dimension_after,
            /// );
            /// ```
            fn key_switch(
                ct_res: &mut [$T],
                ct_in: &[$T],
                ksk: &[$T],
                base_log: usize,
                level: usize,
                dimension_before: usize,
                dimension_after: usize,
            ) {
                let lwe_size_before = dimension_before + 1 ;
                let lwe_size_after = dimension_after + 1 ;
                debug_assert!(ct_in.len() / lwe_size_before == ct_res.len() / lwe_size_after,
                    "There is {} input lwes and {} output lwes ... ",
                    ct_in.len() / lwe_size_before, ct_res.len() / lwe_size_after ) ;
                // for each ciphertext, call mono_key_switch
                for (block_res, block_in) in
                    ct_res.chunks_mut(lwe_size_after).zip(
                        ct_in
                            .chunks(lwe_size_before)
                    )
                {
                    Self::mono_key_switch(
                        block_res,
                        block_in,
                        ksk,
                        base_log,
                        level,
                        dimension_before,
                        dimension_after,
                    );
                }
            }

            /// Keyswitch one LWE ciphertext
            /// # Comments
            /// * warning: panic when base_log*level>=TORUS_BIT
            /// * warning: ciphertext has to be filled with zeros when we call this function!
            /// # Arguments
            /// * `ct_res` - Torus slice containing the output LWE (output)
            /// * `ct_in` - Torus slice containing the input LWE
            /// * `ksk` - Torus slice containing the keyswitching key
            /// * `base_log` - number of bits for the base B (B=2^base_log)
            /// * `level` - number of blocks of the gadget matrix
            /// * `dimension_before` - size of the LWE mask before key switching (typical value: n=1024)
            /// * `dimension_after` - size of the LWE mask after key switching (typical value: n=630)
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{LWE, lwe};
            #[doc = $DOC]
            ///
            /// // parameters
            /// let base_log: usize = 4;
            /// let level: usize = 3;
            /// let dimension_before = 1024;
            /// let dimension_after = 600;
            ///
            /// // ciphertexts before key switching
            /// let mut ciphertext_before: Vec<Torus> = vec![0; dimension_before + 1];
            ///
            /// // ... (fill the ciphertexts)
            ///
            /// // key switching key
            /// let mut ksk: Vec<Torus> =
            ///     vec![0; lwe::get_ksk_size(dimension_before, dimension_after, level) ];
            ///
            /// // ... (create the key switching key)
            ///
            /// // allocate the after key switching ciphertexts
            /// let mut ciphertext_after: Vec<Torus> = vec![0; dimension_after + 1];
            ///
            /// // key switch before -> after
            /// LWE::mono_key_switch(
            ///     &mut ciphertext_after,
            ///     &ciphertext_before,
            ///     &ksk,
            ///     base_log,
            ///     level,
            ///     dimension_before,
            ///     dimension_after,
            /// );
            /// ```
            fn mono_key_switch(
                ct_res: &mut [$T],
                ct_in: &[$T],
                ksk: &[$T],
                base_log: usize,
                level: usize,
                dimension_before: usize,
                dimension_after: usize,
            ) {
                ct_res[dimension_after] = ct_in[dimension_before];
                // compute chunck's size according to level and dimensions parameters
                let chunk_size: usize = level * (dimension_after + 1);

                // create a temporary variable that will contain each signed decomposition in the loop
                let mut decomp: Vec<$T> = vec![0; level];

                // loop over the coefficients in the LWE
                // ai will represent the i-th value of the input mask
                // block represent blocks of the KSK
                for (block,  ai) in ksk
                    .chunks(chunk_size)
                    .zip(ct_in.iter())
                {
                    let ai_rounded = Types::round_to_closest_multiple(*ai, base_log, level);
                    Types::torus_small_sign_decompose(&mut decomp, ai_rounded, base_log);

                    // loop over the number of levels
                    // d is the i-th element of the signed decomposition
                    for (block_i,  d) in block
                        .chunks(dimension_after + 1)
                        .zip(decomp.iter())
                    {
                        Tensor::sub_scalar_mul(ct_res, block_i, *d);
                    }
                }
            }
            /// Encrypts a bunch of messages with the same secret key sk
            /// # Arguments
            /// * `res` - Torus slice containing the output LWE (output)
            /// * `sk` - torus slice representing a boolean slice, the secret key for the encryption
            /// * `mu` - a slice of encoded messages to be encrypted
            /// * `dimension`- size of LWE mask
            /// * `std_dev` - standard deviation of the normal distribution for the error added in the encryption
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{LWE, SecretKey};
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // parameters
            /// let dimension = 1024;
            /// let nb_ct = 10;
            ///
            /// // generate the secret key
            /// let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, 1);
            /// let mut sk: Vec<Torus> = vec![0; sk_len];
            /// Tensor::uniform_random_default(&mut sk);
            ///
            /// // generate random messages
            /// let mut messages: Vec<Torus> = vec![0; nb_ct];
            ///
            /// // ... (fill the messages to encrypt)
            ///
            /// // allocate the ciphertexts
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
            ///
            /// // encryption
            /// LWE::sk_encrypt(
            ///     &mut ciphertexts,
            ///     &sk,
            ///     &messages,
            ///     dimension,
            ///     0.00003,
            /// );
            /// ```
            fn sk_encrypt(
                res: &mut [$T],
                sk: &[$T],
                mu: &[$T],
                dimension: usize,
                std_dev: f64,
            ) {
                // we need to encrypt several Lwes
                let lwe_size = dimension + 1;
                debug_assert!(res.len() / lwe_size == mu.len(),
                    "There is room for encrypting {} messages and there is {} messages.",
                    res.len() / lwe_size, mu.len() ) ;
                for (res_i, m) in res.chunks_mut(lwe_size).zip(mu.iter()) {
                    Self::mono_sk_encrypt(res_i, sk, m, std_dev );
                }

            }

            /// Encrypts a message with a secret key sk
            /// # Warnings
            /// * re write without the temporary result tensor
            /// # Arguments
            /// * `res` - Torus slice containing the output LWE (output)
            /// * `sk` - torus slice representing a boolean slice, the secret key for the encryption
            /// * `mu` - encoded message to be encrypted
            /// * `std_dev` - standard deviation of the normal distribution for the error added in the encryption
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{LWE, SecretKey};
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // parameters
            /// let dimension = 1024;
            ///
            /// // generate the secret key
            /// let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, 1);
            /// let mut sk: Vec<Torus> = vec![0; sk_len];
            /// Tensor::uniform_random_default(&mut sk);
            ///
            /// // generate random message
            /// let mut message: Torus = 0;
            ///
            /// // ... (fill the message to encrypt)
            ///
            /// // allocate the ciphertexts
            /// let mut ciphertexts: Vec<Torus> = vec![0; dimension + 1];
            ///
            /// // encryption
            /// LWE::mono_sk_encrypt(
            ///     &mut ciphertexts,
            ///     &sk,
            ///     &message,
            ///     0.00003,
            /// );
            /// ```
            fn mono_sk_encrypt(
                res: &mut [$T],
                sk: &[$T],
                mu: &$T,
                std_dev: f64,
            ) {
                let (body_res, mask_res) = res.split_last_mut().expect("Wrong length") ;
                // generate a uniformly random mask
                Tensor::uniform_random_default(mask_res);

                // generate an error from the normal distribution described by std_dev
                *body_res = Tensor::get_normal_random_default(0., std_dev);

                // compute the multisum between the secret key and the mask
                *body_res = body_res.wrapping_add(Tensor::get_binary_multisum(mask_res, sk));
                // add the encoded message
                *body_res = body_res.wrapping_add(*mu);
            }

            /// Performs a trivial encryption for a bunch of ciphertexts
            /// meaning that the mask is set to zero
            /// adds noise in the body, set std_dev to zero if not wanted
            /// # Arguments
            /// * `res` - Torus slice containing the produced ciphertexts (output)
            /// * `mu` - tensor of encoded messages to be encrypted
            /// * `std_dev` - standard deviation of the normal distribution
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{LWE, secret_key};
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // parameters
            /// let dimension = 1024;
            /// let nb_ct = 10;
            ///
            /// // generate random messages
            /// let mut messages: Vec<Torus> = vec![0; nb_ct];
            ///
            /// // ... (fill the messages to encrypt)
            ///
            /// // allocate the ciphertexts
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
            ///
            /// // encryption
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
            /// LWE::trivial_sk_encrypt(
            ///     &mut ciphertexts,
            ///     &messages,
            ///     dimension,
            ///     0.00003,
            /// );
            /// ```
            fn trivial_sk_encrypt(
                res: &mut [$T],
                mu: &[$T],
                dimension:usize,
                std_dev: f64,
            ) {
                let lwe_size = dimension + 1;
                debug_assert!(res.len() / lwe_size == mu.len(),
                "There is room for encrypting {} messages and there is {} messages.",
                res.len() / lwe_size, mu.len() ) ;
                for (res_i, m) in res.chunks_mut(lwe_size).zip(mu.iter()) {
                    let (body_res, mask_res) = res_i.split_last_mut().expect("Wrong length") ;
                // set mask elements to zero
                    for elt in mask_res.iter_mut() {
                        *elt = 0;
                    }

                    // generate an error from the normal distribution described by std_dev
                    *body_res = Tensor::get_normal_random_default(0., std_dev);
                    // add the encoded message
                    *body_res = body_res.wrapping_add(*m);
                }
            }

            /// Decrypts a bunch of ciphertexts encrypted with the same key
            /// # Arguments
            /// * `result` - Torus slice containing the decryption of the LWE (output)
            /// * `sk` - torus slice representing a boolean slice, the secret key for encryption
            /// * `ciphertexts` - Torus slice containing the input ciphertexts
            /// * `dimension` - size of each mask
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{LWE, SecretKey};
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // parameters
            /// let nb_ct: usize = 10;
            /// let dimension = 1024;
            ///
            /// // secret key
            /// let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, 1);
            /// let mut sk: Vec<Torus> = vec![0; sk_len];
            /// Tensor::uniform_random_default(&mut sk);
            ///
            /// // ciphertexts to be decrypted
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
            ///
            /// // ... (fill the ciphertexts)
            ///
            /// // allocation for the decrypted messages
            /// let mut result: Vec<Torus> = vec![0; nb_ct];
            ///
            /// // decryption
            /// LWE::compute_phase(&mut result, &sk, &ciphertexts, dimension);
            /// ```
            fn compute_phase(
                result: &mut [$T],
                sk: &[$T],
                ciphertexts: &[$T],
                dimension: usize,
            ) {
                let lwe_size = dimension + 1 ;
                debug_assert!(ciphertexts.len() / lwe_size == result.len(),
                    "There is room for decrypting {} messages and there is {} lwes.",
                    result.len(), ciphertexts.len() / lwe_size ) ;
                for (ciphertexts_i, res) in ciphertexts.chunks(lwe_size).zip(result.iter_mut())
                {
                    let (&body_ciphertext, mask_ciphertext) = ciphertexts_i.split_last().expect("Wrong length") ;
                    // put body inside result
                    *res = res.wrapping_add(body_ciphertext);
                    // subtract the multisum between the key and the mask
                    *res = res.wrapping_sub(Self::get_binary_multisum(mask_ciphertext, sk)) ;
                }

            }

            /// Computes a scalar multiplication between a tensor of LWE ciphertexts and a tensor of signed scalar value t_w
            /// there are as many ciphertexts as there are signed values in t_w
            /// * warning: naive implementation calling mono_scalar_mul() without any optimization!
            /// # Arguments
            /// * `t_res` - Torus slice containing the produced ciphertexts (output)
            /// * `t_in` - Torus slice containing the input ciphertexts
            /// * `t_w` - Torus slice containing the signed weights as Torus elements
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::LWE;
            #[doc = $DOC]
            ///
            /// // settings
            /// let nb_ct: usize = 10;
            /// let dimension = 1024;
            ///
            /// // allocation for the ciphertexts
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
            ///
            /// // ... (fill the ciphertexts)
            ///
            /// // allocation for the weights
            /// let mut weights: Vec<Torus> = vec![0; nb_ct];
            ///
            /// // ... (fill the weights)
            ///
            /// // allocation for the result
            /// let mut ciphertexts_sm: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
            ///
            /// // scalar multiplication
            /// LWE::scalar_mul(
            ///     &mut ciphertexts_sm,
            ///     &ciphertexts,
            ///     &weights,
            ///     dimension,
            /// );
            /// ```
            fn scalar_mul(
                t_res: &mut [$T],
                t_in: &[$T],
                t_w: &[$T],
                dimension: usize,
            ) {
                debug_assert!(t_res.len() == t_in.len() , "t_res.len() = {} != t_in.len() = {}", t_res.len(), t_in.len()) ;
                debug_assert!(t_res.len() / (dimension + 1) == t_w.len(),  "There is {} lwes and {} weights.", t_res.len( ) / (dimension + 1), t_w.len()) ;
                for (res_i, in_i, w) in izip!(
                    t_res.chunks_mut(dimension + 1),
                    t_in.chunks(dimension + 1),
                    t_w.iter()
                ) {
                    for (res_i_j, in_i_j) in res_i.iter_mut().zip(in_i.iter())
                    {
                        *res_i_j = in_i_j.wrapping_mul(*w) ;
                    }
                }
            }

            /// Computes a scalar multiplication between one LWE ciphertexts and a signed scalar value w
            /// # Arguments
            /// * `ct_res` - Torus slice containing the  produced ciphertext (output)
            /// * `ct_in` - Torus slice containing the input ciphertext
            /// * `w` - Torus element containing the signed weight as Torus elements
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::LWE;
            #[doc = $DOC]
            ///
            /// // settings
            /// let dimension = 1024;
            ///
            /// // allocation for the ciphertexts
            /// let mut ciphertext: Vec<Torus> = vec![0; dimension + 1];
            ///
            /// // ... (fill the ciphertexts)
            ///
            /// // allocation for the weights
            /// let mut weight: Torus = 0;
            ///
            /// // ... (fill the weights)
            ///
            /// // allocation for the result
            /// let mut ciphertext_sm: Vec<Torus> = vec![0; dimension + 1];
            ///
            /// // scalar multiplication
            /// LWE::mono_scalar_mul(
            ///     &mut ciphertext_sm,
            ///     &ciphertext,
            ///     weight,
            /// );
            /// ```
            fn mono_scalar_mul(
                ct_res: &mut [$T],
                ct_in: &[$T],
                w: $T,
            ) {
                for (res_i, in_i) in ct_res.iter_mut().zip(ct_in.iter()) {
                    *res_i = in_i.wrapping_mul(w);
                }
            }

            /// Computes the multisum between a vector of ciphertexts and a weight vector of the same size and add a bias to it
            /// it output one ciphertext
            /// # Arguments
            /// * `ct_res` - Torus slice containing the produced ciphertext (output)
            /// * `ct_in` - Torus slice containing the input ciphertexts
            /// * `weights` - Torus slice containing signed weights for the multisum as Torus elements
            /// * `bias` - Torus element containing the bias value
            /// * `dimension` - Size of the LWE mask
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::LWE;
            /// use concrete_lib::Types;
            #[doc = $DOC]
            ///
            /// // parameters
            /// let nb_ct = 10; // the number of ciphertexts in this multisum
            /// let dimension = 1024;
            ///
            /// // generate the signed weights and the bias and represent them as Torus elements
            /// let mut weights: Vec<Torus> = vec![(-3 as <Torus as Types>::STorus) as Torus; nb_ct];
            /// let mut bias: Torus = 0;
            ///
            /// // ... (fill the weights and the bias)
            ///
            /// // allocate the ciphertexts that will end up in the multisum
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension+ 1)];
            ///
            /// // ... (fill the ciphertexts)
            ///
            /// // allocation for the results
            /// let mut res: Vec<Torus> = vec![0; dimension + 1];
            ///
            /// // computation of the multisum
            /// LWE::mono_multisum_with_bias(
            ///     &mut res,
            ///     &ciphertexts,
            ///     &weights,
            ///     bias,
            ///     dimension
            /// );
            /// ```
            fn mono_multisum_with_bias(
                ct_res: &mut [$T],
                ct_in: &[$T],
                weights: &[$T],
                bias: $T,
                dimension: usize,
            ) {
                debug_assert!(ct_in.len() / (dimension + 1) == weights.len(),  "There is {} lwes and {} weights.", ct_in.len( ) / (dimension + 1), weights.len()) ;
                // loop over the ciphertexts and the weights
                for (ct_in_i, w_i) in
                    ct_in.chunks(dimension + 1).zip(weights.iter())
                {
                    for (res_j, ct_in_i_j) in ct_res.iter_mut().zip(ct_in_i.iter()) {
                        *res_j = res_j.wrapping_add(ct_in_i_j.wrapping_mul(*w_i));
                    }
                }

                // add the bias
                ct_res[dimension] = ct_res[dimension].wrapping_add(bias);
            }

            /// Computes the multisum between a n vectors of ciphertexts and n weight vectors of the same size and add a different bias to it
            /// it output n ciphertexts
            /// # Warnings
            /// * it is a naive implementation that simply uses mono_multisum_with_bias
            /// * not tested
            /// # Arguments
            /// * `t_res` - Torus slice containing the produced ciphertexts (output)
            /// * `t_in` - Torus element containing the input ciphertexts
            /// * `t_weights` - Torus slice containing every signed weights for the multisums as Torus elements
            /// * `t_bias` - Torus element containing every bias values
            /// * `dimension` - size of the lwe mask
            /// ```rust
            /// use concrete_lib::core_api::crypto::LWE;
            /// use concrete_lib::Types;
            #[doc = $DOC]
            ///
            /// // parameters
            /// let nb_ms = 3; // the number of multisums we will compute
            /// let nb_ct = 10; // the number of ciphertexts in this multisum
            /// let dimension = 1024;
            ///
            /// // generate the signed weights and the biases and represent them as Torus elements
            /// let mut weights: Vec<Torus> = vec![(-3 as <Torus as Types>::STorus) as Torus; nb_ms * nb_ct];
            /// let mut biases: Vec<Torus> = vec![0; nb_ms];
            ///
            /// // ... (fill the weights and the biases)
            ///
            /// // allocate the ciphertexts that will end up in the multisum
            /// let mut ciphertexts: Vec<Torus> = vec![0; nb_ms * nb_ct * (dimension+ 1)];
            ///
            /// // ... (fill the ciphertexts)
            ///
            /// // allocation for the results
            /// let mut res: Vec<Torus> = vec![0; nb_ms * (dimension + 1)];
            ///
            /// // computation of the multisums
            /// LWE::multisum_with_bias(
            ///     &mut res,
            ///     &ciphertexts,
            ///     &weights,
            ///     &biases,
            ///     dimension
            /// );
            /// ```
            fn multisum_with_bias(
                t_res: &mut [$T],
                t_in: &[$T],
                t_weights: &[$T],
                t_bias: &[$T],
                dimension: usize
            ) {
                let nb_multisum: usize = t_res.len() - dimension ; // number of multisum computed here
                let len_multisum: usize = t_in.len() / nb_multisum  ;

                for (res_i, in_i, weights, bias) in izip!(t_res.chunks_mut(dimension +1 ), t_in.chunks(len_multisum), t_weights.chunks(nb_multisum), t_bias.iter()) {
                    Self::mono_multisum_with_bias(res_i, in_i, weights, *bias, dimension);
                }

            }

            /// Create the LWE key switching key
            /// # Arguments
            /// * `ksk` - Torus slice containing the TRGSW ciphertext (output)
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * `dimension_before` - size of the mask of input lwe
            /// * `dimension_after` - size of the mask of the key switching key
            /// * `std`: standard deviation of the encryption noise
            /// * `sk_before`: secret key before the key switch, i.e. input ciphertexts of the key switch
            /// * `sk_after`: secret key after the kef switch, i.e. output ciphertexts of the key switch
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{LWE, lwe, SecretKey};
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // parameters
            /// let base_log: usize = 4;
            /// let level: usize = 3;
            /// let dimension_before = 1024;
            /// let dimension_after = 600;
            ///
            /// // allocation for the before and the after keys
            /// let sk_len_before: usize = <Torus as SecretKey>::get_secret_key_length(dimension_before, 1);
            /// let sk_len_after: usize = <Torus as SecretKey>::get_secret_key_length(dimension_after, 1);
            ///
            /// // create the before and the after keys
            /// let mut sk_before: Vec<Torus> = vec![0; sk_len_before];
            /// let mut sk_after: Vec<Torus> = vec![0; sk_len_after];
            ///
            /// // fill the before and the after keys with uniform random
            /// Tensor::uniform_random_default(&mut sk_before);
            /// Tensor::uniform_random_default(&mut sk_after);
            ///
            /// // key switching key allocation
            /// let mut ksk: Vec<Torus> =
            ///     vec![0; lwe::get_ksk_size(dimension_before, dimension_after, level)];
            ///
            /// // key switching key creation
            /// LWE::create_key_switching_key(
            ///     &mut ksk,
            ///     base_log,
            ///     level,
            ///     dimension_after,
            ///     0.00003,
            ///     &sk_before,
            ///     &sk_after,
            /// );
            /// ```
            fn create_key_switching_key(
                ksk: &mut [$T],
                base_log: usize,
                level: usize,
                dimension_after: usize,
                std: f64,
                sk_before: &[$T],
                sk_after: &[$T],
            ) {
                // computes some needed sizes
                let ksk_size: usize = (dimension_after + 1)* level;

                // loop over the output
                for (i, ksk_i) in enumerate(
                    ksk
                        .chunks_mut(ksk_size)
                ) {
                    // fetch the i-th before key bit
                    let bit: bool = SecretKey::get_bit(sk_before, i);
                    let mut messages: Vec<$T> = vec![0; level];
                    if bit {
                        for (j, m) in enumerate(messages.iter_mut()) {
                            *m = Types::set_val_at_level_l(1, base_log, j);
                        }
                    }

                    // encrypts the i-th before key bit
                    Self::sk_encrypt(ksk_i, &sk_after, &messages, dimension_after, std);
                }
            }

            /// Create a trivial LWE key switching key
            /// # Arguments
            /// * `ksk` - Torus slice containing the TRGSW ciphertext (output)
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * `dimension_after` - size of the mask of the key switching key
            /// * `std`: standard deviation of the encryption noise
            /// * `sk_before`: secret key before the key switch, i.e. input ciphertexts of the key switch
            /// * `sk_after`: secret key after the kef switch, i.e. output ciphertexts of the key switch
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::{LWE, lwe, SecretKey};
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // parameters
            /// let base_log: usize = 4;
            /// let level: usize = 3;
            /// let dimension_before = 1024;
            /// let dimension_after = 600;
            ///
            /// // allocation for the before and the after keys
            /// let sk_len_before: usize = <Torus as SecretKey>::get_secret_key_length(dimension_before, 1);
            /// let sk_len_after: usize = <Torus as SecretKey>::get_secret_key_length(dimension_after, 1);
            ///
            /// // create the before and the after keys
            /// let mut sk_before: Vec<Torus> = vec![0; sk_len_before];
            /// let mut sk_after: Vec<Torus> = vec![0; sk_len_after];
            ///
            /// // fill the before and the after keys with uniform random
            /// Tensor::uniform_random_default(&mut sk_before);
            /// Tensor::uniform_random_default(&mut sk_after);
            ///
            /// // key switching key allocation
            /// let mut ksk: Vec<Torus> =
            ///     vec![0; lwe::get_ksk_size(dimension_before, dimension_after, level)];
            ///
            /// // trivial key switching key generation
            /// LWE::create_trivial_key_switching_key(
            ///     &mut ksk,
            ///     base_log,
            ///     level,
            ///     dimension_after,
            ///     0.00003,
            ///     &sk_before,
            ///     &sk_after,
            /// );
            /// ```
            fn create_trivial_key_switching_key(
                ksk: &mut [$T],
                base_log: usize,
                level: usize,
                dimension_after: usize,
                std: f64,
                sk_before: &[$T],
                _sk_after: &[$T],
            ) {
                // computes some needed sizes
                let ksk_size: usize = (dimension_after + 1)* level;

                // loop over the output
                for (i, ksk_i) in enumerate(
                    ksk
                        .chunks_mut(ksk_size)
                ) {
                    // fetch the i-th before key bit
                    let bit: bool = SecretKey::get_bit(sk_before, i);
                    let mut messages: Vec<$T> = vec![0; level];
                    if bit {
                        for (j, m) in enumerate(messages.iter_mut()) {
                            *m = Types::set_val_at_level_l(1, base_log, j);
                        }
                    }

                    // encrypts the i-th before key bit
                    Self::trivial_sk_encrypt(ksk_i, &messages, dimension_after, std);
                }
            }
        }
    };
}

impl_trait_lwe!(u32, "type Torus = u32;");
impl_trait_lwe!(u64, "type Torus = u64;");

#[no_mangle]
/// Computes the size of the key switching key according to some parameters
/// # Arguments
/// * `dimension_before` - size of the LWE masks before key switching (typical value: n=1024)
/// * `dimension_after` - size of the output LWE mask (typical value: n=630)
/// * `level` - number of blocks of the gadget matrix
/// # Output
/// * the desired length
/// # Example
/// ```rust
/// use concrete_lib::core_api::crypto::lwe;
/// type Torus = u32; // or u64
///
/// // parameters
/// let level: usize = 3;
/// let dimension_before = 1024;
/// let dimension_after = 600;
///
/// // key switching mask's body allocation
/// let mut ksk: Vec<Torus> = vec![0; lwe::get_ksk_size(dimension_before, dimension_after, level)];
/// ```
pub fn get_ksk_size(dimension_before: usize, dimension_after: usize, level: usize) -> usize {
    dimension_before * dimension_after * level + dimension_before * level
}
