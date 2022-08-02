use crate::ciphertext::Ciphertext;
use crate::ServerKey;
use rayon::prelude::*;

impl ServerKey {
    /// Computes homomorphically an addition between two ciphertexts encrypting integer
    /// values in the CRT decomposition.
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    /// let size = 4;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, size);
    ///
    /// let clear_1 = 14;
    /// let clear_2 = 14;
    /// let basis = vec![2, 3, 5];
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt(clear_1, basis.clone());
    /// let ctxt_2 = cks.encrypt_crt(clear_2, basis);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.unchecked_add_crt_assign_parallelized(&mut ctxt_1, &ctxt_2);
    ///
    /// // Decrypt
    /// let res = cks.decrypt_crt(&ctxt_1);
    /// assert_eq!((clear_1 + clear_2) % 30, res);
    /// ```
    pub fn unchecked_add_crt_assign_parallelized(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &Ciphertext,
    ) {
        ct_left
            .ct_vec
            .par_iter_mut()
            .zip(&ct_right.ct_vec)
            .for_each(|(ct_left, ct_right)| {
                self.key.unchecked_add_assign(ct_left, ct_right);
            });
    }

    pub fn unchecked_add_crt_parallelized(
        &self,
        ct_left: &Ciphertext,
        ct_right: &Ciphertext,
    ) -> Ciphertext {
        let mut ct_res = ct_left.clone();
        self.unchecked_add_crt_assign_parallelized(&mut ct_res, ct_right);
        ct_res
    }

    /// Computes homomorphically an addition between two ciphertexts encrypting integer values in
    /// the CRT decomposition.
    ///
    /// This checks that the addition is possible. In the case where the carry buffers are full,
    /// then it is automatically cleared to allow the operation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    /// let size = 4;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, size);
    ///
    /// let clear_1 = 29;
    /// let clear_2 = 29;
    /// let basis = vec![2, 3, 5];
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt(clear_1, basis.clone());
    /// let mut ctxt_2 = cks.encrypt_crt(clear_2, basis);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.smart_add_crt_assign_parallelized(&mut ctxt_1, &mut ctxt_2);
    ///
    /// // Decrypt
    /// let res = cks.decrypt_crt(&ctxt_1);
    /// assert_eq!((clear_1 + clear_2) % 30, res);
    /// ```
    pub fn smart_add_crt_assign_parallelized(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &mut Ciphertext,
    ) {
        ct_left
            .ct_vec
            .par_iter_mut()
            .zip(&mut ct_right.ct_vec)
            .for_each(|(block_left, block_right)| {
                self.key.smart_add_assign(block_left, block_right);
            });
    }

    pub fn smart_add_crt_parallelized(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &mut Ciphertext,
    ) -> Ciphertext {
        let mut ct_res = ct_left.clone();
        self.smart_add_crt_assign_parallelized(&mut ct_res, ct_right);
        ct_res
    }

    /// Computes homomorphically a multiplication between two ciphertexts encrypting integer
    /// values in the CRT decomposition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_3;
    /// let size = 3;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&PARAM_MESSAGE_3_CARRY_3, size);
    ///
    /// let clear_1 = 29;
    /// let clear_2 = 23;
    /// let basis = vec![2, 3, 5];
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt(clear_1, basis.clone());
    /// let ctxt_2 = cks.encrypt_crt(clear_2, basis);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.unchecked_mul_crt_assign_parallelized(&mut ctxt_1, &ctxt_2);
    /// // Decrypt
    /// let res = cks.decrypt_crt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_2) % 30, res);
    /// ```
    pub fn unchecked_mul_crt_assign_parallelized(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &Ciphertext,
    ) {
        ct_left
            .ct_vec
            .par_iter_mut()
            .zip(&ct_right.ct_vec)
            .for_each(|(ct_left, ct_right)| {
                self.key.unchecked_mul_lsb_assign(ct_left, ct_right);
            });
    }

    pub fn unchecked_mul_crt_parallelized(
        &self,
        ct_left: &Ciphertext,
        ct_right: &Ciphertext,
    ) -> Ciphertext {
        let mut ct_res = ct_left.clone();
        self.unchecked_mul_crt_assign_parallelized(&mut ct_res, ct_right);
        ct_res
    }

    /// Computes homomorphically a multiplication between two ciphertexts encrypting integer
    /// values in the CRT decomposition.
    ///
    /// This checks that the addition is possible. In the case where the carry buffers are full,
    /// then it is automatically cleared to allow the operation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_3;
    ///
    /// // We have 4 * 2 = 8 bits of message
    /// let size = 4;
    /// let (cks, sks) = gen_keys(&PARAM_MESSAGE_3_CARRY_3, size);
    ///
    /// let clear_1 = 29;
    /// let clear_2 = 29;
    /// let basis = vec![2, 3, 5];
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt(clear_1, basis.clone());
    /// let mut ctxt_2 = cks.encrypt_crt(clear_2, basis);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.smart_mul_crt_assign_parallelized(&mut ctxt_1, &mut ctxt_2);
    ///
    /// // Decrypt
    /// let res = cks.decrypt_crt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_2) % 30, res);
    /// ```
    pub fn smart_mul_crt_assign_parallelized(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &mut Ciphertext,
    ) {
        ct_left
            .ct_vec
            .par_iter_mut()
            .zip(&mut ct_right.ct_vec)
            .for_each(|(block_left, block_right)| {
                self.key.smart_mul_lsb_assign(block_left, block_right);
            });
    }
    pub fn smart_mul_crt_parallelized(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &mut Ciphertext,
    ) -> Ciphertext {
        let mut ct_res = ct_left.clone();
        self.smart_mul_crt_assign_parallelized(&mut ct_res, ct_right);
        ct_res
    }

    /// Computes a PBS for CRT-compliant functions.
    ///
    /// # Warning
    ///
    /// This allows to compute programmable bootstrapping over integers under the condition that
    /// the function is said to be CRT-compliant. This means that the function should be correct
    /// when evaluated on each modular block independently (e.g. arithmetic functions).
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::DEFAULT_PARAMETERS;
    /// let size = 4;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys(&DEFAULT_PARAMETERS, size);
    ///
    /// let clear_1 = 28;
    /// let basis = vec![2, 3, 5];
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt(clear_1, basis);
    ///
    /// // Compute homomorphically the crt-compliant PBS
    /// sks.pbs_crt_compliant_function_assign_parallelized(&mut ctxt_1, |x| x * x * x);
    ///
    /// // Decrypt
    /// let res = cks.decrypt_crt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_1 * clear_1) % 30, res);
    /// ```
    pub fn pbs_crt_compliant_function_assign_parallelized<F>(&self, ct1: &mut Ciphertext, f: F)
    where
        F: Fn(u64) -> u64,
    {
        let basis = &ct1.message_modulus_vec;

        let accumulators = basis
            .iter()
            .copied()
            .map(|b| self.key.generate_accumulator(|x| f(x) % b))
            .collect::<Vec<_>>();

        ct1.ct_vec
            .par_iter_mut()
            .zip(&accumulators)
            .for_each(|(block, acc)| {
                self.key.keyswitch_programmable_bootstrap_assign(block, acc);
            });
    }

    pub fn pbs_crt_compliant_function_parallelized<F>(
        &self,
        ct1: &mut Ciphertext,
        f: F,
    ) -> Ciphertext
    where
        F: Fn(u64) -> u64,
    {
        let mut ct_res = ct1.clone();
        self.pbs_crt_compliant_function_assign_parallelized(&mut ct_res, f);
        ct_res
    }
}
