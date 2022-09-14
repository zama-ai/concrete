use super::ServerKey;
use crate::CrtCiphertext;

impl ServerKey {
    /// Computes homomorphically a multiplication between two ciphertexts encrypting integer
    /// values in the CRT decomposition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys_crt;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_3;
    ///
    /// // Generate the client key and the server key:
    /// let basis = vec![2, 3, 5];
    /// let (cks, sks) = gen_keys_crt(&PARAM_MESSAGE_3_CARRY_3, basis);
    ///
    /// let clear_1 = 29;
    /// let clear_2 = 23;
    ///
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt(clear_1);
    /// let ctxt_2 = cks.encrypt(clear_2);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.unchecked_mul_crt_assign(&mut ctxt_1, &ctxt_2);
    ///
    /// // Decrypt
    /// let res = cks.decrypt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_2) % 30, res);
    /// ```
    pub fn unchecked_mul_crt_assign(&self, ct_left: &mut CrtCiphertext, ct_right: &CrtCiphertext) {
        for (ct_left, ct_right) in ct_left.blocks.iter_mut().zip(ct_right.blocks.iter()) {
            self.key.unchecked_mul_lsb_assign(ct_left, ct_right);
        }
    }

    pub fn unchecked_mul_crt(
        &self,
        ct_left: &CrtCiphertext,
        ct_right: &CrtCiphertext,
    ) -> CrtCiphertext {
        let mut ct_res = ct_left.clone();
        self.unchecked_mul_crt_assign(&mut ct_res, ct_right);
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
    /// use concrete_integer::gen_keys_crt;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_3;
    ///
    /// let basis = vec![2, 3, 5];
    /// let (cks, sks) = gen_keys_crt(&PARAM_MESSAGE_3_CARRY_3, basis);
    ///
    /// let clear_1 = 29;
    /// let clear_2 = 29;
    ///
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt(clear_1);
    /// let mut ctxt_2 = cks.encrypt(clear_2);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.smart_mul_crt_assign(&mut ctxt_1, &mut ctxt_2);
    ///
    /// // Decrypt
    /// let res = cks.decrypt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_2) % 30, res);
    /// ```
    pub fn smart_mul_crt_assign(&self, ct_left: &mut CrtCiphertext, ct_right: &mut CrtCiphertext) {
        for (block_left, block_right) in ct_left.blocks.iter_mut().zip(ct_right.blocks.iter_mut()) {
            self.key.smart_mul_lsb_assign(block_left, block_right);
        }
    }
    pub fn smart_mul_crt(
        &self,
        ct_left: &mut CrtCiphertext,
        ct_right: &mut CrtCiphertext,
    ) -> CrtCiphertext {
        let mut ct_res = ct_left.clone();
        self.smart_mul_crt_assign(&mut ct_res, ct_right);
        ct_res
    }
}
