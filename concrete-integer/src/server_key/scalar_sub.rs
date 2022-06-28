use crate::ciphertext::Ciphertext;
use crate::server_key::CheckError;
use crate::server_key::CheckError::CarryFull;
use crate::ServerKey;

impl ServerKey {
    /// Computes homomorphically a subtraction between a ciphertext and a scalar.
    ///
    /// This function computes the operation without checking if it exceeds the capacity of the
    /// ciphertext.
    ///
    /// The result is returned as a new ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// // We have 4 * 2 = 8 bits of message
    /// let size = 4;
    /// let (cks, sks) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, size);
    ///
    /// let msg = 40;
    /// let scalar = 3;
    ///
    /// let ct = cks.encrypt(msg);
    ///
    /// // Compute homomorphically an addition:
    /// let ct_res = sks.unchecked_scalar_sub(&ct, scalar);
    ///
    /// // Decrypt:
    /// let dec = cks.decrypt(&ct_res);
    /// assert_eq!(msg - scalar, dec);
    /// ```
    pub fn unchecked_scalar_sub(&self, ct: &Ciphertext, scalar: u64) -> Ciphertext {
        let mut result = ct.clone();
        self.unchecked_scalar_sub_assign(&mut result, scalar);
        result
    }

    pub fn unchecked_scalar_sub_assign(&self, ct: &mut Ciphertext, scalar: u64) {
        //Bits of message put to 1
        let mask = (self.key.message_modulus.0 - 1) as u64;

        let modulus = self.key.message_modulus.0.pow(ct.ct_vec.len() as u32) as u64;

        let neg_scalar = scalar.wrapping_neg() % modulus;

        let mut power = 1_u64;
        //Put each decomposition into a new ciphertext
        for ct_i in ct.ct_vec.iter_mut() {
            let mut decomp = neg_scalar & (mask * power);
            decomp /= power;

            self.key.unchecked_scalar_add_assign(ct_i, decomp as u8);

            //modulus to the power i
            power *= self.key.message_modulus.0 as u64;
        }
    }

    /// Verifies if the subtraction of a ciphertext by scalar can be computed.
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// // We have 4 * 2 = 8 bits of message
    /// let size = 4;
    /// let (cks, sks) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, size);
    ///
    /// let msg = 40;
    /// let scalar = 2;
    ///
    /// let ct1 = cks.encrypt(msg);
    ///
    /// // Check if we can perform an addition
    /// let res = sks.is_scalar_sub_possible(&ct1, scalar);
    ///
    /// assert_eq!(true, res);
    /// ```
    pub fn is_scalar_sub_possible(&self, ct: &Ciphertext, scalar: u64) -> bool {
        //Bits of message put to 1
        let mask = (self.key.message_modulus.0 - 1) as u64;

        let modulus = self.key.message_modulus.0.pow(ct.ct_vec.len() as u32) as u64;

        let neg_scalar = scalar.wrapping_neg() % modulus;

        let mut power = 1_u64;

        for ct_i in ct.ct_vec.iter() {
            let mut decomp = neg_scalar & (mask * power);
            decomp /= power;

            if !self.key.is_scalar_add_possible(ct_i, decomp as u8) {
                return false;
            }

            //modulus to the power i
            power *= self.key.message_modulus.0 as u64;
        }
        true
    }

    /// Computes homomorphically a subtraction of a ciphertext by a scalar.
    ///
    /// If the operation can be performed, the result is returned in a new ciphertext.
    /// Otherwise [CheckError::CarryFull] is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// // We have 4 * 2 = 8 bits of message
    /// let size = 4;
    /// let (cks, sks) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, size);
    ///
    /// let msg = 40;
    /// let scalar = 4;
    ///
    /// let ct = cks.encrypt(msg);
    ///
    /// // Compute tne subtraction:
    /// let ct_res = sks.checked_scalar_sub(&ct, scalar)?;
    ///
    /// // Decrypt:
    /// let dec = cks.decrypt(&ct_res);
    /// assert_eq!(msg - scalar, dec);
    /// # Ok(())
    /// # }
    /// ```
    pub fn checked_scalar_sub(
        &self,
        ct: &Ciphertext,
        scalar: u64,
    ) -> Result<Ciphertext, CheckError> {
        if self.is_scalar_sub_possible(ct, scalar) {
            Ok(self.unchecked_scalar_sub(ct, scalar))
        } else {
            Err(CarryFull)
        }
    }

    /// Computes homomorphically a subtraction of a ciphertext by a scalar.
    ///
    /// If the operation can be performed, the result is returned in a new ciphertext.
    /// Otherwise [CheckError::CarryFull] is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// // We have 4 * 2 = 8 bits of message
    /// let size = 4;
    /// let (cks, sks) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, size);
    ///
    /// let msg = 232;
    /// let scalar = 83;
    ///
    /// let mut ct = cks.encrypt(msg);
    ///
    /// // Compute tne subtraction:
    /// sks.checked_scalar_sub_assign(&mut ct, scalar)?;
    ///
    /// // Decrypt:
    /// let dec = cks.decrypt(&ct);
    /// assert_eq!(msg - scalar, dec);
    /// # Ok(())
    /// # }
    /// ```
    pub fn checked_scalar_sub_assign(
        &self,
        ct: &mut Ciphertext,
        scalar: u64,
    ) -> Result<(), CheckError> {
        if self.is_scalar_sub_possible(ct, scalar) {
            self.unchecked_scalar_sub_assign(ct, scalar);
            Ok(())
        } else {
            Err(CarryFull)
        }
    }

    /// Computes homomorphically a subtraction of a ciphertext by a scalar.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::gen_keys;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// // We have 4 * 2 = 8 bits of message
    /// let size = 4;
    /// let (cks, sks) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, size);
    ///
    /// let msg = 165;
    /// let scalar = 112;
    ///
    /// let mut ct = cks.encrypt(msg);
    ///
    /// // Compute homomorphically an addition:
    /// let ct_res = sks.smart_scalar_sub(&mut ct, scalar);
    ///
    /// // Decrypt:
    /// let dec = cks.decrypt(&ct_res);
    /// assert_eq!(msg - scalar, dec);
    /// ```
    pub fn smart_scalar_sub(&self, ct: &mut Ciphertext, scalar: u64) -> Ciphertext {
        if !self.is_scalar_sub_possible(ct, scalar) {
            self.full_propagate(ct);
        }

        self.unchecked_scalar_sub(ct, scalar)
    }

    pub fn smart_scalar_sub_assign(&self, ct: &mut Ciphertext, scalar: u64) {
        if !self.is_scalar_sub_possible(ct, scalar) {
            self.full_propagate(ct);
        }

        self.unchecked_scalar_sub_assign(ct, scalar);
    }
}
