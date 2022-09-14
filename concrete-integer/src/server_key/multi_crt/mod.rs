#[cfg(test)]
mod test;

use crate::ciphertext::KeyId;
use crate::{CrtMultiCiphertext, CrtMultiClientKey};
use concrete_shortint::server_key::MaxDegree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrtMultiServerKey {
    pub(crate) keys: Vec<concrete_shortint::server_key::ServerKey>,
    pub(crate) key_ids: Vec<KeyId>,
}

impl From<Vec<concrete_shortint::ServerKey>> for CrtMultiServerKey {
    fn from(keys: Vec<concrete_shortint::ServerKey>) -> Self {
        let key_ids = (0..keys.len()).map(KeyId).collect();

        Self { keys, key_ids }
    }
}

impl CrtMultiServerKey {
    /// Allocates and generates a server key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::{CrtMultiClientKey, CrtMultiServerKey};
    /// use concrete_shortint::parameters::{PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2};
    ///
    /// // Generate the client key:
    /// let cks = CrtMultiClientKey::new_many_keys(&[PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2]);
    /// let sks = CrtMultiServerKey::new_many_keys(&cks);
    /// ```
    pub fn new_many_keys(cks: &CrtMultiClientKey) -> CrtMultiServerKey {
        let mut vec_sks = Vec::with_capacity(cks.keys.len());
        let mut vec_id = Vec::with_capacity(cks.keys.len());

        for (key, id) in cks.keys.iter().zip(cks.key_ids.iter()) {
            let max = (key.parameters.message_modulus.0 - 1) * key.parameters.carry_modulus.0 - 1;

            let sks = concrete_shortint::ServerKey::new_with_max_degree(key, MaxDegree(max));
            vec_sks.push(sks);
            vec_id.push(*id);
        }
        CrtMultiServerKey {
            keys: vec_sks,
            key_ids: vec_id,
        }
    }

    /// Computes an homomorphic addition.
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::{gen_key_id, gen_keys_multi_crt};
    /// use concrete_shortint::parameters::{PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1};
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_multi_crt(&[PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1]);
    ///
    /// let clear_1 = 14;
    /// let clear_2 = 11;
    ///
    /// let basis = vec![2, 3, 7];
    /// let keys_id = gen_key_id(&[0, 0, 1]);
    ///
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt(&clear_2, &basis, &keys_id);
    ///
    /// sks.unchecked_add_crt_many_keys_assign(&mut ctxt_1, &mut ctxt_2);
    ///
    /// // Decrypt
    /// let res = cks.decrypt(&ctxt_1);
    /// assert_eq!((clear_1 + clear_2) % 30, res);
    /// ```
    pub fn unchecked_add_crt_many_keys_assign(
        &self,
        ct_left: &mut CrtMultiCiphertext,
        ct_right: &mut CrtMultiCiphertext,
    ) {
        for ((ct_left, ct_right), key_id) in ct_left
            .blocks
            .iter_mut()
            .zip(ct_right.blocks.iter_mut())
            .zip(ct_left.key_ids.iter())
        {
            self.keys[key_id.0].unchecked_add_assign(ct_left, ct_right);
        }
    }

    /// Computes an homomorphic addition.
    ///
    /// # Warning
    ///
    /// Multithreaded
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::{gen_key_id, gen_keys_multi_crt};
    /// use concrete_shortint::parameters::{PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1};
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_multi_crt(&[PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1]);
    ///
    /// let clear_1 = 14;
    /// let clear_2 = 11;
    ///
    /// let basis = vec![2, 3, 7];
    /// let keys_id = gen_key_id(&[0, 0, 1]);
    ///
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt(&clear_2, &basis, &keys_id);
    ///
    /// sks.unchecked_add_crt_many_keys_assign_parallelized(&mut ctxt_1, &mut ctxt_2);
    ///
    /// // Decrypt
    /// let res = cks.decrypt(&ctxt_1);
    /// assert_eq!((clear_1 + clear_2) % 30, res);
    /// ```
    pub fn unchecked_add_crt_many_keys_assign_parallelized(
        &self,
        ct_left: &mut CrtMultiCiphertext,
        ct_right: &mut CrtMultiCiphertext,
    ) {
        ct_left
            .blocks
            .par_iter_mut()
            .zip(ct_right.blocks.par_iter_mut())
            .zip(ct_left.key_ids.par_iter())
            .for_each(|((block_left, block_right), key_id)| {
                let key = &self.keys[key_id.0];
                key.unchecked_add_assign(block_left, block_right);
            });
    }

    /// Computes an homomorphic multiplication.
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::{gen_key_id, gen_keys_multi_crt};
    /// use concrete_shortint::parameters::{
    ///     PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_multi_crt(&vec![
    ///     PARAM_MESSAGE_1_CARRY_1,
    ///     PARAM_MESSAGE_2_CARRY_2,
    ///     PARAM_MESSAGE_3_CARRY_3,
    /// ]);
    ///
    /// let clear_1 = 13;
    /// let clear_2 = 11;
    ///
    /// let basis = vec![2, 3, 5];
    /// let keys_id = gen_key_id(&[0, 1, 2]);
    ///
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt(&clear_2, &basis, &keys_id);
    ///
    /// sks.unchecked_mul_crt_many_keys_assign(&mut ctxt_1, &mut ctxt_2);
    ///
    /// // Decrypt
    /// let res = cks.decrypt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_2) % 30, res);
    /// ```
    pub fn unchecked_mul_crt_many_keys_assign(
        &self,
        ct_left: &mut CrtMultiCiphertext,
        ct_right: &mut CrtMultiCiphertext,
    ) {
        for ((block_left, block_right), id) in ct_left
            .blocks
            .iter_mut()
            .zip(ct_right.blocks.iter_mut())
            .zip(ct_left.key_ids.iter())
        {
            self.keys[id.0].unchecked_mul_lsb_assign(block_left, block_right);
        }
    }

    /// Computes an homomorphic multiplication.
    ///
    /// # Warning
    ///
    /// Multithreaded
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::{gen_key_id, gen_keys_multi_crt};
    /// use concrete_shortint::parameters::{
    ///     PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_multi_crt(&vec![
    ///     PARAM_MESSAGE_1_CARRY_1,
    ///     PARAM_MESSAGE_2_CARRY_2,
    ///     PARAM_MESSAGE_3_CARRY_3,
    /// ]);
    ///
    /// let clear_1 = 13;
    /// let clear_2 = 11;
    ///
    /// let basis = vec![2, 3, 5];
    /// let keys_id = gen_key_id(&[0, 1, 2]);
    ///
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt(&clear_2, &basis, &keys_id);
    ///
    /// sks.unchecked_mul_crt_many_keys_assign_parallelized(&mut ctxt_1, &mut ctxt_2);
    ///
    /// // Decrypt
    /// let res = cks.decrypt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_2) % 30, res);
    /// ```
    pub fn unchecked_mul_crt_many_keys_assign_parallelized(
        &self,
        ct_left: &mut CrtMultiCiphertext,
        ct_right: &mut CrtMultiCiphertext,
    ) {
        ct_left
            .blocks
            .par_iter_mut()
            .zip(ct_right.blocks.par_iter_mut())
            .zip(ct_left.key_ids.par_iter())
            .for_each(|((block_left, block_right), id)| {
                self.keys[id.0].unchecked_mul_lsb_assign(block_left, block_right);
            });
    }

    /// Computes an homomorphic evaluation of an CRT-compliant univariate function.
    ///
    /// # Warning The function has to be CRT-compliant.
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::{gen_key_id, gen_keys_multi_crt};
    /// use concrete_shortint::parameters::{
    ///     PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_multi_crt(&vec![
    ///     PARAM_MESSAGE_1_CARRY_1,
    ///     PARAM_MESSAGE_2_CARRY_2,
    ///     PARAM_MESSAGE_3_CARRY_3,
    /// ]);
    ///
    /// let clear_1 = 14;
    /// let clear_2 = 11;
    ///
    /// let basis = vec![2, 3, 5];
    /// let keys_id = gen_key_id(&[0, 1, 2]);
    ///
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt(&clear_2, &basis, &keys_id);
    ///
    /// sks.arithmetic_function_crt_many_keys_assign(&mut ctxt_1, |x| x * x);
    ///
    /// // Decrypt
    /// let res = cks.decrypt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_1) % 30, res);
    /// ```
    pub fn arithmetic_function_crt_many_keys_assign<F>(&self, ct: &mut CrtMultiCiphertext, f: F)
    where
        F: Fn(u64) -> u64,
    {
        let basis = &ct.moduli;
        let keys_id = &ct.key_ids;
        for ((block, key_id), basis) in ct.blocks.iter_mut().zip(keys_id.iter()).zip(basis.iter()) {
            let acc = self.keys[key_id.0].generate_accumulator(|x| f(x) % basis);
            self.keys[key_id.0].keyswitch_programmable_bootstrap_assign(block, &acc);
        }
    }

    /// Computes an homomorphic evaluation of an CRT-compliant univariate function.
    ///
    /// # Warning
    ///
    /// - Multithreaded
    /// - The function has to be CRT-compliant.
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::{gen_key_id, gen_keys_multi_crt};
    /// use concrete_shortint::parameters::{
    ///     PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_multi_crt(&vec![
    ///     PARAM_MESSAGE_1_CARRY_1,
    ///     PARAM_MESSAGE_2_CARRY_2,
    ///     PARAM_MESSAGE_3_CARRY_3,
    /// ]);
    ///
    /// let clear_1 = 14;
    /// let clear_2 = 11;
    ///
    /// let basis = vec![2, 3, 5];
    /// let keys_id = gen_key_id(&[0, 1, 2]);
    ///
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt(&clear_2, &basis, &keys_id);
    ///
    /// sks.arithmetic_function_crt_many_keys_assign_parallelized(&mut ctxt_1, |x| x * x);
    ///
    /// // Decrypt
    /// let res = cks.decrypt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_1) % 30, res);
    /// ```
    pub fn arithmetic_function_crt_many_keys_assign_parallelized<F>(
        &self,
        ct: &mut CrtMultiCiphertext,
        f: F,
    ) where
        F: Fn(u64) -> u64,
    {
        let basis = &ct.moduli;
        let keys_id = &ct.key_ids;

        let accumulators = keys_id
            .iter()
            .zip(basis.iter())
            .map(|(key_id, basis)| self.keys[key_id.0].generate_accumulator(|x| f(x) % basis))
            .collect::<Vec<_>>();

        ct.blocks
            .par_iter_mut()
            .zip(keys_id)
            .zip(&accumulators)
            .for_each(|((block, key_id), acc)| {
                self.keys[key_id.0].keyswitch_programmable_bootstrap_assign(block, acc);
            });
    }
}
