#[cfg(test)]
mod test;

use crate::ciphertext::KeyId;
use crate::client_key::utils::i_crt;
#[cfg(test)]
use crate::keycache::KEY_CACHE;
use crate::Ciphertext;
#[cfg(test)]
use concrete_shortint::parameters::get_parameters_from_message_and_carry;
use concrete_shortint::parameters::MessageModulus;
use concrete_shortint::server_key::MaxDegree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct CRTVecClientKey {
    pub(crate) key: Vec<concrete_shortint::client_key::ClientKey>,
    pub(crate) key_id: Vec<KeyId>,
}

/// Generates a key_id vector
///
/// Key Ids are used to choose a specific key for each block.
pub fn gen_key_id(id: &[usize]) -> Vec<KeyId> {
    let mut key_id: Vec<KeyId> = vec![];
    for i in id.iter() {
        key_id.push(KeyId(*i));
    }
    key_id
}

/// Generates automatically the appropriate keys regarding the input basis.
/// # Example
///
/// ```rust
/// use concrete_integer::crt::gen_keys_from_basis_and_carry_space;
/// let basis: Vec<u64> = vec![2, 3, 5];
/// let carry_space = vec![4, 4, 4];
/// let (cks, sks) = gen_keys_from_basis_and_carry_space(&basis, &carry_space);
/// ```
#[cfg(test)]
pub fn gen_keys_from_basis_and_carry_space(
    basis: &[u64],
    carry: &[u64],
) -> (CRTVecClientKey, CRTVecServerKey, Vec<KeyId>) {
    let mut vec_param = vec![];
    let mut vec_id = vec![];

    for ((i, base), car) in basis.iter().enumerate().zip(carry) {
        let tmp_param = get_parameters_from_message_and_carry(*base as usize, *car as usize);
        let tmp_param_exists = vec_param.iter().find(|&&x| x == tmp_param);
        if tmp_param_exists != None {
            vec_id.push(vec_param.iter().position(|&x| x == tmp_param).unwrap());
        } else {
            vec_param.push(get_parameters_from_message_and_carry(
                *base as usize,
                *car as usize,
            ));
            vec_id.push(i);
        }
    }
    let vec_key_id = gen_key_id(&vec_id);

    let mut vec_sks = vec![];
    let mut vec_cks = vec![];
    for param in vec_param.iter() {
        let (cks_shortint, sks_shortint) = KEY_CACHE.get_shortint_from_params(*param);
        vec_sks.push(sks_shortint);
        vec_cks.push(cks_shortint);
    }

    (
        CRTVecClientKey::new_many_keys_from_shortint(&vec_cks),
        CRTVecServerKey::new_many_keys_from_shortint(&vec_sks),
        vec_key_id,
    )
}

impl CRTVecClientKey {
    /// Allocates and generates a client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::crt::CRTVecClientKey;
    /// use concrete_shortint::parameters::{
    ///     DEFAULT_PARAMETERS, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key:
    /// let cks =
    ///     CRTVecClientKey::new_many_keys(&vec![PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3]);
    /// ```
    pub fn new_many_keys(
        parameter_set: &[concrete_shortint::parameters::Parameters],
    ) -> CRTVecClientKey {
        let mut key: Vec<concrete_shortint::client_key::ClientKey> = vec![];
        let mut id: Vec<KeyId> = vec![];

        for (i, param) in parameter_set.iter().enumerate() {
            key.push(concrete_shortint::client_key::ClientKey::new(*param));
            id.push(KeyId(i));
        }
        CRTVecClientKey { key, key_id: id }
    }

    /// Generates client keys from exiting block keys (i.e., `concrete-shortint`keys).
    pub fn new_many_keys_from_shortint(
        input_keys: &[concrete_shortint::client_key::ClientKey],
    ) -> CRTVecClientKey {
        //let mut keys: Vec<concrete_shortint::client_key::ClientKey> = vec![];
        let mut id: Vec<KeyId> = vec![];

        for (i, _) in input_keys.iter().enumerate() {
            id.push(KeyId(i));
        }
        CRTVecClientKey {
            key: input_keys.to_vec(),
            key_id: id,
        }
    }

    /// Encrypts an integer using the CRT decomposition, where each block is associated to
    /// dedicated key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::crt::{gen_key_id, CRTVecClientKey};
    /// use concrete_shortint::parameters::{
    ///     DEFAULT_PARAMETERS, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key:
    /// let mut cks =
    ///     CRTVecClientKey::new_many_keys(&vec![PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3]);
    ///
    /// let msg = 15_u64;
    ///
    /// // Encryption of one message:
    /// let basis: Vec<u64> = vec![2, 3, 5];
    /// // first two blocks are encrypted using the first key, third  block with the second key
    /// let keys_id = gen_key_id(&vec![0, 0, 1]);
    /// let mut ct = cks.encrypt_crt_several_keys(&msg, &basis, &keys_id);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt_crt_several_keys(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn encrypt_crt_several_keys(
        &self,
        message: &u64,
        base_vec: &[u64],
        keys_id: &[KeyId],
    ) -> Ciphertext {
        //Empty vector of ciphertexts
        let mut ctxt_vect: Vec<concrete_shortint::ciphertext::Ciphertext> = Vec::new();
        let bv = base_vec.to_vec();
        let keys = keys_id.to_vec();

        //Put each decomposition into a new ciphertext
        for (modulus, id) in base_vec.iter().zip(keys_id.iter()) {
            // encryption
            let ct = self.key[id.0]
                .encrypt_with_message_modulus(*message, MessageModulus(*modulus as usize));

            //put it in the vector of ciphertexts
            ctxt_vect.push(ct);
        }

        Ciphertext {
            ct_vec: ctxt_vect,
            message_modulus_vec: bv,
            key_id_vec: keys,
        }
    }

    /// Decrypts an integer in the multi-key CRT settings.
    ///
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::crt::{gen_key_id, CRTVecClientKey};
    /// use concrete_shortint::parameters::{
    ///     DEFAULT_PARAMETERS, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key:
    /// let mut cks =
    ///     CRTVecClientKey::new_many_keys(&vec![PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3]);
    ///
    /// let msg = 27_u64;
    ///
    /// // Encryption of one message:
    /// let basis: Vec<u64> = vec![2, 3, 5];
    /// // first two blocks are encrypted using the first key, third  block with the second key
    /// let keys_id = gen_key_id(&vec![0, 0, 1]);
    /// let mut ct = cks.encrypt_crt_several_keys(&msg, &basis, &keys_id);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt_crt_several_keys(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn decrypt_crt_several_keys(&self, ctxt: &Ciphertext) -> u64 {
        let mut val: Vec<u64> = vec![];

        //Decrypting each block individually
        for ((c_i, b_i), k_i) in ctxt
            .ct_vec
            .iter()
            .zip(ctxt.message_modulus_vec.iter())
            .zip(ctxt.key_id_vec.iter())
        {
            //decrypt the component i of the integer and multiply it by the radix product
            val.push(self.key[k_i.0].decrypt_message_and_carry(c_i) % b_i);
        }

        // println!("Val = {:?}, basis = {:?}", val, ctxt.message_modulus_vec);

        // Computing the inverse CRT to recompose the message
        let result = i_crt(&ctxt.message_modulus_vec, &val);

        let mut product = 1_u64;
        for b_i in ctxt.message_modulus_vec.iter() {
            product = product.wrapping_mul(*b_i);
        }

        result % product
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CRTVecServerKey {
    pub(crate) key: Vec<concrete_shortint::server_key::ServerKey>,
    pub(crate) key_id: Vec<KeyId>,
}

/// Generate a couple of client and server keys from a vector of cryptographic parameters.
///
/// # Example
///
/// ```rust
/// use concrete_integer::crt::gen_several_keys;
/// use concrete_shortint::parameters::{PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2};
/// let size = 4;
///
/// let id = 0;
///
/// // generate the client key and the server key:
/// let (cks, sks) = gen_several_keys(&vec![PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2]);
/// ```
pub fn gen_several_keys(
    parameters_set: &[concrete_shortint::parameters::Parameters],
) -> (CRTVecClientKey, CRTVecServerKey) {
    // generate the client key
    let cks = CRTVecClientKey::new_many_keys(parameters_set);

    // generate the server key
    let sks = CRTVecServerKey::new_many_keys(&cks);

    // return
    (cks, sks)
}

impl CRTVecServerKey {
    /// Allocates and generates a server key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::crt::{CRTVecClientKey, CRTVecServerKey};
    /// use concrete_shortint::parameters::{PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2};
    ///
    /// // Generate the client key:
    /// let cks =
    ///     CRTVecClientKey::new_many_keys(&vec![PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2]);
    /// let sks = CRTVecServerKey::new_many_keys(&cks);
    /// ```
    pub fn new_many_keys(cks: &CRTVecClientKey) -> CRTVecServerKey {
        let mut vec_sks: Vec<concrete_shortint::server_key::ServerKey> = vec![];
        let mut vec_id: Vec<KeyId> = vec![];

        for (key, id) in cks.key.iter().zip(cks.key_id.iter()) {
            let max = (key.parameters.message_modulus.0 - 1) * key.parameters.carry_modulus.0 - 1;

            let sks =
                concrete_shortint::server_key::ServerKey::new_with_max_degree(key, MaxDegree(max));
            vec_sks.push(sks);
            vec_id.push(*id);
        }
        CRTVecServerKey {
            key: vec_sks,
            key_id: vec_id,
        }
    }

    /// Allocates and generates a server key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::crt::{CRTVecClientKey, CRTVecServerKey};
    /// use concrete_shortint::parameters::{PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2};
    ///
    /// // Generate the client key:
    /// let cks =
    ///     CRTVecClientKey::new_many_keys(&vec![PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2]);
    /// let sks = CRTVecServerKey::new_many_keys(&cks);
    /// ```
    pub fn new_many_keys_from_shortint(sks: &[concrete_shortint::ServerKey]) -> CRTVecServerKey {
        let mut vec_id: Vec<KeyId> = vec![];

        for id in 0..sks.len() {
            vec_id.push(KeyId(id));
        }

        CRTVecServerKey {
            key: sks.to_vec(),
            key_id: vec_id,
        }
    }

    /// Computes an homomorphic addition.
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::crt::{gen_key_id, gen_several_keys};
    /// use concrete_shortint::parameters::{PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1};
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_several_keys(&vec![PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1]);
    ///
    /// let clear_1 = 14;
    /// let clear_2 = 11;
    /// let basis = vec![2, 3, 7];
    /// let keys_id = gen_key_id(&vec![0, 0, 1]);
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt_several_keys(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt_crt_several_keys(&clear_2, &basis, &keys_id);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.unchecked_add_crt_many_keys_assign(&mut ctxt_1, &mut ctxt_2);
    /// // Decrypt
    /// let res = cks.decrypt_crt_several_keys(&ctxt_1);
    /// assert_eq!((clear_1 + clear_2) % 30, res);
    /// ```
    pub fn unchecked_add_crt_many_keys_assign(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &mut Ciphertext,
    ) {
        for ((ct_left, ct_right), key_id) in ct_left
            .ct_vec
            .iter_mut()
            .zip(ct_right.ct_vec.iter_mut())
            .zip(ct_left.key_id_vec.iter())
        {
            self.key[key_id.0].unchecked_add_assign(ct_left, ct_right);
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
    /// use concrete_integer::crt::{gen_key_id, gen_several_keys};
    /// use concrete_shortint::parameters::{PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1};
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_several_keys(&vec![PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_1]);
    ///
    /// let clear_1 = 14;
    /// let clear_2 = 11;
    /// let basis = vec![2, 3, 7];
    /// let keys_id = gen_key_id(&vec![0, 0, 1]);
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt_several_keys(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt_crt_several_keys(&clear_2, &basis, &keys_id);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.unchecked_add_crt_many_keys_assign_parallelized(&mut ctxt_1, &mut ctxt_2);
    /// // Decrypt
    /// let res = cks.decrypt_crt_several_keys(&ctxt_1);
    /// assert_eq!((clear_1 + clear_2) % 30, res);
    /// ```
    pub fn unchecked_add_crt_many_keys_assign_parallelized(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &mut Ciphertext,
    ) {
        ct_left
            .ct_vec
            .par_iter_mut()
            .zip(ct_right.ct_vec.par_iter_mut())
            .zip(ct_left.key_id_vec.par_iter())
            .for_each(|((block_left, block_right), key_id)| {
                let key = &self.key[key_id.0];
                key.unchecked_add_assign(block_left, block_right);
            });
    }

    /// Computes an homomorphic multiplication.
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::crt::{gen_key_id, gen_several_keys};
    /// use concrete_shortint::parameters::{
    ///     PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_several_keys(&vec![
    ///     PARAM_MESSAGE_1_CARRY_1,
    ///     PARAM_MESSAGE_2_CARRY_2,
    ///     PARAM_MESSAGE_3_CARRY_3,
    /// ]);
    ///
    /// let clear_1 = 13;
    /// let clear_2 = 11;
    /// let basis = vec![2, 3, 5];
    /// let keys_id = gen_key_id(&vec![0, 1, 2]);
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt_several_keys(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt_crt_several_keys(&clear_2, &basis, &keys_id);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.unchecked_mul_crt_many_keys_assign(&mut ctxt_1, &mut ctxt_2);
    /// // Decrypt
    /// let res = cks.decrypt_crt_several_keys(&ctxt_1);
    /// assert_eq!((clear_1 * clear_2) % 30, res);
    /// ```
    pub fn unchecked_mul_crt_many_keys_assign(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &mut Ciphertext,
    ) {
        for ((block_left, block_right), id) in ct_left
            .ct_vec
            .iter_mut()
            .zip(ct_right.ct_vec.iter_mut())
            .zip(ct_left.key_id_vec.iter())
        {
            self.key[id.0].unchecked_mul_lsb_assign(block_left, block_right);
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
    /// use concrete_integer::crt::{gen_key_id, gen_several_keys};
    /// use concrete_shortint::parameters::{
    ///     PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_several_keys(&vec![
    ///     PARAM_MESSAGE_1_CARRY_1,
    ///     PARAM_MESSAGE_2_CARRY_2,
    ///     PARAM_MESSAGE_3_CARRY_3,
    /// ]);
    ///
    /// let clear_1 = 13;
    /// let clear_2 = 11;
    /// let basis = vec![2, 3, 5];
    /// let keys_id = gen_key_id(&vec![0, 1, 2]);
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt_several_keys(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt_crt_several_keys(&clear_2, &basis, &keys_id);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.unchecked_mul_crt_many_keys_assign_parallelized(&mut ctxt_1, &mut ctxt_2);
    /// // Decrypt
    /// let res = cks.decrypt_crt_several_keys(&ctxt_1);
    /// assert_eq!((clear_1 * clear_2) % 30, res);
    /// ```
    pub fn unchecked_mul_crt_many_keys_assign_parallelized(
        &self,
        ct_left: &mut Ciphertext,
        ct_right: &mut Ciphertext,
    ) {
        ct_left
            .ct_vec
            .par_iter_mut()
            .zip(ct_right.ct_vec.par_iter_mut())
            .zip(ct_left.key_id_vec.par_iter())
            .for_each(|((block_left, block_right), id)| {
                self.key[id.0].unchecked_mul_lsb_assign(block_left, block_right);
            });
    }

    /// Computes an homomorphic evaluation of an CRT-compliant univariate function.
    ///
    /// # Warning The function has to be CRT-compliant.
    ///
    /// # Example
    ///
    ///```rust
    /// use concrete_integer::crt::{gen_key_id, gen_several_keys};
    /// use concrete_shortint::parameters::{
    ///     PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_several_keys(&vec![
    ///     PARAM_MESSAGE_1_CARRY_1,
    ///     PARAM_MESSAGE_2_CARRY_2,
    ///     PARAM_MESSAGE_3_CARRY_3,
    /// ]);
    ///
    /// let clear_1 = 14;
    /// let clear_2 = 11;
    /// let basis = vec![2, 3, 5];
    /// let keys_id = gen_key_id(&vec![0, 1, 2]);
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt_several_keys(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt_crt_several_keys(&clear_2, &basis, &keys_id);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.arithmetic_function_crt_many_keys_assign(&mut ctxt_1, |x| x * x);
    /// // Decrypt
    /// let res = cks.decrypt_crt_several_keys(&ctxt_1);
    /// assert_eq!((clear_1 * clear_1) % 30, res);
    /// ```
    pub fn arithmetic_function_crt_many_keys_assign<F>(&self, ct: &mut Ciphertext, f: F)
    where
        F: Fn(u64) -> u64,
    {
        let basis = &ct.message_modulus_vec;
        let keys_id = &ct.key_id_vec;
        for ((block, key_id), basis) in ct.ct_vec.iter_mut().zip(keys_id.iter()).zip(basis.iter()) {
            let acc = self.key[key_id.0].generate_accumulator(|x| f(x) % basis);
            self.key[key_id.0].keyswitch_programmable_bootstrap_assign(block, &acc);
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
    /// use concrete_integer::crt::{gen_key_id, gen_several_keys};
    /// use concrete_shortint::parameters::{
    ///     PARAM_MESSAGE_1_CARRY_1, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_several_keys(&vec![
    ///     PARAM_MESSAGE_1_CARRY_1,
    ///     PARAM_MESSAGE_2_CARRY_2,
    ///     PARAM_MESSAGE_3_CARRY_3,
    /// ]);
    ///
    /// let clear_1 = 14;
    /// let clear_2 = 11;
    /// let basis = vec![2, 3, 5];
    /// let keys_id = gen_key_id(&vec![0, 1, 2]);
    /// // Encrypt two messages
    /// let mut ctxt_1 = cks.encrypt_crt_several_keys(&clear_1, &basis, &keys_id);
    /// let mut ctxt_2 = cks.encrypt_crt_several_keys(&clear_2, &basis, &keys_id);
    ///
    /// // Compute homomorphically a multiplication
    /// sks.arithmetic_function_crt_many_keys_assign_parallelized(&mut ctxt_1, |x| x * x);
    /// // Decrypt
    /// let res = cks.decrypt_crt_several_keys(&ctxt_1);
    /// assert_eq!((clear_1 * clear_1) % 30, res);
    /// ```
    pub fn arithmetic_function_crt_many_keys_assign_parallelized<F>(
        &self,
        ct: &mut Ciphertext,
        f: F,
    ) where
        F: Fn(u64) -> u64,
    {
        let basis = &ct.message_modulus_vec;
        let keys_id = &ct.key_id_vec;

        let accumulators = keys_id
            .iter()
            .zip(basis.iter())
            .map(|(key_id, basis)| self.key[key_id.0].generate_accumulator(|x| f(x) % basis))
            .collect::<Vec<_>>();

        ct.ct_vec
            .par_iter_mut()
            .zip(keys_id)
            .zip(&accumulators)
            .for_each(|((block, key_id), acc)| {
                self.key[key_id.0].keyswitch_programmable_bootstrap_assign(block, acc);
            });
    }
}
