mod add;
mod mul;

use crate::ciphertext::CrtCiphertext;
use crate::ServerKey;
use rayon::prelude::*;

impl ServerKey {
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
    /// use concrete_integer::gen_keys_crt;
    /// use concrete_shortint::parameters::DEFAULT_PARAMETERS;
    ///
    /// // Generate the client key and the server key:
    /// let basis = vec![2, 3, 5];
    /// let (cks, sks) = gen_keys_crt(&DEFAULT_PARAMETERS, basis);
    ///
    /// let clear_1 = 28;
    ///
    /// let mut ctxt_1 = cks.encrypt(clear_1);
    ///
    /// // Compute homomorphically the crt-compliant PBS
    /// sks.pbs_crt_compliant_function_assign_parallelized(&mut ctxt_1, |x| x * x * x);
    ///
    /// // Decrypt
    /// let res = cks.decrypt(&ctxt_1);
    /// assert_eq!((clear_1 * clear_1 * clear_1) % 30, res);
    /// ```
    pub fn pbs_crt_compliant_function_assign_parallelized<F>(&self, ct1: &mut CrtCiphertext, f: F)
    where
        F: Fn(u64) -> u64,
    {
        let basis = &ct1.moduli;

        let accumulators = basis
            .iter()
            .copied()
            .map(|b| self.key.generate_accumulator(|x| f(x) % b))
            .collect::<Vec<_>>();

        ct1.blocks
            .par_iter_mut()
            .zip(&accumulators)
            .for_each(|(block, acc)| {
                self.key.keyswitch_programmable_bootstrap_assign(block, acc);
            });
    }

    pub fn pbs_crt_compliant_function_parallelized<F>(
        &self,
        ct1: &mut CrtCiphertext,
        f: F,
    ) -> CrtCiphertext
    where
        F: Fn(u64) -> u64,
    {
        let mut ct_res = ct1.clone();
        self.pbs_crt_compliant_function_assign_parallelized(&mut ct_res, f);
        ct_res
    }
}
