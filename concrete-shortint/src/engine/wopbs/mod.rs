//!
//! # WARNING: this module is experimental.
use crate::ciphertext::Degree;
use crate::engine::{EngineResult, ShortintEngine};
use crate::wopbs::WopbsKey;
use crate::{Ciphertext, ClientKey, ServerKey};
use concrete_core::backends::core::private::crypto::circuit_bootstrap::DeltaLog;
use concrete_core::backends::core::private::crypto::glwe::FunctionalPackingKeyswitchKey;
use concrete_core::backends::core::private::crypto::lwe::LweCiphertext;
use concrete_core::backends::core::private::crypto::vertical_packing::vertical_packing_cbs_binary_v0;
use concrete_core::backends::core::private::crypto::wop_pbs_vp::extract_bit_v0_v1;
use concrete_core::prelude::{
    CleartextVectorCreationEngine, FunctionalPackingKeyswitchKeyCreationEngine,
    GlweSecretKeyEntity, LweCiphertext64,
};

impl ShortintEngine {
    //Fills with functional packing key switch keys.
    pub(crate) fn new_wopbs_key_v0(
        &mut self,
        cks: &ClientKey,
        sks: &ServerKey,
    ) -> EngineResult<WopbsKey> {
        //Generation of the private functional packing keyswitch for the circuit bootstrap
        let mut vec_pfks_key: Vec<FunctionalPackingKeyswitchKey<Vec<u64>>> = vec![];

        //Here the cleartext represents a polynomial
        for key_coef in cks.glwe_secret_key.0.as_polynomial_list().polynomial_iter() {
            let mut vec_cleartext: Vec<u64> = vec![];
            for key_coef_monomial in key_coef.coefficient_iter() {
                vec_cleartext.push(*key_coef_monomial);
            }
            let polynomial_as_cleartext = self
                .engine
                .create_cleartext_vector(vec_cleartext.as_slice())?;

            vec_pfks_key.push(
                self.engine
                    .create_functional_packing_keyswitch_key(
                        &cks.lwe_secret_key,
                        &cks.glwe_secret_key,
                        cks.parameters.pfks_level,
                        cks.parameters.pfks_base_log,
                        cks.parameters.pfks_modular_std_dev,
                        |x| 0_u64.wrapping_sub(x),
                        &polynomial_as_cleartext,
                    )
                    .unwrap()
                    .0,
            );
        }
        let mut vec_tmp: Vec<u64> = vec![0_u64; cks.glwe_secret_key.polynomial_size().0];
        vec_tmp[0] = 1;
        let polynomial_as_cleartext = self.engine.create_cleartext_vector(&vec_tmp)?;
        vec_pfks_key.push(
            self.engine
                .create_functional_packing_keyswitch_key(
                    &cks.lwe_secret_key,
                    &cks.glwe_secret_key,
                    cks.parameters.pfks_level,
                    cks.parameters.pfks_base_log,
                    cks.parameters.pfks_modular_std_dev,
                    |x| x,
                    &polynomial_as_cleartext,
                )
                .unwrap()
                .0,
        );

        // This will be updated
        let small_bsk = sks.bootstrapping_key.clone();

        let wopbs_key = WopbsKey {
            vec_pfks_key,
            small_bsk,
            param: cks.parameters,
        };
        Ok(wopbs_key)
    }

    pub(crate) fn extract_bit(
        &mut self,
        delta_log: DeltaLog,
        lwe_in: &mut LweCiphertext<Vec<u64>>,
        server_key: &ServerKey,
        number_values_to_extract: usize,
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        let (buffers, _) = self.buffers_for_key(server_key);
        extract_bit_v0_v1(
            delta_log,
            lwe_in,
            &server_key.key_switching_key.0,
            &server_key.bootstrapping_key.0,
            &mut buffers.fourier,
            number_values_to_extract,
        )
    }

    pub(crate) fn circuit_bootstrap_vertical_packing(
        &mut self,
        wopbs_key: &WopbsKey,
        sks: &ServerKey,
        ct_in: &mut Ciphertext,
        lut: &[Vec<u64>],
    ) -> EngineResult<Vec<Ciphertext>> {
        let delta = (1_usize << 63) / (sks.message_modulus.0 * sks.carry_modulus.0);
        let delta_log = DeltaLog(f64::log2(delta as f64) as usize);
        println!("delta log = {}", delta_log.0);

        let (buffers, _engine) = self.buffers_for_key(sks);

        let nb_bit_to_extract =
            f64::log2((sks.message_modulus.0 * sks.carry_modulus.0) as f64) as usize;

        let mut vec_lwe = extract_bit_v0_v1(
            delta_log,
            &mut ct_in.ct.0,
            &sks.key_switching_key.0,
            &sks.bootstrapping_key.0,
            &mut buffers.fourier,
            nb_bit_to_extract,
        );

        vec_lwe.reverse();

        let vec_ct_out = vertical_packing_cbs_binary_v0(
            lut.to_vec(),
            &mut buffers.fourier,
            &wopbs_key.small_bsk.0,
            &vec_lwe.clone(),
            wopbs_key.param.cbs_level,
            wopbs_key.param.cbs_base_log,
            &wopbs_key.vec_pfks_key,
        );

        let mut result: Vec<Ciphertext> = vec![];
        for lwe in vec_ct_out.iter() {
            let tmp = lwe.clone();
            result.push(Ciphertext {
                ct: LweCiphertext64(tmp),
                degree: Degree(sks.message_modulus.0 - 1),
                message_modulus: sks.message_modulus,
                carry_modulus: sks.carry_modulus,
            })
        }
        Ok(result)
    }

    pub(crate) fn circuit_bootstrap_vertical_packing_without_padding(
        &mut self,
        wopbs_key: &WopbsKey,
        sks: &ServerKey,
        ct_in: &mut Ciphertext,
        lut: &[Vec<u64>],
    ) -> EngineResult<Vec<Ciphertext>> {
        //NO PADDING BIT
        let delta = (1_usize << 63) / (sks.message_modulus.0 * sks.carry_modulus.0) * 2;
        let delta_log = DeltaLog(f64::log2(delta as f64) as usize);
        //println!("delta log = {}", delta_log.0);

        let (buffers, _) = self.buffers_for_key(sks);

        let nb_bit_to_extract =
            f64::log2((sks.message_modulus.0 * sks.carry_modulus.0) as f64) as usize;

        let mut vec_lwe = extract_bit_v0_v1(
            delta_log,
            &mut ct_in.ct.0,
            &sks.key_switching_key.0,
            &sks.bootstrapping_key.0,
            &mut buffers.fourier,
            nb_bit_to_extract,
        );

        vec_lwe.reverse();

        let vec_ct_out = vertical_packing_cbs_binary_v0(
            lut.to_vec(),
            &mut buffers.fourier,
            &wopbs_key.small_bsk.0,
            &vec_lwe.clone(),
            wopbs_key.param.cbs_level,
            wopbs_key.param.cbs_base_log,
            &wopbs_key.vec_pfks_key,
        );

        let mut result: Vec<Ciphertext> = vec![];
        for lwe in vec_ct_out.iter() {
            let tmp = lwe.clone();
            result.push(Ciphertext {
                ct: LweCiphertext64(tmp),
                degree: Degree(sks.message_modulus.0 - 1),
                message_modulus: sks.message_modulus,
                carry_modulus: sks.carry_modulus,
            })
        }
        Ok(result)
    }

    /// Temporary wrapper used in `concrete-integer`.
    ///
    /// # Warning Experimental
    pub fn vertical_packing_cbs_binary_v0(
        &mut self,
        wopbs_key: &WopbsKey,
        server_key: &ServerKey,
        vec_lut: Vec<Vec<u64>>,
        vec_lwe_in: &[LweCiphertext<Vec<u64>>],
    ) -> Vec<LweCiphertext<Vec<u64>>> {
        let (buffers, _) = self.buffers_for_key(server_key);
        vertical_packing_cbs_binary_v0(
            vec_lut,
            &mut buffers.fourier,
            &wopbs_key.small_bsk.0,
            vec_lwe_in,
            wopbs_key.param.cbs_level,
            wopbs_key.param.cbs_base_log,
            &wopbs_key.vec_pfks_key,
        )
    }
}
