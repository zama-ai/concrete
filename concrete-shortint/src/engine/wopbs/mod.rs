//! # WARNING: this module is experimental.
use crate::ciphertext::Degree;
use crate::engine::{EngineResult, ShortintEngine};
use crate::wopbs::WopbsKey;
use crate::{Ciphertext, ClientKey, ServerKey};

use concrete_core::prelude::*;

impl ShortintEngine {
    //Fills with functional packing key switch keys.
    pub(crate) fn new_wopbs_key(
        &mut self,
        cks: &ClientKey,
        sks: &ServerKey,
    ) -> EngineResult<WopbsKey> {
        let small_bsk = sks.bootstrapping_key.clone();

        let cbs_pfpksk = self
            .engine
            .generate_new_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
                &cks.lwe_secret_key,
                &cks.glwe_secret_key,
                cks.parameters.pfks_base_log,
                cks.parameters.pfks_level,
                Variance(cks.parameters.pfks_modular_std_dev.get_variance()),
            )?;

        let wopbs_key = WopbsKey {
            small_bsk,
            cbs_pfpksk,
            param: cks.parameters,
        };
        Ok(wopbs_key)
    }

    pub(crate) fn extract_bits(
        &mut self,
        delta_log: DeltaLog,
        lwe_in: &LweCiphertext64,
        server_key: &ServerKey,
        extracted_bit_count: ExtractedBitsCount,
    ) -> EngineResult<LweCiphertextVector64> {
        let lwe_size = server_key
            .key_switching_key
            .output_lwe_dimension()
            .to_lwe_size();
        let mut output = self.engine.create_lwe_ciphertext_vector_from(
            vec![0u64; lwe_size.0 * extracted_bit_count.0],
            lwe_size,
        )?;

        self.fft_engine.discard_extract_bits_lwe_ciphertext(
            &mut output,
            lwe_in,
            &server_key.bootstrapping_key,
            &server_key.key_switching_key,
            extracted_bit_count,
            DeltaLog(delta_log.0),
        )?;
        Ok(output)
    }

    pub(crate) fn circuit_bootstrap_with_bits(
        &mut self,
        wopbs_key: &WopbsKey,
        sks: &ServerKey,
        extracted_bits: &LweCiphertextVectorView64<'_>,
        lut: &PlaintextVector64,
        count: LweCiphertextCount,
    ) -> EngineResult<LweCiphertextVector64> {
        let mut output_cbs_vp_ct_container =
            vec![0u64; sks.bootstrapping_key.output_lwe_dimension().to_lwe_size().0 * count.0];

        let mut output_cbs_vp_ct = self.engine.create_lwe_ciphertext_vector_from(
            output_cbs_vp_ct_container.as_mut_slice(),
            sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        )?;

        self.fft_engine
            .discard_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_vector(
                &mut output_cbs_vp_ct,
                extracted_bits,
                &sks.bootstrapping_key,
                lut,
                wopbs_key.param.cbs_level,
                wopbs_key.param.cbs_base_log,
                &wopbs_key.cbs_pfpksk,
            )?;

        let output_vector = self.engine.create_lwe_ciphertext_vector_from(
            output_cbs_vp_ct_container,
            sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        )?;

        Ok(output_vector)
    }

    pub(crate) fn extract_bits_circuit_bootstrapping(
        &mut self,
        wopbs_key: &WopbsKey,
        sks: &ServerKey,
        ct_in: &Ciphertext,
        lut: &[u64],
        delta_log: DeltaLog,
        nb_bit_to_extract: ExtractedBitsCount,
    ) -> EngineResult<Ciphertext> {
        let extracted_bits = self.extract_bits(delta_log, &ct_in.ct, sks, nb_bit_to_extract)?;

        let data = self
            .engine
            .consume_retrieve_lwe_ciphertext_vector(extracted_bits)?;
        let extrated_bits_view = self.engine.create_lwe_ciphertext_vector_from(
            data.as_slice(),
            ct_in.ct.lwe_dimension().to_lwe_size(),
        )?;

        let plaintext_lut = self.engine.create_plaintext_vector_from(lut)?;

        let ciphertext = self.circuit_bootstrap_with_bits(
            wopbs_key,
            sks,
            &extrated_bits_view,
            &plaintext_lut,
            LweCiphertextCount(1),
        )?;

        let container = self
            .engine
            .consume_retrieve_lwe_ciphertext_vector(ciphertext)?;
        let ct = self.engine.create_lwe_ciphertext_from(container)?;

        Ok(Ciphertext {
            ct,
            degree: Degree(sks.message_modulus.0 - 1),
            message_modulus: sks.message_modulus,
            carry_modulus: sks.carry_modulus,
        })
    }

    pub(crate) fn programmable_bootstrapping_without_padding(
        &mut self,
        wopbs_key: &WopbsKey,
        sks: &ServerKey,
        ct_in: &Ciphertext,
        lut: &[u64],
    ) -> EngineResult<Ciphertext> {
        let delta = (1_usize << 63) / (sks.message_modulus.0 * sks.carry_modulus.0) * 2;
        let delta_log = DeltaLog(f64::log2(delta as f64) as usize);

        let nb_bit_to_extract =
            f64::log2((sks.message_modulus.0 * sks.carry_modulus.0) as f64) as usize;

        let ciphertext = self.extract_bits_circuit_bootstrapping(
            wopbs_key,
            sks,
            ct_in,
            lut,
            delta_log,
            ExtractedBitsCount(nb_bit_to_extract),
        )?;

        Ok(ciphertext)
    }

    pub(crate) fn programmable_bootstrapping(
        &mut self,
        wopbs_key: &WopbsKey,
        sks: &ServerKey,
        ct_in: &Ciphertext,
        lut: &[u64],
    ) -> EngineResult<Ciphertext> {
        let delta = (1_usize << 63) / (sks.message_modulus.0 * sks.carry_modulus.0);
        let delta_log = DeltaLog(f64::log2(delta as f64) as usize);

        let nb_bit_to_extract =
            f64::log2((sks.message_modulus.0 * sks.carry_modulus.0) as f64) as usize;

        let ciphertext = self.extract_bits_circuit_bootstrapping(
            wopbs_key,
            sks,
            ct_in,
            lut,
            delta_log,
            ExtractedBitsCount(nb_bit_to_extract),
        )?;

        Ok(ciphertext)
    }

    pub(crate) fn programmable_bootstrapping_native_crt(
        &mut self,
        wopbs_key: &WopbsKey,
        sks: &ServerKey,
        ct_in: &mut Ciphertext,
        lut: &[u64],
    ) -> EngineResult<Ciphertext> {
        let nb_bit_to_extract =
            f64::log2((ct_in.message_modulus.0 * ct_in.carry_modulus.0) as f64).ceil() as usize;
        let delta_log = DeltaLog(64 - nb_bit_to_extract);

        // trick ( ct - delta/2 + delta/2^4  )
        let lwe_size = ct_in.ct.lwe_dimension().to_lwe_size().0;
        let mut cont = vec![0u64; lwe_size];
        cont[lwe_size - 1] =
            (1 << (64 - nb_bit_to_extract - 1)) - (1 << (64 - nb_bit_to_extract - 5));
        let tmp = self.engine.create_lwe_ciphertext_from(cont)?;
        self.engine.fuse_sub_lwe_ciphertext(&mut ct_in.ct, &tmp)?;

        let ciphertext = self.extract_bits_circuit_bootstrapping(
            wopbs_key,
            sks,
            ct_in,
            lut,
            delta_log,
            ExtractedBitsCount(nb_bit_to_extract),
        )?;

        Ok(ciphertext)
    }

    /// Temporary wrapper used in `concrete-integer`.
    ///
    /// # Warning Experimental
    pub fn circuit_bootstrapping_vertical_packing(
        &mut self,
        wopbs_key: &WopbsKey,
        server_key: &ServerKey,
        vec_lut: Vec<Vec<u64>>,
        extracted_bits_blocks: Vec<LweCiphertextVector64>,
    ) -> Vec<LweCiphertext64> {
        let lwe_size = extracted_bits_blocks[0].lwe_dimension().to_lwe_size();

        let mut all_datas = vec![];
        for lwe_vec in extracted_bits_blocks.into_iter() {
            let data = self
                .engine
                .consume_retrieve_lwe_ciphertext_vector(lwe_vec)
                .unwrap();

            all_datas.extend_from_slice(data.as_slice());
        }

        let flatenned_extracted_bits_view = self
            .engine
            .create_lwe_ciphertext_vector_from(all_datas.as_slice(), lwe_size)
            .unwrap();

        let flattened_lut: Vec<u64> = vec_lut.iter().flatten().copied().collect();
        let plaintext_lut = self
            .engine
            .create_plaintext_vector_from(&flattened_lut)
            .unwrap();
        let output = self
            .circuit_bootstrap_with_bits(
                wopbs_key,
                server_key,
                &flatenned_extracted_bits_view,
                &plaintext_lut,
                LweCiphertextCount(vec_lut.len()),
            )
            .unwrap();

        assert_eq!(output.lwe_ciphertext_count().0, vec_lut.len());
        let output_container = self
            .engine
            .consume_retrieve_lwe_ciphertext_vector(output)
            .unwrap();
        let lwes: Result<Vec<_>, _> = output_container
            .chunks_exact(output_container.len() / vec_lut.len())
            .map(|s| self.engine.create_lwe_ciphertext_from(s.to_vec()))
            .collect();

        let lwes = lwes.unwrap();

        assert_eq!(lwes.len(), vec_lut.len());
        lwes
    }
}
