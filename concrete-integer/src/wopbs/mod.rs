//! Module with the definition of the WopbsKey (WithOut padding PBS Key).
//!
//! This module implements the generation of another server public key, which allows to compute
//! an alternative version of the programmable bootstrapping. This does not require the use of a
//! bit of padding.
//!
//! # WARNING: this module is experimental.

#[cfg(test)]
mod test;

use crate::{Ciphertext, ClientKey, ServerKey};
use concrete_core::backends::core::private::crypto::circuit_bootstrap::DeltaLog;
use concrete_core::backends::core::private::crypto::lwe::LweCiphertext;
use concrete_core::prelude::LweCiphertext64;
use concrete_shortint::ciphertext::Degree;

pub struct WopbsKeyV0 {
    wopbs_key: concrete_shortint::wopbs::WopbsKey,
}

impl WopbsKeyV0 {
    pub fn new_wopbs_key(cks: &ClientKey, sks: &ServerKey) -> WopbsKeyV0 {
        WopbsKeyV0 {
            wopbs_key: concrete_shortint::wopbs::WopbsKey::new_wopbs_key(&cks.key, &sks.key),
        }
    }

    pub fn new_from_shortint(wopbskey: &concrete_shortint::wopbs::WopbsKey) -> WopbsKeyV0 {
        let key = wopbskey.clone();
        WopbsKeyV0 { wopbs_key: key }
    }

    pub fn circuit_bootstrap_vertical_packing_v0(
        &self,
        sks: &ServerKey,
        ct_in: &mut Ciphertext,
        lut: &[Vec<u64>],
    ) -> Ciphertext {
        let mut vec_lwe: Vec<LweCiphertext<Vec<u64>>> = vec![];

        // Extraction of each bit for each block
        for block in ct_in.ct_vec.iter_mut() {
            let delta = (1_usize << 63) / (block.message_modulus.0 * block.carry_modulus.0);
            let delta_log = DeltaLog(f64::log2(delta as f64) as usize);
            let nb_bit_to_extract =
                f64::log2((block.message_modulus.0 * block.carry_modulus.0) as f64) as usize;

            let mut tmp =
                self.wopbs_key
                    .extract_bit(delta_log, &mut block.ct.0, &sks.key, nb_bit_to_extract);
            vec_lwe.append(&mut tmp);
        }

        vec_lwe.reverse();

        let vec_ct_out = self.wopbs_key.vertical_packing_cbs_binary_v0(
            &sks.key,
            lut.to_vec(),
            vec_lwe.as_slice(),
        );

        let mut ct_vec_out: Vec<concrete_shortint::Ciphertext> = vec![];
        for (block, block_out) in ct_in.ct_vec.iter().zip(vec_ct_out.into_iter()) {
            ct_vec_out.push(concrete_shortint::Ciphertext {
                ct: LweCiphertext64(block_out),
                degree: Degree(block.message_modulus.0 - 1),
                message_modulus: block.message_modulus,
                carry_modulus: block.carry_modulus,
            });
        }

        Ciphertext {
            ct_vec: ct_vec_out,
            message_modulus_vec: ct_in.message_modulus_vec.clone(),
            key_id_vec: ct_in.key_id_vec.clone(),
        }
    }
}
