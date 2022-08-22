//! Module with the definition of the treePBS.
//!
//! This module implements the generation of another server public key, which allows to compute
//! an alternative version of the programmable bootstrapping. Initially defined for integers,
//! this method could be useful to get faster shortint operations.
//!
//! # WARNING: this module is experimental.

#[cfg(test)]
mod tests;

use crate::engine::ShortintEngine;
use crate::{Ciphertext, ClientKey, ServerKey};
use concrete_core::prelude::PackingKeyswitchKey64;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct TreepbsKey {
    pub pksk: PackingKeyswitchKey64,
}

impl TreepbsKey {
    pub fn new_tree_key(cks: &ClientKey) -> TreepbsKey {
        ShortintEngine::with_thread_local_mut(|engine| engine.new_treepbs_key(cks).unwrap())
    }

    pub fn mul_lsb_treepbs(
        &mut self,
        sks: &ServerKey,
        ct_left: &Ciphertext,
        ct_right: &Ciphertext,
    ) -> Ciphertext {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine
                .mul_lsb_treepbs(self, sks, ct_left, ct_right)
                .unwrap()
        })
    }

    pub fn bivaluepbs<F1, F2>(
        &mut self,
        sks: &ServerKey,
        ct_in: &Ciphertext,
        f_1: F1,
        f_2: F2,
    ) -> Vec<Ciphertext>
    where
        F1: Fn(u64) -> u64,
        F2: Fn(u64) -> u64,
    {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine.bivaluepbs(sks, ct_in, f_1, f_2).unwrap()
        })
    }

    pub fn mul_treepbs_with_multivalue(
        &mut self,
        sks: &ServerKey,
        ct_left: &Ciphertext,
        ct_right: &Ciphertext,
    ) -> Ciphertext {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine
                .mul_treepbs_with_multivalue(self, sks, ct_left, ct_right)
                .unwrap()
        })
    }

    pub fn message_and_carry_extract(
        &mut self,
        sks: &ServerKey,
        ct_in: &Ciphertext,
    ) -> Vec<Ciphertext> {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine.message_and_carry_extract(sks, ct_in).unwrap()
        })
    }
}
