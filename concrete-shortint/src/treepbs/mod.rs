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
use concrete_core::prelude::{
    AbstractEngine, DefaultSerializationEngine, EntityDeserializationEngine,
    EntitySerializationEngine, LwePackingKeyswitchKey64,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone)]
pub struct TreepbsKey {
    pub pksk: LwePackingKeyswitchKey64,
}

impl TreepbsKey {
    pub fn new_tree_key(cks: &ClientKey) -> TreepbsKey {
        ShortintEngine::with_thread_local_mut(|engine| engine.new_treepbs_key(cks).unwrap())
    }

    pub fn mul_lsb_treepbs(
        &self,
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
        &self,
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
        &self,
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
        &self,
        sks: &ServerKey,
        ct_in: &Ciphertext,
    ) -> Vec<Ciphertext> {
        ShortintEngine::with_thread_local_mut(|engine| {
            engine.message_and_carry_extract(sks, ct_in).unwrap()
        })
    }
}

#[derive(Serialize, Deserialize)]
struct SerializableTreePbsKey {
    pksk: Vec<u8>,
}

impl Serialize for TreepbsKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut default_ser_eng =
            DefaultSerializationEngine::new(()).map_err(serde::ser::Error::custom)?;

        let pksk = default_ser_eng
            .serialize(&self.pksk)
            .map_err(serde::ser::Error::custom)?;

        SerializableTreePbsKey { pksk }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TreepbsKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let thing =
            SerializableTreePbsKey::deserialize(deserializer).map_err(serde::de::Error::custom)?;
        let mut default_ser_eng =
            DefaultSerializationEngine::new(()).map_err(serde::de::Error::custom)?;

        let pksk = default_ser_eng
            .deserialize(thing.pksk.as_slice())
            .map_err(serde::de::Error::custom)?;

        Ok(Self { pksk })
    }
}
