//! The public key for homomorphic computation.
//!
//! This module implements the generation of the server's public key, together with all the
//! available homomorphic Boolean gates ($\mathrm{AND}$, $\mathrm{MUX}$, $\mathrm{NAND}$,
//! $\mathrm{NOR}$,
//! $\mathrm{NOT}$, $\mathrm{OR}$, $\mathrm{XNOR}$, $\mathrm{XOR}$).

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

use crate::ciphertext::Ciphertext;
use crate::client_key::ClientKey;
use crate::engine::bootstrapping::CpuServerKey;
use crate::engine::{with_thread_local_cpu_engine_mut, BinaryGatesEngine};

pub trait BinaryBooleanGates<L, R> {
    fn and(&self, ct_left: L, ct_right: R) -> Ciphertext;
    fn nand(&self, ct_left: L, ct_right: R) -> Ciphertext;
    fn nor(&self, ct_left: L, ct_right: R) -> Ciphertext;
    fn or(&self, ct_left: L, ct_right: R) -> Ciphertext;
    fn xor(&self, ct_left: L, ct_right: R) -> Ciphertext;
    fn xnor(&self, ct_left: L, ct_right: R) -> Ciphertext;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ServerKey {
    cpu_key: CpuServerKey,
}

impl ServerKey {
    pub fn new(cks: &ClientKey) -> Self {
        let cpu_key = with_thread_local_cpu_engine_mut(|engine| engine.create_server_key(cks));

        Self { cpu_key }
    }

    pub fn trivial_encrypt(&self, message: bool) -> Ciphertext {
        Ciphertext::Trivial(message)
    }

    pub fn not(&self, ct: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.not(ct))
    }

    pub fn mux(
        &self,
        ct_condition: &Ciphertext,
        ct_then: &Ciphertext,
        ct_else: &Ciphertext,
    ) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| {
            engine.mux(ct_condition, ct_then, ct_else, &self.cpu_key)
        })
    }
}

impl BinaryBooleanGates<&Ciphertext, &Ciphertext> for ServerKey {
    fn and(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.and(ct_left, ct_right, &self.cpu_key))
    }

    fn nand(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.nand(ct_left, ct_right, &self.cpu_key))
    }

    fn nor(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.nor(ct_left, ct_right, &self.cpu_key))
    }

    fn or(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.or(ct_left, ct_right, &self.cpu_key))
    }

    fn xor(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.xor(ct_left, ct_right, &self.cpu_key))
    }

    fn xnor(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.xnor(ct_left, ct_right, &self.cpu_key))
    }
}

impl BinaryBooleanGates<&Ciphertext, bool> for ServerKey {
    fn and(&self, ct_left: &Ciphertext, ct_right: bool) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.and(ct_left, ct_right, &self.cpu_key))
    }

    fn nand(&self, ct_left: &Ciphertext, ct_right: bool) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.nand(ct_left, ct_right, &self.cpu_key))
    }

    fn nor(&self, ct_left: &Ciphertext, ct_right: bool) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.nor(ct_left, ct_right, &self.cpu_key))
    }

    fn or(&self, ct_left: &Ciphertext, ct_right: bool) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.or(ct_left, ct_right, &self.cpu_key))
    }

    fn xor(&self, ct_left: &Ciphertext, ct_right: bool) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.xor(ct_left, ct_right, &self.cpu_key))
    }

    fn xnor(&self, ct_left: &Ciphertext, ct_right: bool) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.xnor(ct_left, ct_right, &self.cpu_key))
    }
}

impl BinaryBooleanGates<bool, &Ciphertext> for ServerKey {
    fn and(&self, ct_left: bool, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.and(ct_left, ct_right, &self.cpu_key))
    }

    fn nand(&self, ct_left: bool, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.nand(ct_left, ct_right, &self.cpu_key))
    }

    fn nor(&self, ct_left: bool, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.nor(ct_left, ct_right, &self.cpu_key))
    }

    fn or(&self, ct_left: bool, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.or(ct_left, ct_right, &self.cpu_key))
    }

    fn xor(&self, ct_left: bool, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.xor(ct_left, ct_right, &self.cpu_key))
    }

    fn xnor(&self, ct_left: bool, ct_right: &Ciphertext) -> Ciphertext {
        with_thread_local_cpu_engine_mut(|engine| engine.xnor(ct_left, ct_right, &self.cpu_key))
    }
}
