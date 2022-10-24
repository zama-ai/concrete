//! All the `ShortintEngine` method related to client side (encrypt / decrypt)
use super::{EngineResult, ShortintEngine};
use crate::ciphertext::Degree;
use crate::parameters::{CarryModulus, MessageModulus};
use crate::{Ciphertext, ClientKey, PublicKey};
use concrete_core::prelude::*;

impl ShortintEngine {
    pub(crate) fn new_public_key(
        &mut self,
        client_key: &ClientKey,
        public_key_lwe_zero_encryption_count: LwePublicKeyZeroEncryptionCount,
    ) -> EngineResult<PublicKey> {
        Ok(PublicKey {
            lwe_public_key: self.engine.generate_new_lwe_public_key(
                &client_key.lwe_secret_key,
                Variance(client_key.parameters.lwe_modular_std_dev.get_variance()),
                public_key_lwe_zero_encryption_count,
            )?,
            parameters: client_key.parameters.to_owned(),
        })
    }

    pub(crate) fn encrypt_with_public_key(
        &mut self,
        public_key: &PublicKey,
        message: u64,
    ) -> EngineResult<Ciphertext> {
        self.encrypt_with_message_modulus_and_public_key(
            public_key,
            message,
            public_key.parameters.message_modulus,
        )
    }

    pub(crate) fn encrypt_with_message_modulus_and_public_key(
        &mut self,
        public_key: &PublicKey,
        message: u64,
        message_modulus: MessageModulus,
    ) -> EngineResult<Ciphertext> {
        //This ensures that the space message_modulus*carry_modulus < param.message_modulus *
        // param.carry_modulus
        let carry_modulus = (public_key.parameters.message_modulus.0
            * public_key.parameters.carry_modulus.0)
            / message_modulus.0;

        //The delta is the one defined by the parameters
        let delta = (1_u64 << 63)
            / (public_key.parameters.message_modulus.0 * public_key.parameters.carry_modulus.0)
                as u64;

        //The input is reduced modulus the message_modulus
        let m = message % message_modulus.0 as u64;

        let shifted_message = m * delta;
        // encode the message
        let plain: Plaintext64 = self.engine.create_plaintext_from(&shifted_message)?;

        // This allocates the required ct
        let mut encrypted_ct = self.engine.trivially_encrypt_lwe_ciphertext(
            public_key.lwe_public_key.lwe_dimension().to_lwe_size(),
            &plain,
        )?;

        // encryption
        self.engine.discard_encrypt_lwe_ciphertext_with_public_key(
            &public_key.lwe_public_key,
            &mut encrypted_ct,
            &plain,
        )?;

        Ok(Ciphertext {
            ct: encrypted_ct,
            degree: Degree(message_modulus.0 - 1),
            message_modulus,
            carry_modulus: CarryModulus(carry_modulus),
        })
    }

    pub(crate) fn encrypt_without_padding_with_public_key(
        &mut self,
        public_key: &PublicKey,
        message: u64,
    ) -> EngineResult<Ciphertext> {
        //Multiply by 2 to reshift and exclude the padding bit
        let delta = ((1_u64 << 63)
            / (public_key.parameters.message_modulus.0 * public_key.parameters.carry_modulus.0)
                as u64)
            * 2;

        let shifted_message = message * delta;
        // encode the message
        let plain: Plaintext64 = self.engine.create_plaintext_from(&shifted_message)?;

        // This allocates the required ct
        let mut encrypted_ct = self.engine.trivially_encrypt_lwe_ciphertext(
            public_key.lwe_public_key.lwe_dimension().to_lwe_size(),
            &plain,
        )?;

        // encryption
        self.engine.discard_encrypt_lwe_ciphertext_with_public_key(
            &public_key.lwe_public_key,
            &mut encrypted_ct,
            &plain,
        )?;

        Ok(Ciphertext {
            ct: encrypted_ct,
            degree: Degree(public_key.parameters.message_modulus.0 - 1),
            message_modulus: public_key.parameters.message_modulus,
            carry_modulus: public_key.parameters.carry_modulus,
        })
    }

    pub(crate) fn encrypt_native_crt_with_public_key(
        &mut self,
        public_key: &PublicKey,
        message: u64,
        message_modulus: u8,
    ) -> EngineResult<Ciphertext> {
        let carry_modulus = 1;
        let m = (message % message_modulus as u64) as u128;
        let shifted_message = m * (1 << 64) / message_modulus as u128;
        // encode the message
        let plain: Plaintext64 = self
            .engine
            .create_plaintext_from(&(shifted_message as u64))?;

        // This allocates the required ct
        let mut encrypted_ct = self.engine.trivially_encrypt_lwe_ciphertext(
            public_key.lwe_public_key.lwe_dimension().to_lwe_size(),
            &plain,
        )?;

        // encryption
        self.engine.discard_encrypt_lwe_ciphertext_with_public_key(
            &public_key.lwe_public_key,
            &mut encrypted_ct,
            &plain,
        )?;

        Ok(Ciphertext {
            ct: encrypted_ct,
            degree: Degree(message_modulus as usize - 1),
            message_modulus: MessageModulus(message_modulus as usize),
            carry_modulus: CarryModulus(carry_modulus),
        })
    }

    pub(crate) fn unchecked_encrypt_with_public_key(
        &mut self,
        public_key: &PublicKey,
        message: u64,
    ) -> EngineResult<Ciphertext> {
        let delta = (1_u64 << 63)
            / (public_key.parameters.message_modulus.0 * public_key.parameters.carry_modulus.0)
                as u64;
        let shifted_message = message * delta;
        // encode the message
        let plain: Plaintext64 = self.engine.create_plaintext_from(&shifted_message)?;

        // This allocates the required ct
        let mut encrypted_ct = self.engine.trivially_encrypt_lwe_ciphertext(
            public_key.lwe_public_key.lwe_dimension().to_lwe_size(),
            &plain,
        )?;

        // encryption
        self.engine.discard_encrypt_lwe_ciphertext_with_public_key(
            &public_key.lwe_public_key,
            &mut encrypted_ct,
            &plain,
        )?;

        Ok(Ciphertext {
            ct: encrypted_ct,
            degree: Degree(
                public_key.parameters.message_modulus.0 * public_key.parameters.carry_modulus.0 - 1,
            ),
            message_modulus: public_key.parameters.message_modulus,
            carry_modulus: public_key.parameters.carry_modulus,
        })
    }
}
