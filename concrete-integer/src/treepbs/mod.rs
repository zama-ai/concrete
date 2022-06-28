use crate::{Ciphertext, ClientKey, ServerKey};
use concrete_core::backends::core::private::crypto::bootstrap::multivaluepbs::{
    generate_fourier_polynomial_three_variables, generate_fourier_polynomial_three_variables_base,
    generate_fourier_polynomial_two_variables, generate_fourier_polynomial_two_variables_base,
};
use concrete_core::backends::core::private::crypto::bootstrap::FourierBuffers;
use concrete_core::backends::core::private::crypto::lwe::LweCiphertext;
use concrete_core::backends::core::private::math::polynomial::Polynomial;
use concrete_core::prelude::{LweBootstrapKeyEntity, LweCiphertext64};
use concrete_shortint::ciphertext::Degree;

pub struct TreepbsKey(pub(crate) concrete_shortint::treepbs::TreepbsKey);

impl TreepbsKey {
    pub fn new(cks: &ClientKey) -> TreepbsKey {
        TreepbsKey(concrete_shortint::treepbs::TreepbsKey::new_tree_key(
            &cks.key,
        ))
    }

    pub fn two_block_pbs<F>(&self, sks: &ServerKey, ct: &Ciphertext, f: F) -> Ciphertext
    where
        F: Fn(u64) -> u64,
    {
        //Create the buffers
        //=======================================================================
        let mut lwe_out_1 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut lwe_out_2 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut buffers = FourierBuffers::new(
            sks.key.bootstrapping_key.polynomial_size(),
            sks.key.bootstrapping_key.glwe_dimension().to_glwe_size(),
        );
        //=======================================================================

        //Keyswitch the ciphertexts
        //=======================================================================
        let vec_lwe_in: Vec<LweCiphertext<Vec<u64>>> =
            ct.ct_vec.iter().map(|ct| ct.ct.0.clone()).collect();
        let empty_selector =
            LweCiphertext::allocate(0_u64, sks.key.bootstrapping_key.0.key_size().to_lwe_size());
        let mut selectors = vec![empty_selector; vec_lwe_in.len()];
        sks.key
            .key_switching_key
            .0
            .vector_keyswitch(&mut selectors, vec_lwe_in.as_slice());
        //=======================================================================

        let modulus = sks.key.message_modulus.0 * sks.key.carry_modulus.0;

        let base = sks.key.message_modulus.0 as u64;
        let polynomial_size = sks.key.bootstrapping_key.polynomial_size();

        let g_1 = |x: u64, y: u64| f(x + base * y) % base;
        let g_2 = |x: u64, y: u64| (f(x + base * y) / base) % base;

        let mut poly_acc_1 = Vec::with_capacity(modulus);
        generate_fourier_polynomial_two_variables(g_1, modulus, polynomial_size, &mut poly_acc_1);

        let mut poly_acc_2 = Vec::with_capacity(modulus);
        generate_fourier_polynomial_two_variables(g_2, modulus, polynomial_size, &mut poly_acc_2);

        //Create the polynomial to multiply the accumulator with
        //=======================================================================
        let mut poly_block_redundancy = vec![0_u64; sks.key.bootstrapping_key.polynomial_size().0];

        // N/(p/2) = size of each block
        let box_size = sks.key.bootstrapping_key.polynomial_size().0 / modulus;

        let block_size = box_size * modulus;

        for block in poly_block_redundancy.chunks_exact_mut(block_size) {
            block[..box_size].fill(1);
        }

        // println!("poly_redundancy = {:?}", poly_block_redundancy);
        let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
        //=======================================================================

        sks.key.bootstrapping_key.0.treepbs_with_multivalue(
            &self.0.pksk.0,
            &mut lwe_out_1,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers,
            modulus as u64,
            0,
            &poly_redundancy,
            // &rlwe_sk,
            // &lwe_sk,
            &poly_acc_1,
        );

        sks.key.bootstrapping_key.0.treepbs_with_multivalue(
            &self.0.pksk.0,
            &mut lwe_out_2,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers.clone(),
            modulus as u64,
            0,
            &poly_redundancy,
            // &rlwe_sk,
            // &lwe_sk,
            &poly_acc_2,
        );

        let c_1 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_1),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        let c_2 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_2),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        Ciphertext {
            ct_vec: vec![c_1, c_2],
            message_modulus_vec: ct.message_modulus_vec.clone(),
            key_id_vec: ct.key_id_vec.clone(),
        }
    }

    pub fn two_block_pbs_base<F>(&self, sks: &ServerKey, ct: &Ciphertext, f: F) -> Ciphertext
    where
        F: Fn(u64) -> u64,
    {
        //Create the buffers
        //=======================================================================
        let mut lwe_out_1 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut lwe_out_2 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut buffers = FourierBuffers::new(
            sks.key.bootstrapping_key.polynomial_size(),
            sks.key.bootstrapping_key.glwe_dimension().to_glwe_size(),
        );
        //=======================================================================

        //Keyswitch the ciphertexts
        //=======================================================================
        let vec_lwe_in: Vec<LweCiphertext<Vec<u64>>> =
            ct.ct_vec.iter().map(|ct| ct.ct.0.clone()).collect();
        let empty_selector =
            LweCiphertext::allocate(0_u64, sks.key.bootstrapping_key.0.key_size().to_lwe_size());
        let mut selectors = vec![empty_selector; vec_lwe_in.len()];
        sks.key
            .key_switching_key
            .0
            .vector_keyswitch(&mut selectors, vec_lwe_in.as_slice());
        //=======================================================================

        let modulus = sks.key.message_modulus.0 * sks.key.carry_modulus.0;

        let base = sks.key.message_modulus.0 as u64;
        let polynomial_size = sks.key.bootstrapping_key.polynomial_size();

        let g_1 = |x: u64, y: u64| f(x + base * y) % base;
        let g_2 = |x: u64, y: u64| (f(x + base * y) / base) % base;

        let mut poly_acc_1 = Vec::with_capacity(base as usize);
        generate_fourier_polynomial_two_variables_base(
            g_1,
            modulus,
            base as usize,
            polynomial_size,
            &mut poly_acc_1,
        );

        let mut poly_acc_2 = Vec::with_capacity(base as usize);
        generate_fourier_polynomial_two_variables_base(
            g_2,
            modulus,
            base as usize,
            polynomial_size,
            &mut poly_acc_2,
        );

        //Create the polynomial to multiply the accumulator with
        //=======================================================================
        let mut poly_block_redundancy = vec![0_u64; sks.key.bootstrapping_key.polynomial_size().0];

        // N/(p/2) = size of each block
        let box_size = sks.key.bootstrapping_key.polynomial_size().0 / modulus;

        let block_size = box_size * base as usize;

        for block in poly_block_redundancy.chunks_exact_mut(block_size) {
            block[..box_size].fill(1);
        }

        // println!("poly_redundancy = {:?}", poly_block_redundancy);
        let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
        //=======================================================================

        sks.key.bootstrapping_key.0.treepbs_with_multivalue_base(
            &self.0.pksk.0,
            &mut lwe_out_1,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers,
            modulus as u64,
            base,
            0,
            &poly_redundancy,
            // &rlwe_sk,
            // &lwe_sk,
            &poly_acc_1,
        );

        sks.key.bootstrapping_key.0.treepbs_with_multivalue_base(
            &self.0.pksk.0,
            &mut lwe_out_2,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers.clone(),
            modulus as u64,
            base,
            0,
            &poly_redundancy,
            // &rlwe_sk,
            // &lwe_sk,
            &poly_acc_2,
        );

        let c_1 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_1),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        let c_2 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_2),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        Ciphertext {
            ct_vec: vec![c_1, c_2],
            message_modulus_vec: ct.message_modulus_vec.clone(),
            key_id_vec: ct.key_id_vec.clone(),
        }
    }

    pub fn three_block_pbs<F>(&self, sks: &ServerKey, ct: &Ciphertext, f: F) -> Ciphertext
    where
        F: Fn(u64) -> u64,
    {
        //Create the buffers
        //=======================================================================
        let mut lwe_out_1 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut lwe_out_2 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut lwe_out_3 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut buffers = FourierBuffers::new(
            sks.key.bootstrapping_key.polynomial_size(),
            sks.key.bootstrapping_key.glwe_dimension().to_glwe_size(),
        );
        //=======================================================================

        //Keyswitch the ciphertexts
        //=======================================================================
        let vec_lwe_in: Vec<LweCiphertext<Vec<u64>>> =
            ct.ct_vec.iter().map(|ct| ct.ct.0.clone()).collect();
        let empty_selector =
            LweCiphertext::allocate(0_u64, sks.key.bootstrapping_key.0.key_size().to_lwe_size());
        let mut selectors = vec![empty_selector; vec_lwe_in.len()];
        sks.key
            .key_switching_key
            .0
            .vector_keyswitch(&mut selectors, vec_lwe_in.as_slice());
        //=======================================================================

        let modulus = sks.key.message_modulus.0 * sks.key.carry_modulus.0;
        let modulus_2 = modulus * modulus;

        let base = sks.key.message_modulus.0 as u64;
        let base_2 = base * base;
        let polynomial_size = sks.key.bootstrapping_key.polynomial_size();

        let g_1 = |x: u64, y: u64, z: u64| f(x + base * y + base_2 * z) % base;
        let g_2 = |x: u64, y: u64, z: u64| (f(x + base * y + base_2 * z) / base) % base;
        let g_3 = |x: u64, y: u64, z: u64| (f(x + base * y + base_2 * z) / base_2) % base;

        let mut poly_acc_1 = Vec::with_capacity(modulus_2);
        generate_fourier_polynomial_three_variables(g_1, modulus, polynomial_size, &mut poly_acc_1);

        let mut poly_acc_2 = Vec::with_capacity(modulus_2);
        generate_fourier_polynomial_three_variables(g_2, modulus, polynomial_size, &mut poly_acc_2);

        let mut poly_acc_3 = Vec::with_capacity(modulus_2);
        generate_fourier_polynomial_three_variables(g_3, modulus, polynomial_size, &mut poly_acc_3);

        //Create the polynomial to multiply the accumulator with
        //=======================================================================
        let mut poly_block_redundancy = vec![0_u64; sks.key.bootstrapping_key.polynomial_size().0];

        // N/(p/2) = size of each block
        let box_size = sks.key.bootstrapping_key.polynomial_size().0 / modulus;

        let block_size = box_size * modulus;

        for block in poly_block_redundancy.chunks_exact_mut(block_size) {
            block[..box_size].fill(1);
        }

        let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
        //=======================================================================

        sks.key.bootstrapping_key.0.treepbs_with_multivalue(
            &self.0.pksk.0,
            &mut lwe_out_1,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers,
            modulus as u64,
            0,
            &poly_redundancy,
            &poly_acc_1,
        );

        sks.key.bootstrapping_key.0.treepbs_with_multivalue(
            &self.0.pksk.0,
            &mut lwe_out_2,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers,
            modulus as u64,
            0,
            &poly_redundancy,
            &poly_acc_2,
        );

        sks.key.bootstrapping_key.0.treepbs_with_multivalue(
            &self.0.pksk.0,
            &mut lwe_out_3,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers,
            modulus as u64,
            0,
            &poly_redundancy,
            &poly_acc_3,
        );

        let c_1 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_1),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        let c_2 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_2),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        let c_3 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_3),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        Ciphertext {
            ct_vec: vec![c_1, c_2, c_3],
            message_modulus_vec: ct.message_modulus_vec.clone(),
            key_id_vec: ct.key_id_vec.clone(),
        }
    }

    pub fn three_block_pbs_base<F>(&self, sks: &ServerKey, ct: &Ciphertext, f: F) -> Ciphertext
    where
        F: Fn(u64) -> u64,
    {
        //Create the buffers
        //=======================================================================
        let mut lwe_out_1 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut lwe_out_2 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut lwe_out_3 = LweCiphertext::allocate(
            0_u64,
            sks.key
                .bootstrapping_key
                .output_lwe_dimension()
                .to_lwe_size(),
        );

        let mut buffers = FourierBuffers::new(
            sks.key.bootstrapping_key.polynomial_size(),
            sks.key.bootstrapping_key.glwe_dimension().to_glwe_size(),
        );
        //=======================================================================

        //Keyswitch the ciphertexts
        //=======================================================================
        let vec_lwe_in: Vec<LweCiphertext<Vec<u64>>> =
            ct.ct_vec.iter().map(|ct| ct.ct.0.clone()).collect();
        let empty_selector =
            LweCiphertext::allocate(0_u64, sks.key.bootstrapping_key.0.key_size().to_lwe_size());
        let mut selectors = vec![empty_selector; vec_lwe_in.len()];
        sks.key
            .key_switching_key
            .0
            .vector_keyswitch(&mut selectors, vec_lwe_in.as_slice());
        //=======================================================================

        let modulus = sks.key.message_modulus.0 * sks.key.carry_modulus.0;

        let base = sks.key.message_modulus.0 as u64;
        let base_2 = base * base;
        let polynomial_size = sks.key.bootstrapping_key.polynomial_size();

        let g_1 = |x: u64, y: u64, z: u64| f(x + base * y + base_2 * z) % base;
        let g_2 = |x: u64, y: u64, z: u64| (f(x + base * y + base_2 * z) / base) % base;
        let g_3 = |x: u64, y: u64, z: u64| (f(x + base * y + base_2 * z) / base_2) % base;

        let mut poly_acc_1 = Vec::with_capacity(base_2 as usize);
        generate_fourier_polynomial_three_variables_base(
            g_1,
            modulus,
            base as usize,
            polynomial_size,
            &mut poly_acc_1,
        );

        let mut poly_acc_2 = Vec::with_capacity(base_2 as usize);
        generate_fourier_polynomial_three_variables_base(
            g_2,
            modulus,
            base as usize,
            polynomial_size,
            &mut poly_acc_2,
        );

        let mut poly_acc_3 = Vec::with_capacity(base_2 as usize);
        generate_fourier_polynomial_three_variables_base(
            g_3,
            modulus,
            base as usize,
            polynomial_size,
            &mut poly_acc_3,
        );

        //Create the polynomial to multiply the accumulator with
        //=======================================================================
        let mut poly_block_redundancy = vec![0_u64; sks.key.bootstrapping_key.polynomial_size().0];

        // N/(p/2) = size of each block
        let box_size = sks.key.bootstrapping_key.polynomial_size().0 / modulus;

        let block_size = box_size * base as usize;

        for block in poly_block_redundancy.chunks_exact_mut(block_size) {
            block[..box_size].fill(1);
        }

        let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
        //=======================================================================

        sks.key.bootstrapping_key.0.treepbs_with_multivalue_base(
            &self.0.pksk.0,
            &mut lwe_out_1,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers,
            modulus as u64,
            base,
            0,
            &poly_redundancy,
            &poly_acc_1,
        );

        sks.key.bootstrapping_key.0.treepbs_with_multivalue_base(
            &self.0.pksk.0,
            &mut lwe_out_2,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers,
            modulus as u64,
            base,
            0,
            &poly_redundancy,
            &poly_acc_2,
        );

        sks.key.bootstrapping_key.0.treepbs_with_multivalue_base(
            &self.0.pksk.0,
            &mut lwe_out_3,
            selectors.as_slice(),
            &sks.key.key_switching_key.0,
            &mut buffers,
            modulus as u64,
            base,
            0,
            &poly_redundancy,
            &poly_acc_3,
        );

        let c_1 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_1),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        let c_2 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_2),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        let c_3 = concrete_shortint::Ciphertext {
            ct: LweCiphertext64(lwe_out_3),
            degree: Degree(sks.key.message_modulus.0 - 1),
            message_modulus: sks.key.message_modulus,
            carry_modulus: sks.key.carry_modulus,
        };

        Ciphertext {
            ct_vec: vec![c_1, c_2, c_3],
            message_modulus_vec: ct.message_modulus_vec.clone(),
            key_id_vec: ct.key_id_vec.clone(),
        }
    }
}
