use crate::ciphertext::Degree;
use crate::engine::{EngineResult, ShortintEngine};
use crate::treepbs::TreepbsKey;
use crate::{Ciphertext, ClientKey, ServerKey};
use concrete_core::backends::fftw::private::crypto::bootstrap::multivaluepbs::generate_fourier_polynomial_multivalue;
use concrete_core::backends::fftw::private::crypto::bootstrap::FourierBuffers;
use concrete_core::commons::crypto::lwe::LweCiphertext;
use concrete_core::commons::math::polynomial::Polynomial;
use concrete_core::prelude::{
    DispersionParameter, LweBootstrapKeyEntity, LweCiphertext64,
    LwePackingKeyswitchKeyGenerationEngine, Variance,
};

impl ShortintEngine {
    pub(crate) fn new_treepbs_key(&mut self, cks: &ClientKey) -> EngineResult<TreepbsKey> {
        let decomp_log_base = cks.parameters.ks_base_log;
        let decomp_level_count = cks.parameters.ks_level;
        let noise = Variance(cks.parameters.glwe_modular_std_dev.get_variance());

        let pksk = self.engine.generate_new_lwe_packing_keyswitch_key(
            &cks.lwe_secret_key,
            &cks.glwe_secret_key,
            decomp_level_count,
            decomp_log_base,
            noise,
        )?;
        let treepbs_key = TreepbsKey { pksk };
        Ok(treepbs_key)
    }

    pub fn mul_lsb_treepbs(
        &mut self,
        treepbskey: &TreepbsKey,
        sks: &ServerKey,
        ct_left: &Ciphertext,
        ct_right: &Ciphertext,
    ) -> EngineResult<Ciphertext> {
        //Create the buffers
        //=======================================================================
        let mut lwe_out = LweCiphertext::allocate(
            0_u64,
            sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        let mut buffers = FourierBuffers::new(
            sks.bootstrapping_key.polynomial_size(),
            sks.bootstrapping_key.glwe_dimension().to_glwe_size(),
        );

        let modulus = sks.carry_modulus.0 * sks.message_modulus.0;

        let lwe_buffer = LweCiphertext::allocate(
            0_u64,
            sks.bootstrapping_key.0.output_lwe_dimension().to_lwe_size(),
        );
        let mut lwe_buffer_bootstrap = vec![lwe_buffer; modulus];
        //=======================================================================

        //Keyswitch the ciphertexts
        //=======================================================================
        let vec_lwe_in = vec![ct_left.ct.clone().0, ct_right.ct.clone().0];
        let empty_selector =
            LweCiphertext::allocate(0_u64, sks.bootstrapping_key.0.key_size().to_lwe_size());
        let mut selectors = vec![empty_selector; vec_lwe_in.len()];
        sks.key_switching_key
            .0
            .vector_keyswitch(&mut selectors, &vec_lwe_in);
        //=======================================================================

        //Create the polynomial to multiply the accumulator with
        //=======================================================================
        let mut poly_block_redundancy = vec![0_u64; sks.bootstrapping_key.polynomial_size().0];

        // N/(p/2) = size of each block
        let box_size = sks.bootstrapping_key.polynomial_size().0 / modulus;

        let block_size = box_size * modulus;

        for block in poly_block_redundancy.chunks_exact_mut(block_size) {
            block[..box_size].fill(1);
        }

        let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
        //=======================================================================

        //Generate accumulators
        //=========================================================
        let mut acc = Vec::with_capacity(modulus);
        for i in 0..modulus {
            acc.push(
                self.generate_accumulator(sks, |x| (i as u64 * x) % modulus as u64)
                    .unwrap()
                    .0,
            );
        }
        //=========================================================

        sks.bootstrapping_key.0.treepbs(
            &treepbskey.pksk.0,
            &mut lwe_out,
            &selectors,
            &mut lwe_buffer_bootstrap,
            // &mut lwe_buffer_keyswitch,
            &sks.key_switching_key.0,
            acc.as_mut_slice(),
            &mut buffers,
            modulus,
            // self.message_modulus.0,
            0,
            &poly_redundancy,
        );

        Ok(Ciphertext {
            ct: LweCiphertext64(lwe_out),
            degree: Degree(sks.message_modulus.0 - 1),
            message_modulus: sks.message_modulus,
            carry_modulus: sks.carry_modulus,
        })
    }

    pub fn mul_treepbs_with_multivalue(
        &mut self,
        treepbskey: &TreepbsKey,
        sks: &ServerKey,
        ct_left: &Ciphertext,
        ct_right: &Ciphertext,
        // glwe_sk: &GlweSecretKey<BinaryKeyKind, Vec<u64>>,
        // lwe_sk: &LweSecretKey<BinaryKeyKind, Vec<u64>>,
    ) -> EngineResult<Ciphertext> {
        //Create the buffers
        //=======================================================================
        let mut lwe_out = LweCiphertext::allocate(
            0_u64,
            sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        let mut buffers = FourierBuffers::new(
            sks.bootstrapping_key.polynomial_size(),
            sks.bootstrapping_key.glwe_dimension().to_glwe_size(),
        );
        //=======================================================================

        //Keyswitch the ciphertexts
        //=======================================================================
        let vec_lwe_in = vec![ct_left.ct.clone().0, ct_right.ct.clone().0];
        let empty_selector =
            LweCiphertext::allocate(0_u64, sks.bootstrapping_key.0.key_size().to_lwe_size());
        let mut selectors = vec![empty_selector; vec_lwe_in.len()];
        sks.key_switching_key
            .0
            .vector_keyswitch(&mut selectors, &vec_lwe_in);
        //=======================================================================

        //Create the polynomial to multiply the accumulator with
        //=======================================================================
        let mut poly_block_redundancy = vec![0_u64; sks.bootstrapping_key.polynomial_size().0];

        let modulus = sks.message_modulus.0 * sks.carry_modulus.0;
        //let base = sks.message_modulus.0;

        // N/(p/2) = size of each block
        let box_size = sks.bootstrapping_key.polynomial_size().0 / modulus;

        let block_size = box_size * modulus;

        for block in poly_block_redundancy.chunks_exact_mut(block_size) {
            block[..box_size].fill(1);
        }

        let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
        //=======================================================================

        //Generate accumulators
        //=========================================================
        let mut poly_acc = Vec::with_capacity(modulus);
        for i in 0..modulus {
            poly_acc.push(generate_fourier_polynomial_multivalue(
                |x| (i as u64 * x) % modulus as u64,
                modulus,
                sks.bootstrapping_key.polynomial_size(),
            ));
        }
        // println!("poly = {:?}", poly_acc);
        //=========================================================

        sks.bootstrapping_key.0.treepbs_with_multivalue(
            &treepbskey.pksk.0,
            &mut lwe_out,
            &selectors,
            &sks.key_switching_key.0,
            &mut buffers,
            modulus as u64,
            // base as u64,
            0,
            &poly_redundancy,
            // &glwe_sk,
            // &lwe_sk,
            &poly_acc,
        );

        Ok(Ciphertext {
            ct: LweCiphertext64(lwe_out),
            degree: Degree(sks.message_modulus.0 - 1),
            message_modulus: sks.message_modulus,
            carry_modulus: sks.carry_modulus,
        })
    }

    pub fn bivaluepbs<F1, F2>(
        &mut self,
        sks: &ServerKey,
        ct_in: &Ciphertext,
        f_1: F1,
        f_2: F2,
    ) -> EngineResult<Vec<Ciphertext>>
    where
        F1: Fn(u64) -> u64,
        F2: Fn(u64) -> u64,
    {
        //Keyswitch the ciphertext
        let mut selector =
            LweCiphertext::allocate(0_u64, sks.bootstrapping_key.0.key_size().to_lwe_size());
        sks.key_switching_key
            .0
            .keyswitch_ciphertext(&mut selector, &ct_in.ct.0);

        let mut buffers = FourierBuffers::new(
            sks.bootstrapping_key.polynomial_size(),
            sks.bootstrapping_key.glwe_dimension().to_glwe_size(),
        );

        let modulus = (sks.message_modulus.0 * sks.carry_modulus.0) as u64;
        //Generate accumulators
        //=========================================================
        let poly_acc = vec![
            generate_fourier_polynomial_multivalue(
                f_1,
                modulus as usize,
                sks.bootstrapping_key.polynomial_size(),
            ),
            generate_fourier_polynomial_multivalue(
                f_2,
                modulus as usize,
                sks.bootstrapping_key.polynomial_size(),
            ),
        ];
        //=========================================================

        let vec_lwe = sks.bootstrapping_key.0.multivalue_programmable_bootstrap(
            &selector,
            modulus,
            &poly_acc,
            &mut buffers,
        );

        let c_1 = Ciphertext {
            ct: LweCiphertext64(vec_lwe[0].clone()),
            degree: Degree(sks.message_modulus.0 - 1),
            message_modulus: sks.message_modulus,
            carry_modulus: sks.carry_modulus,
        };

        let c_2 = Ciphertext {
            ct: LweCiphertext64(vec_lwe[1].clone()),
            degree: Degree(sks.message_modulus.0 - 1),
            message_modulus: sks.message_modulus,
            carry_modulus: sks.carry_modulus,
        };

        Ok(vec![c_1, c_2])
    }

    pub fn message_and_carry_extract(
        &mut self,
        sks: &ServerKey,
        ct_in: &Ciphertext,
    ) -> EngineResult<Vec<Ciphertext>> {
        let base = sks.message_modulus.0;
        self.bivaluepbs(sks, ct_in, |x| (x % base as u64), |x| (x / base as u64))
    }

    // pub fn two_block_pbs<F>(
    //     &mut self,
    //     treepbskey: &TreepbsKey,
    //     sks: &ServerKey,
    //     ct_1: &Ciphertext,
    //     ct_2: &Ciphertext,
    //     f: F,
    // ) -> EngineResult<Vec<Ciphertext>>
    // where
    //     F: Fn(u64) -> u64,
    // {
    //     //Create the buffers
    //     //=======================================================================
    //     let mut lwe_out_1 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut lwe_out_2 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut buffers = FourierBuffers::new(
    //         sks.bootstrapping_key.polynomial_size(),
    //         sks.bootstrapping_key.glwe_dimension().to_glwe_size(),
    //     );
    //     //=======================================================================
    //
    //     //Keyswitch the ciphertexts
    //     //=======================================================================
    //     let vec_lwe_in = vec![ct_1.ct.clone().0, ct_2.ct.clone().0];
    //     let empty_selector =
    //         LweCiphertext::allocate(0_u64, sks.bootstrapping_key.0.key_size().to_lwe_size());
    //     let mut selectors = vec![empty_selector; vec_lwe_in.len()];
    //     sks.key_switching_key
    //         .0
    //         .vector_keyswitch(&mut selectors, &vec_lwe_in);
    //     //=======================================================================
    //
    //     let modulus = sks.message_modulus.0 * sks.carry_modulus.0;
    //
    //     let base = sks.message_modulus.0 as u64;
    //     let polynomial_size = sks.bootstrapping_key.polynomial_size();
    //
    //     let g_1 = |x: u64, y: u64| f(x + base * y) % base;
    //     let g_2 = |x: u64, y: u64| (f(x + base * y) / base) % base;
    //
    //     let mut poly_acc_1 = Vec::with_capacity(modulus);
    //     generate_fourier_polynomial_two_variables(g_1, modulus, polynomial_size, &mut
    // poly_acc_1);
    //
    //     let mut poly_acc_2 = Vec::with_capacity(modulus);
    //     generate_fourier_polynomial_two_variables(g_2, modulus, polynomial_size, &mut
    // poly_acc_2);
    //
    //     //Create the polynomial to multiply the accumulator with
    //     //=======================================================================
    //     let mut poly_block_redundancy = vec![0_u64; sks.bootstrapping_key.polynomial_size().0];
    //
    //     // N/(p/2) = size of each block
    //     let box_size = sks.bootstrapping_key.polynomial_size().0 / modulus;
    //
    //     let block_size = box_size * modulus;
    //
    //     for block in poly_block_redundancy.chunks_exact_mut(block_size) {
    //         block[..box_size].fill(1);
    //     }
    //
    //     // println!("poly_redundancy = {:?}", poly_block_redundancy);
    //     let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
    //     //=======================================================================
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_1,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers,
    //         modulus as u64,
    //         0,
    //         &poly_redundancy,
    //         // &rlwe_sk,
    //         // &lwe_sk,
    //         &poly_acc_1,
    //     );
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_2,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers.clone(),
    //         modulus as u64,
    //         0,
    //         &poly_redundancy,
    //         // &rlwe_sk,
    //         // &lwe_sk,
    //         &poly_acc_2,
    //     );
    //
    //     let c_1 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_1),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     let c_2 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_2),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     Ok(vec![c_1, c_2])
    // }
    //
    // pub fn two_block_pbs_base<F>(
    //     &mut self,
    //     treepbskey: &TreepbsKey,
    //     sks: &ServerKey,
    //     ct_1: &Ciphertext,
    //     ct_2: &Ciphertext,
    //     f: F,
    // ) -> EngineResult<Vec<Ciphertext>>
    // where
    //     F: Fn(u64) -> u64,
    // {
    //     //Create the buffers
    //     //=======================================================================
    //     let mut lwe_out_1 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut lwe_out_2 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut buffers = FourierBuffers::new(
    //         sks.bootstrapping_key.polynomial_size(),
    //         sks.bootstrapping_key.glwe_dimension().to_glwe_size(),
    //     );
    //     //=======================================================================
    //
    //     //Keyswitch the ciphertexts
    //     //=======================================================================
    //     let vec_lwe_in = vec![ct_1.ct.clone().0, ct_2.ct.clone().0];
    //     let empty_selector =
    //         LweCiphertext::allocate(0_u64, sks.bootstrapping_key.0.key_size().to_lwe_size());
    //     let mut selectors = vec![empty_selector; vec_lwe_in.len()];
    //     sks.key_switching_key
    //         .0
    //         .vector_keyswitch(&mut selectors, &vec_lwe_in);
    //     //=======================================================================
    //
    //     let modulus = sks.message_modulus.0 * sks.carry_modulus.0;
    //
    //     let base = sks.message_modulus.0 as u64;
    //     let polynomial_size = sks.bootstrapping_key.polynomial_size();
    //
    //     let g_1 = |x: u64, y: u64| f(x + base * y) % base;
    //     let g_2 = |x: u64, y: u64| (f(x + base * y) / base) % base;
    //
    //     let mut poly_acc_1 = Vec::with_capacity(base as usize);
    //     generate_fourier_polynomial_two_variables_base(
    //         g_1,
    //         modulus,
    //         base as usize,
    //         polynomial_size,
    //         &mut poly_acc_1,
    //     );
    //
    //     let mut poly_acc_2 = Vec::with_capacity(base as usize);
    //     generate_fourier_polynomial_two_variables_base(
    //         g_2,
    //         modulus,
    //         base as usize,
    //         polynomial_size,
    //         &mut poly_acc_2,
    //     );
    //
    //     //Create the polynomial to multiply the accumulator with
    //     //=======================================================================
    //     let mut poly_block_redundancy = vec![0_u64; sks.bootstrapping_key.polynomial_size().0];
    //
    //     // N/(p/2) = size of each block
    //     let box_size = sks.bootstrapping_key.polynomial_size().0 / modulus;
    //
    //     let block_size = box_size * base as usize;
    //
    //     for block in poly_block_redundancy.chunks_exact_mut(block_size) {
    //         block[..box_size].fill(1);
    //     }
    //
    //     // println!("poly_redundancy = {:?}", poly_block_redundancy);
    //     let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
    //     //=======================================================================
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue_base(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_1,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers,
    //         modulus as u64,
    //         base,
    //         0,
    //         &poly_redundancy,
    //         // &rlwe_sk,
    //         // &lwe_sk,
    //         &poly_acc_1,
    //     );
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue_base(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_2,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers.clone(),
    //         modulus as u64,
    //         base,
    //         0,
    //         &poly_redundancy,
    //         // &rlwe_sk,
    //         // &lwe_sk,
    //         &poly_acc_2,
    //     );
    //
    //     let c_1 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_1),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     let c_2 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_2),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     Ok(vec![c_1, c_2])
    // }
    //
    // pub fn three_block_pbs<F>(
    //     &mut self,
    //     treepbskey: &TreepbsKey,
    //     sks: &ServerKey,
    //     ct_1: &Ciphertext,
    //     ct_2: &Ciphertext,
    //     ct_3: &Ciphertext,
    //     f: F,
    // ) -> EngineResult<Vec<Ciphertext>>
    // where
    //     F: Fn(u64) -> u64,
    // {
    //     //Create the buffers
    //     //=======================================================================
    //     let mut lwe_out_1 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut lwe_out_2 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut lwe_out_3 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut buffers = FourierBuffers::new(
    //         sks.bootstrapping_key.polynomial_size(),
    //         sks.bootstrapping_key.glwe_dimension().to_glwe_size(),
    //     );
    //     //=======================================================================
    //
    //     //Keyswitch the ciphertexts
    //     //=======================================================================
    //     let vec_lwe_in = vec![ct_1.ct.clone().0, ct_2.ct.clone().0, ct_3.ct.0.clone()];
    //     let empty_selector =
    //         LweCiphertext::allocate(0_u64, sks.bootstrapping_key.0.key_size().to_lwe_size());
    //     let mut selectors = vec![empty_selector; vec_lwe_in.len()];
    //     sks.key_switching_key
    //         .0
    //         .vector_keyswitch(&mut selectors, &vec_lwe_in);
    //     //=======================================================================
    //
    //     let modulus = sks.message_modulus.0 * sks.carry_modulus.0;
    //     let modulus_2 = modulus * modulus;
    //
    //     let base = sks.message_modulus.0 as u64;
    //     let base_2 = base * base;
    //     let polynomial_size = sks.bootstrapping_key.polynomial_size();
    //
    //     let g_1 = |x: u64, y: u64, z: u64| f(x + base * y + base_2 * z) % base;
    //     let g_2 = |x: u64, y: u64, z: u64| (f(x + base * y + base_2 * z) / base) % base;
    //     let g_3 = |x: u64, y: u64, z: u64| (f(x + base * y + base_2 * z) / base_2) % base;
    //
    //     let mut poly_acc_1 = Vec::with_capacity(modulus_2);
    //     generate_fourier_polynomial_three_variables(g_1, modulus, polynomial_size, &mut
    // poly_acc_1);
    //
    //     let mut poly_acc_2 = Vec::with_capacity(modulus_2);
    //     generate_fourier_polynomial_three_variables(g_2, modulus, polynomial_size, &mut
    // poly_acc_2);
    //
    //     let mut poly_acc_3 = Vec::with_capacity(modulus_2);
    //     generate_fourier_polynomial_three_variables(g_3, modulus, polynomial_size, &mut
    // poly_acc_3);
    //
    //     //Create the polynomial to multiply the accumulator with
    //     //=======================================================================
    //     let mut poly_block_redundancy = vec![0_u64; sks.bootstrapping_key.polynomial_size().0];
    //
    //     // N/(p/2) = size of each block
    //     let box_size = sks.bootstrapping_key.polynomial_size().0 / modulus;
    //
    //     let block_size = box_size * modulus;
    //
    //     for block in poly_block_redundancy.chunks_exact_mut(block_size) {
    //         block[..box_size].fill(1);
    //     }
    //
    //     let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
    //     //=======================================================================
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_1,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers,
    //         modulus as u64,
    //         0,
    //         &poly_redundancy,
    //         &poly_acc_1,
    //     );
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_2,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers,
    //         modulus as u64,
    //         0,
    //         &poly_redundancy,
    //         &poly_acc_2,
    //     );
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_3,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers,
    //         modulus as u64,
    //         0,
    //         &poly_redundancy,
    //         &poly_acc_3,
    //     );
    //
    //     let c_1 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_1),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     let c_2 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_2),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     let c_3 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_3),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     Ok(vec![c_1, c_2, c_3])
    // }
    //
    // pub fn three_block_pbs_base<F>(
    //     &mut self,
    //     treepbskey: &TreepbsKey,
    //     sks: &ServerKey,
    //     ct_1: &Ciphertext,
    //     ct_2: &Ciphertext,
    //     ct_3: &Ciphertext,
    //     f: F,
    // ) -> EngineResult<Vec<Ciphertext>>
    // where
    //     F: Fn(u64) -> u64,
    // {
    //     //Create the buffers
    //     //=======================================================================
    //     let mut lwe_out_1 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut lwe_out_2 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut lwe_out_3 = LweCiphertext::allocate(
    //         0_u64,
    //         sks.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
    //     );
    //
    //     let mut buffers = FourierBuffers::new(
    //         sks.bootstrapping_key.polynomial_size(),
    //         sks.bootstrapping_key.glwe_dimension().to_glwe_size(),
    //     );
    //     //=======================================================================
    //
    //     //Keyswitch the ciphertexts
    //     //=======================================================================
    //     let vec_lwe_in = vec![ct_1.ct.clone().0, ct_2.ct.clone().0, ct_3.ct.0.clone()];
    //     let empty_selector =
    //         LweCiphertext::allocate(0_u64, sks.bootstrapping_key.0.key_size().to_lwe_size());
    //     let mut selectors = vec![empty_selector; vec_lwe_in.len()];
    //     sks.key_switching_key
    //         .0
    //         .vector_keyswitch(&mut selectors, &vec_lwe_in);
    //     //=======================================================================
    //
    //     let modulus = sks.message_modulus.0 * sks.carry_modulus.0;
    //     let modulus_2 = modulus * modulus;
    //
    //     let base = sks.message_modulus.0 as u64;
    //     let base_2 = base * base;
    //     let polynomial_size = sks.bootstrapping_key.polynomial_size();
    //
    //     let g_1 = |x: u64, y: u64, z: u64| f(x + base * y + base_2 * z) % base;
    //     let g_2 = |x: u64, y: u64, z: u64| (f(x + base * y + base_2 * z) / base) % base;
    //     let g_3 = |x: u64, y: u64, z: u64| (f(x + base * y + base_2 * z) / base_2) % base;
    //
    //     let mut poly_acc_1 = Vec::with_capacity(base_2 as usize);
    //     generate_fourier_polynomial_three_variables_base(g_1, modulus, base as usize,
    // polynomial_size, &mut poly_acc_1);
    //
    //     let mut poly_acc_2 = Vec::with_capacity(base_2 as usize);
    //     generate_fourier_polynomial_three_variables_base(g_2, modulus, base as usize,
    // polynomial_size, &mut poly_acc_2);
    //
    //     let mut poly_acc_3 = Vec::with_capacity(base_2 as usize);
    //     generate_fourier_polynomial_three_variables_base(g_3, modulus, base as usize,
    // polynomial_size, &mut poly_acc_3);
    //
    //     //Create the polynomial to multiply the accumulator with
    //     //=======================================================================
    //     let mut poly_block_redundancy = vec![0_u64; sks.bootstrapping_key.polynomial_size().0];
    //
    //     // N/(p/2) = size of each block
    //     let box_size = sks.bootstrapping_key.polynomial_size().0 / modulus;
    //
    //     let block_size = box_size * base as usize;
    //
    //     for block in poly_block_redundancy.chunks_exact_mut(block_size) {
    //         block[..box_size].fill(1);
    //     }
    //
    //     let poly_redundancy = Polynomial::from_container(poly_block_redundancy);
    //     //=======================================================================
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue_base(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_1,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers,
    //         modulus as u64,
    //         base,
    //         0,
    //         &poly_redundancy,
    //         &poly_acc_1,
    //     );
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue_base(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_2,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers,
    //         modulus as u64,
    //         base,
    //         0,
    //         &poly_redundancy,
    //         &poly_acc_2,
    //     );
    //
    //     sks.bootstrapping_key.0.treepbs_with_multivalue_base(
    //         &treepbskey.pksk.0,
    //         &mut lwe_out_3,
    //         selectors.as_slice(),
    //         &sks.key_switching_key.0,
    //         &mut buffers,
    //         modulus as u64,
    //         base,
    //         0,
    //         &poly_redundancy,
    //         &poly_acc_3,
    //     );
    //
    //     let c_1 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_1),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     let c_2 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_2),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     let c_3 = Ciphertext {
    //         ct: LweCiphertext64(lwe_out_3),
    //         degree: Degree(sks.message_modulus.0 - 1),
    //         message_modulus: sks.message_modulus,
    //         carry_modulus: sks.carry_modulus,
    //     };
    //
    //     Ok(vec![c_1, c_2, c_3])
    // }
}
