use super::decomposition::DecompositionTerm;
use super::fpks::LweKeyBitDecomposition;
use super::polynomial::{update_with_wrapping_add_mul, update_with_wrapping_sub_mul};
use super::types::polynomial::Polynomial;
use super::types::*;
use super::{from_torus, zip_eq};
use core::slice;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::cmp::Ordering;

pub fn mask_bytes_per_coef() -> usize {
    u64::BITS as usize / 8
}

pub fn mask_bytes_per_polynomial(polynomial_size: usize) -> usize {
    polynomial_size * mask_bytes_per_coef()
}

pub fn mask_bytes_per_glwe(glwe_params: GlweParams) -> usize {
    glwe_params.dimension * mask_bytes_per_polynomial(glwe_params.polynomial_size)
}

pub fn mask_bytes_per_ggsw_level(glwe_params: GlweParams) -> usize {
    (glwe_params.dimension + 1) * mask_bytes_per_glwe(glwe_params)
}

pub fn mask_bytes_per_lwe(lwe_dimension: usize) -> usize {
    lwe_dimension * mask_bytes_per_coef()
}

pub fn mask_bytes_per_gsw_level(lwe_dimension: usize) -> usize {
    (lwe_dimension + 1) * mask_bytes_per_lwe(lwe_dimension)
}

pub fn mask_bytes_per_ggsw(decomposition_level_count: usize, glwe_params: GlweParams) -> usize {
    decomposition_level_count * mask_bytes_per_ggsw_level(glwe_params)
}

pub fn mask_bytes_per_pfpksk_chunk(
    decomposition_level_count: usize,
    glwe_params: GlweParams,
) -> usize {
    decomposition_level_count * mask_bytes_per_glwe(glwe_params)
}

pub fn mask_bytes_per_pfpksk(
    decomposition_level_count: usize,
    glwe_params: GlweParams,
    lwe_dimension: usize,
) -> usize {
    (lwe_dimension + 1) * mask_bytes_per_pfpksk_chunk(decomposition_level_count, glwe_params)
}

pub fn noise_bytes_per_coef() -> usize {
    // We use f64 to sample the noise from a normal distribution with the polar form of the
    // Box-Muller algorithm. With this algorithm, the input pair of uniform values will be rejected
    // with a probability of  pi/4 which means that in average, we need ~4/pi pair of uniform
    // values for one pair of normal values. To have a safety margin, we require 32 uniform inputs
    // (>> 4/pi) for one pair of normal values
    8 * 32
}
pub fn noise_bytes_per_polynomial(polynomial_size: usize) -> usize {
    polynomial_size * noise_bytes_per_coef()
}

pub fn noise_bytes_per_glwe(polynomial_size: usize) -> usize {
    noise_bytes_per_polynomial(polynomial_size)
}

pub fn noise_bytes_per_ggsw_level(glwe_params: GlweParams) -> usize {
    (glwe_params.dimension + 1) * noise_bytes_per_glwe(glwe_params.polynomial_size)
}

pub fn noise_bytes_per_lwe() -> usize {
    // Here we take 3 to keep a safety margin
    noise_bytes_per_coef() * 3
}

pub fn noise_bytes_per_gsw_level(lwe_dimension: usize) -> usize {
    (lwe_dimension + 1) * noise_bytes_per_lwe()
}

pub fn noise_bytes_per_ggsw(decomposition_level_count: usize, glwe_params: GlweParams) -> usize {
    decomposition_level_count * noise_bytes_per_ggsw_level(glwe_params)
}

pub fn noise_bytes_per_pfpksk_chunk(
    decomposition_level_count: usize,
    polynomial_size: usize,
) -> usize {
    decomposition_level_count * noise_bytes_per_glwe(polynomial_size)
}

pub fn noise_bytes_per_pfpksk(
    decomposition_level_count: usize,
    polynomial_size: usize,
    lwe_dimension: usize,
) -> usize {
    (lwe_dimension + 1) * noise_bytes_per_pfpksk_chunk(decomposition_level_count, polynomial_size)
}

pub fn fill_with_random_uniform(buffer: &mut [u64], mut csprng: CsprngMut<'_, '_>) {
    #[cfg(target_endian = "little")]
    {
        let len = std::mem::size_of_val(buffer);
        let random_bytes = csprng
            .as_mut()
            .next_bytes(unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr() as _, len) });
        assert_eq!(len, random_bytes);
    }
    #[cfg(target_endian = "big")]
    {
        let mut little_endian = [0u8; core::mem::size_of::<u64>()];
        for e in buffer {
            let random_bytes = csprng.as_mut().next_bytes(&mut little_endian);
            assert_eq!(little_endian.len(), random_bytes);
            *e = u64::from_le_bytes(little_endian);
        }
    }
}

fn random_gaussian_pair(variance: f64, mut csprng: CsprngMut<'_, '_>) -> (f64, f64) {
    loop {
        let mut uniform_rand = [0_u64, 0_u64];
        fill_with_random_uniform(&mut uniform_rand, csprng.as_mut());
        let uniform_rand =
            uniform_rand.map(|x| (x as i64 as f64) * 2.0_f64.powi(1 - u64::BITS as i32));
        let u = uniform_rand[0];
        let v = uniform_rand[1];

        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            let cst = (-2.0 * variance * s.ln() / s).sqrt();
            return (u * cst, v * cst);
        }
    }
}

pub fn fill_with_random_gaussian(buffer: &mut [u64], variance: f64, mut csprng: CsprngMut<'_, '_>) {
    for chunk in buffer.chunks_exact_mut(2) {
        let (g0, g1) = random_gaussian_pair(variance, csprng.as_mut());
        if let Some(first) = chunk.get_mut(0) {
            *first = from_torus(g0);
        }
        if let Some(second) = chunk.get_mut(1) {
            *second = from_torus(g1);
        }
    }
}

impl BootstrapKey<&mut [u64]> {
    pub fn fill_with_new_key(
        self,
        lwe_sk: LweSecretKey<&[u64]>,
        glwe_sk: GlweSecretKey<&[u64]>,
        variance: f64,
        mut csprng: CsprngMut<'_, '_>,
    ) {
        for (mut ggsw, sk_scalar) in
            zip_eq(self.into_ggsw_iter(), lwe_sk.into_data().iter().copied())
        {
            let encoded = sk_scalar;
            gen_noise_ggsw(glwe_sk, ggsw.as_mut_view(), variance, csprng.as_mut());
            encrypt_constant_ggsw_noise_full(glwe_sk, glwe_sk, ggsw, encoded);
        }
    }
}

#[cfg(feature = "parallel")]
impl BootstrapKey<&mut [u64]> {
    pub fn fill_with_new_key_par(
        mut self,
        lwe_sk: LweSecretKey<&[u64]>,
        glwe_sk: GlweSecretKey<&[u64]>,
        variance: f64,
        mut csprng: CsprngMut<'_, '_>,
    ) {
        for ggsw in self.as_mut_view().into_ggsw_iter() {
            gen_noise_ggsw(glwe_sk, ggsw, variance, csprng.as_mut());
        }

        self.into_ggsw_iter_par()
            .zip_eq(lwe_sk.data)
            .for_each(|(ggsw, sk_scalar)| {
                let encoded = *sk_scalar;
                encrypt_constant_ggsw_noise_full(glwe_sk, glwe_sk, ggsw, encoded);
            });
    }
}

impl LweKeyswitchKey<&mut [u64]> {
    pub fn fill_with_keyswitch_key(
        self,
        input_key: LweSecretKey<&[u64]>,
        output_key: LweSecretKey<&[u64]>,
        variance: f64,
        mut csprng: CsprngMut<'_, '_>,
    ) {
        let decomposition_level_count = self.decomp_params.level;
        let decomposition_base_log = self.decomp_params.base_log;

        // loop over the before key blocks
        for (input_key_bit, keyswitch_key_block) in zip_eq(
            input_key.into_data().iter().copied(),
            self.into_lev_ciphertexts(),
        ) {
            // we encrypt the buffer
            for (lwe, message) in zip_eq(
                keyswitch_key_block.into_ciphertext_iter(),
                (1..(decomposition_level_count + 1)).map(|level| {
                    let shift = u64::BITS as usize - decomposition_base_log * level;
                    input_key_bit << shift
                }),
            ) {
                output_key.encrypt_lwe(lwe, message, variance, csprng.as_mut());
            }
        }
    }
}

impl PackingKeyswitchKey<&mut [u64]> {
    pub fn fill_with_packing_keyswitch_key(
        self,
        input_lwe_key: LweSecretKey<&[u64]>,
        output_glwe_key: GlweSecretKey<&[u64]>,
        variance: f64,
        mut csprng: CsprngMut<'_, '_>,
    ) {
        let decomposition_level_count = self.decomp_params.level;
        let decomposition_base_log = self.decomp_params.base_log;

        for (input_key_bit, keyswitch_key_block) in zip_eq(
            input_lwe_key.into_data().iter(),
            self.into_glev_ciphertexts(),
        ) {
            for (mut glwe, message) in zip_eq(
                keyswitch_key_block.into_ciphertext_iter(),
                (1..(decomposition_level_count + 1)).map(|level| {
                    let shift = u64::BITS as usize - decomposition_base_log * level;
                    input_key_bit << shift
                }),
            ) {
                output_glwe_key.encrypt_zero_glwe(glwe.as_mut_view(), variance, csprng.as_mut());
                let (_, body) = glwe.into_mask_and_body();
                let body = body.into_data();
                let first = body.first_mut().unwrap();
                *first = first.wrapping_add(message);
            }
        }
    }

    pub fn fill_with_private_functional_packing_keyswitch_key(
        &mut self,
        input_lwe_key: &LweSecretKey<&[u64]>,
        output_glwe_key: &GlweSecretKey<&[u64]>,
        variance: f64,
        mut csprng: CsprngMut,
        f: impl Fn(u64) -> u64,
        polynomial: Polynomial<&[u64]>,
    ) {
        // We instantiate a buffer
        let mut messages = vec![0_u64; self.decomp_params.level * self.glwe_params.polynomial_size];

        // We retrieve decomposition arguments
        let decomp_level_count = self.decomp_params.level;
        let decomp_base_log = self.decomp_params.base_log;
        let polynomial_size = self.glwe_params.polynomial_size;

        // add minus one for the function which will be applied to the decomposed body
        // ( Scalar::MAX = -Scalar::ONE )
        let input_key_bit_iter = input_lwe_key.data.iter().chain(std::iter::once(&u64::MAX));

        // loop over the before key blocks
        for (&input_key_bit, keyswitch_key_block) in
            zip_eq(input_key_bit_iter, self.bit_decomp_iter_mut())
        {
            // We reset the buffer
            messages.fill(0);

            // We fill the buffer with the powers of the key bits
            for (level, message) in zip_eq(
                1..=decomp_level_count,
                messages.chunks_exact_mut(polynomial_size),
            ) {
                let multiplier = DecompositionTerm::new(
                    level,
                    decomp_base_log,
                    f(1).wrapping_mul(input_key_bit),
                )
                .to_recomposition_summand();
                for (self_i, other_i) in zip_eq(message, polynomial.as_ref().into_data()) {
                    *self_i = (*self_i).wrapping_add(other_i.wrapping_mul(multiplier));
                }
            }

            // We encrypt the buffer

            for (mut glwe, message) in zip_eq(
                keyswitch_key_block.into_glwe_list().into_glwe_iter(),
                messages.chunks_exact(polynomial_size),
            ) {
                output_glwe_key.encrypt_zero_glwe(glwe.as_mut_view(), variance, csprng.as_mut());

                let (_, body) = glwe.into_mask_and_body();

                for (r, e) in zip_eq(body.into_data().iter_mut(), message) {
                    *r = r.wrapping_add(*e)
                }
            }
        }
    }
}

impl<'a> PackingKeyswitchKey<&'a mut [u64]> {
    pub fn fill_with_private_functional_packing_keyswitch_key_par(
        &'a mut self,
        input_lwe_key: &LweSecretKey<&[u64]>,
        output_glwe_key: &GlweSecretKey<&[u64]>,
        variance: f64,
        mut csprng: CsprngMut,
        f: impl Sync + Fn(u64) -> u64,
        polynomial: Polynomial<&[u64]>,
    ) {
        // We retrieve decomposition arguments
        let decomp_level_count = self.decomp_params.level;
        let decomp_base_log = self.decomp_params.base_log;
        let polynomial_size = self.glwe_params.polynomial_size;

        // loop over the before key blocks
        for keyswitch_key_block in self.bit_decomp_iter_mut() {
            // We encrypt the buffer

            for mut glwe in keyswitch_key_block.into_glwe_list().into_glwe_iter() {
                output_glwe_key.gen_noise_glwe(glwe.as_mut_view(), variance, csprng.as_mut());
            }
        }
        let input_dimension = self.input_dimension;

        // loop over the before key blocks
        self.bit_decomp_iter_mut_par()
            .enumerate()
            .for_each(|(i, keyswitch_key_block)| {
                // add minus one for the function which will be applied to the decomposed body
                // ( Scalar::MAX = -Scalar::ONE )
                let input_key_bit = match i.cmp(&input_dimension) {
                    Ordering::Less => input_lwe_key.data[i],
                    Ordering::Equal => u64::MAX,
                    Ordering::Greater => unreachable!(),
                };

                // We instantiate a buffer
                let mut messages = vec![0_u64; decomp_level_count * polynomial_size];

                // We reset the buffer
                messages.fill(0);

                // We fill the buffer with the powers of the key bits
                for (level, message) in zip_eq(
                    1..=decomp_level_count,
                    messages.chunks_exact_mut(polynomial_size),
                ) {
                    let multiplier = DecompositionTerm::new(
                        level,
                        decomp_base_log,
                        f(1).wrapping_mul(input_key_bit),
                    )
                    .to_recomposition_summand();
                    for (self_i, other_i) in zip_eq(message, polynomial.as_ref().into_data()) {
                        *self_i = (*self_i).wrapping_add(other_i.wrapping_mul(multiplier));
                    }
                }

                // We encrypt the buffer

                for (mut glwe, message) in zip_eq(
                    keyswitch_key_block.into_glwe_list().into_glwe_iter(),
                    messages.chunks_exact(polynomial_size),
                ) {
                    output_glwe_key.encrypt_zero_glwe_noise_full(glwe.as_mut_view());

                    let (_, body) = glwe.into_mask_and_body();

                    for (r, e) in zip_eq(body.into_data().iter_mut(), message) {
                        *r = r.wrapping_add(*e)
                    }
                }
            });
    }

    pub fn bit_decomp_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = LweKeyBitDecomposition<&mut [u64]>> {
        let glwe_params = self.glwe_params;

        let level = self.decomp_params.level;

        let chunks_size = level * (glwe_params.dimension + 1) * glwe_params.polynomial_size;

        self.as_mut_view()
            .into_data()
            .chunks_exact_mut(chunks_size)
            .map(move |sub| LweKeyBitDecomposition::new(sub, glwe_params, level))
    }
}

#[cfg(feature = "parallel")]
impl<'a> PackingKeyswitchKey<&'a mut [u64]> {
    pub fn bit_decomp_iter_mut_par(
        &'a mut self,
    ) -> impl 'a + IndexedParallelIterator<Item = LweKeyBitDecomposition<&'a mut [u64]>> {
        let glwe_params = self.glwe_params;

        let level = self.decomp_params.level;

        let chunks_size = level * (glwe_params.dimension + 1) * glwe_params.polynomial_size;

        self.as_mut_view()
            .into_data()
            .par_chunks_exact_mut(chunks_size)
            .map(move |sub| LweKeyBitDecomposition::new(sub, glwe_params, level))
    }
}

impl PackingKeyswitchKey<&[u64]> {
    pub fn bit_decomp_iter(&self) -> impl Iterator<Item = LweKeyBitDecomposition<&[u64]>> {
        let glwe_params = self.glwe_params;

        let level = self.decomp_params.level;

        let size =
            self.decomp_params.level * (glwe_params.dimension + 1) * glwe_params.polynomial_size;
        self.data
            .chunks_exact(size)
            .map(move |sub| LweKeyBitDecomposition::new(sub, glwe_params, level))
    }
}

pub fn encrypt_constant_ggsw(
    in_glwe_sk: GlweSecretKey<&[u64]>,
    out_glwe_sk: GlweSecretKey<&[u64]>,
    ggsw: GgswCiphertext<&mut [u64]>,
    encoded: u64,
    variance: f64,
    mut csprng: CsprngMut<'_, '_>,
) {
    debug_assert_eq!(in_glwe_sk.glwe_params, ggsw.in_glwe_params());
    debug_assert_eq!(out_glwe_sk.glwe_params, ggsw.out_glwe_params());

    let base_log = ggsw.decomp_params.base_log;

    let out_glwe_params = ggsw.out_glwe_params();

    for matrix in ggsw.into_level_matrices_iter() {
        let factor = encoded.wrapping_neg()
            << (u64::BITS as usize - (base_log * matrix.decomposition_level));

        let last_row_index = matrix.in_glwe_dimension;

        for (row_index, row) in matrix.into_rows_iter().enumerate() {
            encrypt_constant_ggsw_row(
                in_glwe_sk,
                out_glwe_sk,
                (row_index, last_row_index),
                factor,
                GlweCiphertext::new(row.into_data(), out_glwe_params),
                variance,
                csprng.as_mut(),
            );
        }
    }
}

pub fn encrypt_constant_ggsw_row(
    in_glwe_sk: GlweSecretKey<&[u64]>,
    out_glwe_sk: GlweSecretKey<&[u64]>,
    (row_index, last_row_index): (usize, usize),
    factor: u64,
    mut row: GlweCiphertext<&mut [u64]>,
    variance: f64,
    csprng: CsprngMut<'_, '_>,
) {
    if row_index < last_row_index {
        // Not the last row
        let sk_poly = in_glwe_sk.get_polynomial(row_index);
        let encoded = sk_poly.iter().map(|&e| e.wrapping_mul(factor));

        out_glwe_sk.encrypt_zero_glwe(row.as_mut_view(), variance, csprng);
        let (_, body) = row.into_mask_and_body();
        for (r, e) in zip_eq(body.into_data().iter_mut(), encoded) {
            *r = r.wrapping_add(e)
        }
    } else {
        // The last row needs a slightly different treatment
        out_glwe_sk.encrypt_zero_glwe(row.as_mut_view(), variance, csprng);
        let (_, body) = row.into_mask_and_body();
        let first = body.into_data().first_mut().unwrap();
        *first = first.wrapping_add(factor.wrapping_neg());
    }
}
pub fn gen_noise_ggsw(
    out_glwe_sk: GlweSecretKey<&[u64]>,
    ggsw: GgswCiphertext<&mut [u64]>,
    variance: f64,
    mut csprng: CsprngMut<'_, '_>,
) {
    let glwe_params = ggsw.out_glwe_params();

    for matrix in ggsw.into_level_matrices_iter() {
        for row in matrix.into_rows_iter() {
            out_glwe_sk.gen_noise_glwe(
                GlweCiphertext::new(row.into_data(), glwe_params),
                variance,
                csprng.as_mut(),
            );
        }
    }
}

pub fn encrypt_constant_ggsw_row_noise_full(
    in_glwe_sk: GlweSecretKey<&[u64]>,
    out_glwe_sk: GlweSecretKey<&[u64]>,
    (row_index, last_row_index): (usize, usize),
    factor: u64,
    mut row: GlweCiphertext<&mut [u64]>,
) {
    if row_index < last_row_index {
        // Not the last row
        let sk_poly = in_glwe_sk.get_polynomial(row_index);
        let encoded = sk_poly.iter().map(|&e| e.wrapping_mul(factor));

        out_glwe_sk.encrypt_zero_glwe_noise_full(row.as_mut_view());
        let (_, body) = row.into_mask_and_body();
        for (r, e) in zip_eq(body.into_data().iter_mut(), encoded) {
            *r = r.wrapping_add(e)
        }
    } else {
        // The last row needs a slightly different treatment
        out_glwe_sk.encrypt_zero_glwe_noise_full(row.as_mut_view());
        let (_, body) = row.into_mask_and_body();
        let first = body.into_data().first_mut().unwrap();
        *first = first.wrapping_add(factor.wrapping_neg());
    }
}

pub fn encrypt_constant_ggsw_noise_full(
    in_glwe_sk: GlweSecretKey<&[u64]>,
    out_glwe_sk: GlweSecretKey<&[u64]>,
    ggsw: GgswCiphertext<&mut [u64]>,
    encoded: u64,
) {
    let base_log = ggsw.decomp_params.base_log;

    let out_glwe_params = ggsw.out_glwe_params();

    for matrix in ggsw.into_level_matrices_iter() {
        let factor = encoded.wrapping_neg()
            << (u64::BITS as usize - (base_log * matrix.decomposition_level));

        let last_row_index = matrix.out_glwe_dimension;

        for (row_index, row) in matrix.into_rows_iter().enumerate() {
            encrypt_constant_ggsw_row_noise_full(
                in_glwe_sk,
                out_glwe_sk,
                (row_index, last_row_index),
                factor,
                GlweCiphertext::new(row.into_data(), out_glwe_params),
            );
        }
    }
}

impl GlweSecretKey<&[u64]> {
    pub fn encrypt_zero_glwe_noise_full(self, encrypted: GlweCiphertext<&mut [u64]>) {
        let (mask, mut body) = encrypted.into_mask_and_body();

        let mask = mask.as_view();
        for (poly, bin_poly) in zip_eq(mask.iter_polynomial(), self.iter()) {
            update_with_wrapping_add_mul(body.as_mut_view(), poly, bin_poly)
        }
    }

    pub fn encrypt_zero_glwe(
        self,
        encrypted: GlweCiphertext<&mut [u64]>,
        variance: f64,
        mut csprng: CsprngMut<'_, '_>,
    ) {
        let (mut mask, mut body) = encrypted.into_mask_and_body();
        fill_with_random_uniform(mask.as_mut_view().into_data(), csprng.as_mut());
        fill_with_random_gaussian(body.as_mut_view().into_data(), variance, csprng);

        let mask = mask.as_view();

        for (poly, bin_poly) in zip_eq(mask.iter_polynomial(), self.iter()) {
            let body = body.as_mut_view();
            update_with_wrapping_add_mul(body, poly, bin_poly)
        }
    }
    pub fn decrypt_glwe_inplace(self, encrypted: GlweCiphertext<&[u64]>, out: &mut [u64]) {
        let (mask, body) = encrypted.into_mask_and_body();

        let mask = mask.as_view();
        out.copy_from_slice(body.into_data());
        for (poly, bin_poly) in zip_eq(mask.iter_polynomial(), self.iter()) {
            update_with_wrapping_sub_mul(
                Polynomial::new(out, encrypted.glwe_params.polynomial_size),
                poly,
                bin_poly,
            )
        }
    }

    pub fn decrypt_glwe(self, encrypted: GlweCiphertext<&[u64]>) -> Vec<u64> {
        let mut out = vec![0; encrypted.glwe_params.polynomial_size];

        self.decrypt_glwe_inplace(encrypted, &mut out);

        out
    }

    pub fn gen_noise_glwe(
        self,
        encrypted: GlweCiphertext<&mut [u64]>,
        variance: f64,
        mut csprng: CsprngMut<'_, '_>,
    ) {
        let (mut mask, mut body) = encrypted.into_mask_and_body();
        fill_with_random_uniform(mask.as_mut_view().into_data(), csprng.as_mut());
        fill_with_random_gaussian(body.as_mut_view().into_data(), variance, csprng);
    }
}

impl LweSecretKey<&[u64]> {
    pub fn encrypt_lwe(
        self,
        encrypted: LweCiphertext<&mut [u64]>,
        plaintext: u64,
        variance: f64,
        mut csprng: CsprngMut<'_, '_>,
    ) {
        let (body, mask) = encrypted.into_data().split_last_mut().unwrap();

        fill_with_random_uniform(mask, csprng.as_mut());
        *body = from_torus(random_gaussian_pair(variance, csprng.as_mut()).0);
        *body = body.wrapping_add(
            zip_eq(mask.iter().copied(), self.into_data().iter().copied())
                .fold(0_u64, |acc, (lhs, rhs)| acc.wrapping_add(lhs * rhs)),
        );
        *body = body.wrapping_add(plaintext);
    }

    pub fn decrypt_lwe(self, encrypted: LweCiphertext<&[u64]>) -> u64 {
        let (body, mask) = encrypted.into_data().split_last().unwrap();

        body.wrapping_sub(
            zip_eq(mask.iter().copied(), self.into_data().iter().copied())
                .fold(0_u64, |acc, (lhs, rhs)| acc.wrapping_add(lhs * rhs)),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::c_api::types::tests::to_generic;
    use crate::implementation::types::{CsprngMut, LweCiphertext, LweSecretKey};
    use concrete_csprng::generators::{RandomGenerator, SoftwareRandomGenerator};
    use concrete_csprng::seeders::Seed;

    fn encrypt_decrypt(
        mut csprng: CsprngMut,
        pt: u64,
        dim: usize,
        encryption_variance: f64,
    ) -> u64 {
        let mut ct = LweCiphertext::zero(dim);

        let sk = LweSecretKey::new_random(csprng.as_mut(), dim);

        sk.as_view()
            .encrypt_lwe(ct.as_mut_view(), pt, encryption_variance, csprng);

        sk.as_view().decrypt_lwe(ct.as_view())
    }

    #[test]
    fn encryption_decryption_correctness() {
        let mut csprng = SoftwareRandomGenerator::new(Seed(0));

        for _ in 0..100 {
            let a: u64 = u64::from_le_bytes(std::array::from_fn(|_| csprng.next().unwrap()));

            let b = encrypt_decrypt(to_generic(&mut csprng), a, 1024, 0.0000000001);

            let diff = b.wrapping_sub(a) as i64;

            assert!((diff as f64).abs() / 2.0_f64.powi(64) < 0.0001);
        }
    }
}
