#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::{DynStack, ReborrowMut, SizeOverflow, StackReq};

use crate::implementation::external_product::external_product;
use crate::implementation::types::GlweCiphertext;
use crate::implementation::zip_eq;

use super::cmux::{cmux, cmux_scratch};
use super::external_product::external_product_scratch;
use super::fft::FftView;
use super::polynomial::update_with_wrapping_unit_monomial_div;
use super::types::ciphertext_list::LweCiphertextList;
use super::types::packing_keyswitch_key_list::PackingKeyswitchKeyList;
use super::types::polynomial::Polynomial;
use super::types::polynomial_list::PolynomialList;
use super::types::{
    BootstrapKey, DecompParams, GgswCiphertext, GlweParams, LweCiphertext, LweKeyswitchKey,
};
use super::{Container, Split};

pub fn extract_bits_scratch(
    lwe_dimension: usize,
    ksk_after_key_size: usize,
    glwe_params: GlweParams,
    fft: FftView,
) -> Result<StackReq, SizeOverflow> {
    let align = CACHELINE_ALIGN;

    let GlweParams {
        dimension,
        polynomial_size,
    } = glwe_params;

    let lwe_in_buffer = StackReq::try_new_aligned::<u64>(lwe_dimension + 1, align)?;
    let lwe_out_ks_buffer = StackReq::try_new_aligned::<u64>(ksk_after_key_size + 1, align)?;
    let pbs_accumulator =
        StackReq::try_new_aligned::<u64>((dimension + 1) * polynomial_size, align)?;
    let lwe_out_pbs_buffer =
        StackReq::try_new_aligned::<u64>(dimension * polynomial_size + 1, align)?;
    let lwe_bit_left_shift_buffer = lwe_in_buffer;
    let bootstrap_scratch = BootstrapKey::bootstrap_scratch(glwe_params, fft)?;

    lwe_in_buffer
        .try_and(lwe_out_ks_buffer)?
        .try_and(pbs_accumulator)?
        .try_and(lwe_out_pbs_buffer)?
        .try_and(StackReq::try_any_of([
            lwe_bit_left_shift_buffer,
            bootstrap_scratch,
        ])?)
}

/// Function to extract `number_of_bits_to_extract` from an [`LweCiphertext`] starting at the bit
/// number `delta_log` (0-indexed) included.
///
/// Output bits are ordered from the MSB to the LSB. Each one of them is output in a distinct LWE
/// ciphertext, containing the encryption of the bit scaled by q/2 (i.e., the most significant bit
/// in the plaintext representation).
pub fn extract_bits(
    mut lwe_list_out: LweCiphertextList<&mut [u64]>,
    lwe_in: LweCiphertext<&[u64]>,
    ksk: LweKeyswitchKey<&[u64]>,
    fourier_bsk: BootstrapKey<&[f64]>,
    delta_log: usize,
    number_of_bits_to_extract: usize,
    fft: FftView<'_>,
    stack: DynStack<'_>,
) {
    let ciphertext_n_bits = u64::BITS as usize;
    let number_of_bits_to_extract = number_of_bits_to_extract;

    debug_assert!(
        ciphertext_n_bits >= number_of_bits_to_extract + delta_log,
        "Tried to extract {} bits, while the maximum number of extractable bits for {} bits
        ciphertexts and a scaling factor of 2^{} is {}",
        number_of_bits_to_extract,
        ciphertext_n_bits,
        delta_log,
        ciphertext_n_bits - delta_log,
    );
    debug_assert_eq!(lwe_list_out.lwe_dimension, ksk.output_dimension,);
    debug_assert_eq!(lwe_list_out.count, number_of_bits_to_extract,);
    debug_assert_eq!(lwe_in.lwe_dimension, fourier_bsk.output_lwe_dimension(),);

    let polynomial_size = fourier_bsk.glwe_params.polynomial_size;
    let glwe_dimension = fourier_bsk.glwe_params.dimension;

    let align = CACHELINE_ALIGN;

    let (mut lwe_in_buffer_data, stack) =
        stack.collect_aligned(align, lwe_in.into_data().iter().copied());
    let mut lwe_in_buffer = LweCiphertext::new(&mut *lwe_in_buffer_data, lwe_in.lwe_dimension);

    let (mut lwe_out_ks_buffer_data, stack) =
        stack.make_aligned_with(ksk.output_dimension + 1, align, |_| 0_u64);
    let mut lwe_out_ks_buffer =
        LweCiphertext::new(&mut *lwe_out_ks_buffer_data, lwe_list_out.lwe_dimension);

    let (mut pbs_accumulator_data, stack) =
        stack.make_aligned_with((glwe_dimension + 1) * polynomial_size, align, |_| 0_u64);
    let mut pbs_accumulator =
        GlweCiphertext::new(&mut *pbs_accumulator_data, fourier_bsk.glwe_params);

    let lwe_size = glwe_dimension * polynomial_size + 1;
    let (mut lwe_out_pbs_buffer_data, mut stack) =
        stack.make_aligned_with(lwe_size, align, |_| 0_u64);
    let mut lwe_out_pbs_buffer = LweCiphertext::new(
        &mut *lwe_out_pbs_buffer_data,
        glwe_dimension * polynomial_size,
    );

    // We iterate on the list in reverse as we want to store the extracted MSB at index 0
    for (bit_idx, mut output_ct) in lwe_list_out
        .as_mut_view()
        .ciphertext_iter_mut()
        .rev()
        .enumerate()
    {
        // Shift on padding bit
        let (lwe_bit_left_shift_buffer_data, _) = stack.rb_mut().collect_aligned(
            align,
            lwe_in_buffer
                .as_view()
                .data
                .iter()
                .map(|s| *s << (ciphertext_n_bits - delta_log - bit_idx - 1)),
        );

        // Key switch to input PBS key
        ksk.keyswitch_ciphertext(
            lwe_out_ks_buffer.as_mut_view(),
            LweCiphertext::new(&*lwe_bit_left_shift_buffer_data, ksk.input_dimension),
        );

        drop(lwe_bit_left_shift_buffer_data);

        // Store the keyswitch output unmodified to the output list (as we need to to do other
        // computations on the output of the keyswitch)
        output_ct
            .as_mut_view()
            .into_data()
            .copy_from_slice(lwe_out_ks_buffer.as_view().into_data());

        // If this was the last extracted bit, break
        // we subtract 1 because if the number_of_bits_to_extract is 1 we want to stop right away
        if bit_idx == number_of_bits_to_extract - 1 {
            break;
        }

        // Add q/4 to center the error while computing a negacyclic LUT
        let out_ks_body = lwe_out_ks_buffer
            .as_mut_view()
            .into_data()
            .last_mut()
            .unwrap();
        *out_ks_body = out_ks_body.wrapping_add(1_u64 << (ciphertext_n_bits - 2));

        // Fill lut for the current bit (equivalent to trivial encryption as mask is 0s)
        // The LUT is filled with -alpha in each coefficient where alpha = delta*2^{bit_idx-1}
        for poly_coeff in pbs_accumulator.as_mut_view().into_body().into_data() {
            *poly_coeff = (1_u64 << (delta_log - 1 + bit_idx)).wrapping_neg();
        }

        fourier_bsk.bootstrap(
            lwe_out_pbs_buffer.as_mut_view(),
            lwe_out_ks_buffer.as_view(),
            pbs_accumulator.as_view(),
            fft,
            stack.rb_mut(),
        );

        // Add alpha where alpha = delta*2^{bit_idx-1} to end up with an encryption of 0 if the
        // extracted bit was 0 and 1 in the other case
        let out_pbs_body = lwe_out_pbs_buffer
            .as_mut_view()
            .into_data()
            .last_mut()
            .unwrap();

        *out_pbs_body = out_pbs_body.wrapping_add(1_u64 << (delta_log + bit_idx - 1));

        // Remove the extracted bit from the initial LWE to get a 0 at the extracted bit location.
        for (out, inp) in zip_eq(
            lwe_in_buffer.as_mut_view().into_data(),
            lwe_out_pbs_buffer.as_view().into_data(),
        ) {
            *out = out.wrapping_sub(*inp);
        }
    }
}

pub fn circuit_bootstrap_boolean_scratch(
    lwe_in_size: usize,
    bsk_output_lwe_size: usize,
    glwe_params: GlweParams,
    fft: FftView,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_new_aligned::<u64>(bsk_output_lwe_size, CACHELINE_ALIGN)?.try_and(
        homomorphic_shift_boolean_scratch(lwe_in_size, glwe_params, fft)?,
    )
}

/// Circuit bootstrapping for boolean messages, i.e. containing only one bit of message
///
/// The output GGSW ciphertext `ggsw_out` decomposition base log and level count are used as the
/// circuit_bootstrap_boolean decomposition base log and level count.
pub fn circuit_bootstrap_boolean(
    fourier_bsk: BootstrapKey<&[f64]>,
    lwe_in: LweCiphertext<&[u64]>,
    ggsw_out: GgswCiphertext<&mut [u64]>,
    delta_log: usize,
    fpksk_list: PackingKeyswitchKeyList<&[u64]>,
    fft: FftView<'_>,
    stack: DynStack<'_>,
) {
    let level_cbs = ggsw_out.decomp_params.level;
    let base_log_cbs = ggsw_out.decomp_params.base_log;

    debug_assert_ne!(level_cbs, 0);
    debug_assert_ne!(base_log_cbs, 0);

    let fpksk_input_lwe_key_dimension = fpksk_list.input_dimension;
    let fourier_bsk_output_lwe_dimension = fourier_bsk.output_lwe_dimension();

    let glwe_params = fpksk_list.glwe_params;
    debug_assert_eq!(glwe_params, ggsw_out.out_glwe_params());
    debug_assert_eq!(glwe_params, ggsw_out.in_glwe_params());

    debug_assert_eq!(
        fpksk_input_lwe_key_dimension,
        fourier_bsk_output_lwe_dimension,
    );

    debug_assert_eq!(glwe_params.dimension + 1, fpksk_list.count);

    // Output for every bootstrapping
    let (mut lwe_out_bs_buffer_data, mut stack) = stack.make_aligned_with(
        fourier_bsk_output_lwe_dimension + 1,
        CACHELINE_ALIGN,
        |_| 0_u64,
    );
    let mut lwe_out_bs_buffer = LweCiphertext::new(
        &mut *lwe_out_bs_buffer_data,
        fourier_bsk_output_lwe_dimension,
    );

    // Output for every pfksk that that come from the output GGSW
    let mut out_pfksk_buffer_iter = ggsw_out
        .into_data()
        .chunks_exact_mut((glwe_params.dimension + 1) * glwe_params.polynomial_size)
        .map(|data| GlweCiphertext::new(data, glwe_params));

    for decomposition_level in 1..=level_cbs {
        homomorphic_shift_boolean(
            fourier_bsk,
            lwe_out_bs_buffer.as_mut_view(),
            lwe_in,
            decomposition_level,
            base_log_cbs,
            delta_log,
            fft,
            stack.rb_mut(),
        );

        for pfksk in fpksk_list.into_ppksk_key() {
            let glwe_out = out_pfksk_buffer_iter.next().unwrap();
            pfksk.private_functional_keyswitch_ciphertext(glwe_out, lwe_out_bs_buffer.as_view());
        }
    }
}

pub fn homomorphic_shift_boolean_scratch(
    lwe_in_size: usize,
    glwe_params: GlweParams,
    fft: FftView<'_>,
) -> Result<StackReq, SizeOverflow> {
    let align = CACHELINE_ALIGN;
    StackReq::try_new_aligned::<u64>(lwe_in_size, align)?
        .try_and(StackReq::try_new_aligned::<u64>(
            (glwe_params.dimension + 1) * glwe_params.polynomial_size,
            align,
        )?)?
        .try_and(BootstrapKey::bootstrap_scratch(glwe_params, fft)?)
}

/// Homomorphic shift for LWE without padding bit
///
/// Starts by shifting the message bit at bit #delta_log to the padding bit and then shifts it to
/// the right by base_log * level.
pub fn homomorphic_shift_boolean(
    fourier_bsk: BootstrapKey<&[f64]>,
    mut lwe_out: LweCiphertext<&mut [u64]>,
    lwe_in: LweCiphertext<&[u64]>,
    level_cbs: usize,
    base_log_cbs: usize,
    delta_log: usize,
    fft: FftView<'_>,
    stack: DynStack<'_>,
) {
    let ciphertext_n_bits = u64::BITS;
    let lwe_in_size = lwe_in.lwe_dimension + 1;
    let polynomial_size = fourier_bsk.glwe_params.polynomial_size;

    let (mut lwe_left_shift_buffer_data, stack) =
        stack.make_aligned_with(lwe_in_size, CACHELINE_ALIGN, |_| 0_u64);
    let mut lwe_left_shift_buffer =
        LweCiphertext::new(&mut *lwe_left_shift_buffer_data, lwe_in.lwe_dimension);
    // Shift message LSB on padding bit, at this point we expect to have messages with only 1 bit
    // of information

    let shift = 1 << (ciphertext_n_bits - delta_log as u32 - 1);

    debug_assert_eq!(shift, 1);

    for (a, b) in zip_eq(
        lwe_left_shift_buffer.as_mut_view().into_data(),
        lwe_in.into_data(),
    ) {
        *a = b.wrapping_mul(shift);
    }

    // Add q/4 to center the error while computing a negacyclic LUT
    let shift_buffer_body = lwe_left_shift_buffer
        .as_mut_view()
        .into_data()
        .last_mut()
        .unwrap();
    *shift_buffer_body = shift_buffer_body.wrapping_add(1_u64 << (ciphertext_n_bits - 2));

    let (mut pbs_accumulator_data, stack) = stack.make_aligned_with(
        polynomial_size * (fourier_bsk.glwe_params.dimension + 1),
        CACHELINE_ALIGN,
        |_| 0_u64,
    );
    let mut pbs_accumulator =
        GlweCiphertext::new(&mut *pbs_accumulator_data, fourier_bsk.glwe_params);

    // Fill lut (equivalent to trivial encryption as mask is 0s)
    // The LUT is filled with -alpha in each coefficient where
    // alpha = 2^{log(q) - 1 - base_log * level}
    let alpha = 1_u64 << (ciphertext_n_bits - 1 - base_log_cbs as u32 * level_cbs as u32);

    for body in pbs_accumulator.as_mut_view().into_body().into_data() {
        *body = alpha.wrapping_neg();
    }

    // Applying a negacyclic LUT on a ciphertext with one bit of message in the MSB and no bit
    // of padding
    fourier_bsk.bootstrap(
        lwe_out.as_mut_view(),
        lwe_left_shift_buffer.as_view(),
        pbs_accumulator.as_view(),
        fft,
        stack,
    );

    // Add alpha where alpha = 2^{log(q) - 1 - base_log * level}
    // To end up with an encryption of 0 if the message bit was 0 and 1 in the other case
    let out_body = lwe_out.as_mut_view().into_data().last_mut().unwrap();
    *out_body = out_body
        .wrapping_add(1_u64 << (ciphertext_n_bits - 1 - base_log_cbs as u32 * level_cbs as u32));
}

pub type FourierGgswCiphertextListView<'a> = FourierGgswCiphertextList<&'a [f64]>;
pub type FourierGgswCiphertextListMutView<'a> = FourierGgswCiphertextList<&'a mut [f64]>;
pub type GlweCiphertextListView<'a, Scalar> = GlweCiphertextList<&'a [Scalar]>;
pub type GlweCiphertextListMutView<'a, Scalar> = GlweCiphertextList<&'a mut [Scalar]>;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub struct GlweCiphertextList<C: Container> {
    pub data: C,
    pub count: usize,
    pub glwe_params: GlweParams,
}

#[derive(Debug, Clone)]
pub struct FourierGgswCiphertextList<C: Container<Item = f64>> {
    pub fourier: PolynomialList<C>,
    pub count: usize,
    pub glwe_params: GlweParams,
    pub decomp_params: DecompParams,
}

impl<C: Container> GlweCiphertextList<C> {
    pub fn new(data: C, count: usize, glwe_params: GlweParams) -> Self {
        debug_assert_eq!(
            data.len(),
            count * glwe_params.polynomial_size * (glwe_params.dimension + 1),
        );
        Self {
            data,
            count,
            glwe_params,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn as_view(&self) -> GlweCiphertextListView<'_, C::Item> {
        GlweCiphertextListView {
            data: self.data.as_ref(),
            count: self.count,
            glwe_params: self.glwe_params,
        }
    }

    pub fn as_mut_view(&mut self) -> GlweCiphertextListMutView<'_, C::Item>
    where
        C: AsMut<[C::Item]>,
    {
        GlweCiphertextListMutView {
            data: self.data.as_mut(),
            count: self.count,
            glwe_params: self.glwe_params,
        }
    }

    pub fn into_glwe_iter(self) -> impl DoubleEndedIterator<Item = GlweCiphertext<C>>
    where
        C: Split,
    {
        self.data
            .split_into(self.count)
            .map(move |slice| GlweCiphertext::new(slice, self.glwe_params))
    }
}

impl<C: Container<Item = f64>> FourierGgswCiphertextList<C> {
    pub fn new(
        data: C,
        count: usize,
        glwe_params: GlweParams,
        decomp_params: DecompParams,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            count
                * glwe_params.polynomial_size
                * (glwe_params.dimension + 1)
                * (glwe_params.dimension + 1)
                * decomp_params.level
        );

        Self {
            fourier: PolynomialList {
                data,
                polynomial_size: glwe_params.polynomial_size,
                count,
            },
            count,
            glwe_params,
            decomp_params,
        }
    }

    pub fn as_view(&self) -> FourierGgswCiphertextListView<'_> {
        let fourier = PolynomialList {
            data: self.fourier.data.as_ref(),
            polynomial_size: self.fourier.polynomial_size,
            count: self.count,
        };
        FourierGgswCiphertextListView {
            fourier,
            count: self.count,
            decomp_params: self.decomp_params,
            glwe_params: self.glwe_params,
        }
    }

    pub fn as_mut_view(&mut self) -> FourierGgswCiphertextListMutView<'_>
    where
        C: AsMut<[f64]>,
    {
        let fourier = PolynomialList {
            data: self.fourier.data.as_mut(),
            polynomial_size: self.fourier.polynomial_size,
            count: self.count,
        };
        FourierGgswCiphertextListMutView {
            fourier,
            count: self.count,
            decomp_params: self.decomp_params,
            glwe_params: self.glwe_params,
        }
    }

    pub fn into_ggsw_iter(self) -> impl DoubleEndedIterator<Item = GgswCiphertext<C>>
    where
        C: Split,
    {
        self.fourier.data.split_into(self.count).map(move |slice| {
            GgswCiphertext::new(
                slice,
                self.glwe_params.polynomial_size,
                self.glwe_params.dimension,
                self.glwe_params.dimension,
                self.decomp_params,
            )
        })
    }

    pub fn split_at(self, mid: usize) -> (Self, Self)
    where
        C: Split,
    {
        let glwe_dim = self.glwe_params.dimension;
        let polynomial_size = self.fourier.polynomial_size;

        let (left, right) = self.fourier.data.split_at(
            mid * polynomial_size * (glwe_dim + 1) * (glwe_dim + 1) * self.decomp_params.level,
        );
        (
            Self::new(left, mid, self.glwe_params, self.decomp_params),
            Self::new(
                right,
                self.count - mid,
                self.glwe_params,
                self.decomp_params,
            ),
        )
    }
}

pub fn cmux_tree_memory_optimized_scratch(
    glwe_params: GlweParams,
    nb_layer: usize,
    fft: FftView<'_>,
) -> Result<StackReq, SizeOverflow> {
    let t_scratch = StackReq::try_new_aligned::<u64>(
        (glwe_params.dimension + 1) * glwe_params.polynomial_size * nb_layer,
        CACHELINE_ALIGN,
    )?;

    StackReq::try_all_of([
        t_scratch,                             // t_0
        t_scratch,                             // t_1
        StackReq::try_new::<usize>(nb_layer)?, // t_fill
        t_scratch,                             // diff
        external_product_scratch(glwe_params, fft)?,
    ])
}

/// Performs a tree of cmux in a way that limits the total allocated memory to avoid issues for
/// bigger trees.
pub fn cmux_tree_memory_optimized(
    mut output_glwe: GlweCiphertext<&mut [u64]>,
    lut_per_layer: PolynomialList<&[u64]>,
    ggsw_list: FourierGgswCiphertextListView<'_>,
    fft: FftView<'_>,
    stack: DynStack<'_>,
) {
    debug_assert_eq!(lut_per_layer.count, 1 << ggsw_list.count);
    debug_assert!(ggsw_list.count > 0);

    let glwe_dim = ggsw_list.glwe_params.dimension;
    let polynomial_size = ggsw_list.glwe_params.polynomial_size;
    let nb_layer = ggsw_list.count;

    debug_assert!(stack.can_hold(
        cmux_tree_memory_optimized_scratch(output_glwe.glwe_params, nb_layer, fft).unwrap()
    ));

    // These are accumulator that will be used to propagate the result from layer to layer
    // At index 0 you have the lut that will be loaded, and then the result for each layer gets
    // computed at the next index, last layer result gets stored in `result`.
    // This allow to use memory space in C * nb_layer instead of C' * 2 ^ nb_layer
    let (mut t_0_data, stack) = stack.make_aligned_with(
        polynomial_size * (glwe_dim + 1) * nb_layer,
        CACHELINE_ALIGN,
        |_| 0_u64,
    );
    let (mut t_1_data, stack) = stack.make_aligned_with(
        polynomial_size * (glwe_dim + 1) * nb_layer,
        CACHELINE_ALIGN,
        |_| 0_u64,
    );

    let mut t_0 = GlweCiphertextList::new(t_0_data.as_mut(), nb_layer, ggsw_list.glwe_params);
    let mut t_1 = GlweCiphertextList::new(t_1_data.as_mut(), nb_layer, ggsw_list.glwe_params);

    let (mut t_fill, mut stack) = stack.make_with(nb_layer, |_| 0_usize);

    let mut lut_polynomial_iter = lut_per_layer.iter_polynomial();
    loop {
        let even = lut_polynomial_iter.next();
        let odd = lut_polynomial_iter.next();

        let (lut_2i, lut_2i_plus_1) = match (even, odd) {
            (Some(even), Some(odd)) => (even, odd),
            _ => break,
        };

        let mut t_iter = zip_eq(
            t_0.as_mut_view().into_glwe_iter(),
            t_1.as_mut_view().into_glwe_iter(),
        )
        .enumerate();

        let (mut j_counter, (mut t0_j, mut t1_j)) = t_iter.next().unwrap();

        t0_j.as_mut_view()
            .into_body()
            .into_data()
            .copy_from_slice(lut_2i.into_data());

        t1_j.as_mut_view()
            .into_body()
            .into_data()
            .copy_from_slice(lut_2i_plus_1.into_data());

        t_fill[0] = 2;

        for (j, ggsw) in ggsw_list.as_view().into_ggsw_iter().rev().enumerate() {
            if t_fill[j] == 2 {
                let (diff_data, stack) = stack.rb_mut().collect_aligned(
                    CACHELINE_ALIGN,
                    zip_eq(t1_j.as_view().into_data(), t0_j.as_view().data)
                        .map(|(a, b)| a.wrapping_sub(*b)),
                );
                let diff = GlweCiphertext::new(&*diff_data, ggsw_list.glwe_params);

                if j != nb_layer - 1 {
                    let (j_counter_plus_1, (mut t_0_j_plus_1, mut t_1_j_plus_1)) =
                        t_iter.next().unwrap();

                    debug_assert_eq!(j_counter, j);
                    debug_assert_eq!(j_counter_plus_1, j + 1);

                    let mut output = if t_fill[j + 1] == 0 {
                        t_0_j_plus_1.as_mut_view()
                    } else {
                        t_1_j_plus_1.as_mut_view()
                    };

                    output
                        .as_mut_view()
                        .into_data()
                        .copy_from_slice(t0_j.as_view().data);
                    external_product(output, ggsw, diff, fft, stack);
                    t_fill[j + 1] += 1;
                    t_fill[j] = 0;

                    drop(diff_data);

                    (j_counter, t0_j, t1_j) = (j_counter_plus_1, t_0_j_plus_1, t_1_j_plus_1);
                } else {
                    let mut output = output_glwe.as_mut_view();
                    output
                        .as_mut_view()
                        .into_data()
                        .copy_from_slice(t0_j.as_view().data);
                    external_product(output, ggsw, diff, fft, stack);
                }
            } else {
                break;
            }
        }
    }
}

pub fn circuit_bootstrap_boolean_vertical_packing_scratch(
    lwe_list_in_count: usize,
    lwe_list_out_count: usize,
    lwe_in_size: usize,
    big_lut_polynomial_count: usize,
    bsk_output_lwe_size: usize,
    fpksk_output_polynomial_size: usize,
    glwe_dimension: usize,
    level_cbs: usize,
    fft: FftView<'_>,
) -> Result<StackReq, SizeOverflow> {
    // We deduce the number of luts in the vec_lut from the number of cipherxtexts in lwe_list_out
    let number_of_luts = lwe_list_out_count;
    let small_lut_size = big_lut_polynomial_count / number_of_luts;

    StackReq::try_all_of([
        StackReq::try_new_aligned::<f64>(
            lwe_list_in_count
                * fpksk_output_polynomial_size
                * (glwe_dimension + 1)
                * (glwe_dimension + 1)
                * level_cbs,
            CACHELINE_ALIGN,
        )?,
        StackReq::try_new_aligned::<u64>(
            fpksk_output_polynomial_size * (glwe_dimension + 1) * (glwe_dimension + 1) * level_cbs,
            CACHELINE_ALIGN,
        )?,
        StackReq::try_any_of([
            circuit_bootstrap_boolean_scratch(
                lwe_in_size,
                bsk_output_lwe_size,
                GlweParams {
                    dimension: glwe_dimension,
                    polynomial_size: fpksk_output_polynomial_size,
                },
                fft,
            )?,
            fft.forward_scratch()?,
            vertical_packing_scratch(
                GlweParams {
                    dimension: glwe_dimension,
                    polynomial_size: fpksk_output_polynomial_size,
                },
                small_lut_size,
                lwe_list_in_count,
                fft,
            )?,
        ])?,
    ])
}

/// Perform a circuit bootstrap followed by a vertical packing on ciphertexts encrypting boolean
/// messages.
///
/// The circuit bootstrapping uses the private functional packing key switch.
///
/// This is supposed to be used only with boolean (1 bit of message) LWE ciphertexts.
pub fn circuit_bootstrap_boolean_vertical_packing(
    luts: PolynomialList<&[u64]>,
    fourier_bsk: BootstrapKey<&[f64]>,
    mut lwe_list_out: LweCiphertextList<&mut [u64]>,
    lwe_list_in: LweCiphertextList<&[u64]>,
    fpksk_list: PackingKeyswitchKeyList<&[u64]>,
    cbs_dp: DecompParams,
    fft: FftView<'_>,
    stack: DynStack<'_>,
) {
    debug_assert!(stack.can_hold(
        circuit_bootstrap_boolean_vertical_packing_scratch(
            lwe_list_in.count,
            lwe_list_out.count,
            lwe_list_in.lwe_dimension + 1,
            luts.count,
            fourier_bsk.output_lwe_dimension() + 1,
            fpksk_list.glwe_params.polynomial_size,
            fourier_bsk.glwe_params.dimension + 1,
            cbs_dp.level,
            fft
        )
        .unwrap()
    ));
    debug_assert_ne!(lwe_list_in.count, 0);
    debug_assert_eq!(
        lwe_list_out.lwe_dimension,
        fourier_bsk.output_lwe_dimension(),
    );

    let glwe_dim = fpksk_list.glwe_params.dimension;
    let (mut ggsw_list_data, stack) = stack.make_aligned_with(
        lwe_list_in.count
            * fpksk_list.glwe_params.polynomial_size
            * (glwe_dim + 1)
            * (glwe_dim + 1)
            * cbs_dp.level,
        CACHELINE_ALIGN,
        |_| f64::default(),
    );
    let (mut ggsw_res_data, mut stack) = stack.make_aligned_with(
        fpksk_list.glwe_params.polynomial_size * (glwe_dim + 1) * (glwe_dim + 1) * cbs_dp.level,
        CACHELINE_ALIGN,
        |_| 0_u64,
    );

    let mut ggsw_list = FourierGgswCiphertextList::new(
        &mut *ggsw_list_data,
        lwe_list_in.count,
        fpksk_list.glwe_params,
        cbs_dp,
    );

    let mut ggsw_res = GgswCiphertext::new(
        &mut *ggsw_res_data,
        fpksk_list.glwe_params.polynomial_size,
        fpksk_list.glwe_params.dimension,
        fpksk_list.glwe_params.dimension,
        cbs_dp,
    );

    for (lwe_in, ggsw) in zip_eq(
        lwe_list_in.ciphertext_iter(),
        ggsw_list.as_mut_view().into_ggsw_iter(),
    ) {
        circuit_bootstrap_boolean(
            fourier_bsk,
            lwe_in,
            ggsw_res.as_mut_view(),
            u64::BITS as usize - 1,
            fpksk_list,
            fft,
            stack.rb_mut(),
        );

        ggsw.fill_with_forward_fourier(ggsw_res.as_view(), fft, stack.rb_mut());
    }

    // We deduce the number of luts in the vec_lut from the number of cipherxtexts in lwe_list_out
    // debug_assert_eq!(lwe_list_out.count, small_lut_count);

    debug_assert_eq!(lwe_list_out.count, luts.count);

    for (lut, lwe_out) in zip_eq(luts.iter_polynomial(), lwe_list_out.ciphertext_iter_mut()) {
        vertical_packing(lut, lwe_out, ggsw_list.as_view(), fft, stack.rb_mut());
    }
}

fn print_ct(ct: u64) {
    print!("{}", (((ct >> 53) + 1) >> 1) % (1 << 10));
}

pub fn vertical_packing_scratch(
    glwe_params: GlweParams,
    lut_polynomial_count: usize,
    ggsw_list_count: usize,
    fft: FftView<'_>,
) -> Result<StackReq, SizeOverflow> {
    let bits = core::mem::size_of::<u64>() * 8;

    // Get the base 2 logarithm (rounded down) of the number of polynomials in the list i.e. if
    // there is one polynomial, the number will be 0
    let log_lut_number: usize = bits - 1 - lut_polynomial_count.leading_zeros() as usize;

    let log_number_of_luts_for_cmux_tree = if log_lut_number > ggsw_list_count {
        // this means that we dont have enough GGSW to perform the CMux tree, we can only do the
        // Blind rotation
        0
    } else {
        log_lut_number
    };

    StackReq::try_all_of([
        // cmux_tree_lut_res
        StackReq::try_new_aligned::<u64>(
            (glwe_params.dimension + 1) * glwe_params.polynomial_size,
            CACHELINE_ALIGN,
        )?,
        StackReq::try_any_of([
            blind_rotate_scratch(glwe_params, fft)?,
            cmux_tree_memory_optimized_scratch(glwe_params, log_number_of_luts_for_cmux_tree, fft)?,
        ])?,
    ])
}

fn log2(a: usize) -> usize {
    let result = u64::BITS as usize - 1 - a.leading_zeros() as usize;

    debug_assert_eq!(a, 1 << result);

    result
}

// GGSW ciphertexts are stored from the msb (vec_ggsw[0]) to the lsb (vec_ggsw[last])
pub fn vertical_packing(
    lut: Polynomial<&[u64]>,
    lwe_out: LweCiphertext<&mut [u64]>,
    ggsw_list: FourierGgswCiphertextListView,
    fft: FftView,
    stack: DynStack<'_>,
) {
    let glwe_params = ggsw_list.glwe_params;
    let polynomial_size = glwe_params.polynomial_size;
    let glwe_dimension = glwe_params.dimension;

    debug_assert_eq!(lwe_out.lwe_dimension, polynomial_size * glwe_dimension);

    let log_lut_number = log2(lut.len());

    debug_assert_eq!(ggsw_list.count, log_lut_number);

    let log_poly_size = log2(polynomial_size);

    let (mut cmux_tree_lut_res_data, mut stack) = stack.make_aligned_with(
        polynomial_size * (glwe_dimension + 1),
        CACHELINE_ALIGN,
        |_| 0_u64,
    );
    let mut cmux_tree_lut_res = GlweCiphertext::new(&mut *cmux_tree_lut_res_data, glwe_params);

    let br_ggsw = match log_lut_number.cmp(&log_poly_size) {
        Ordering::Less => {
            cmux_tree_lut_res
                .as_mut_view()
                .into_data()
                .fill_with(|| 0_u64);
            cmux_tree_lut_res.as_mut_view().into_body().into_data()[0..lut.len()]
                .copy_from_slice(lut.into_data());
            ggsw_list
        }
        Ordering::Equal => {
            cmux_tree_lut_res
                .as_mut_view()
                .into_data()
                .fill_with(|| 0_u64);
            cmux_tree_lut_res
                .as_mut_view()
                .into_body()
                .into_data()
                .copy_from_slice(lut.into_data());
            ggsw_list
        }
        Ordering::Greater => {
            let log_number_of_luts_for_cmux_tree = log_lut_number - log_poly_size;

            // split the vec of GGSW in two, the msb GGSW is for the CMux tree and the lsb GGSW is
            // for the last blind rotation.
            let (cmux_ggsw, br_ggsw) = ggsw_list.split_at(log_number_of_luts_for_cmux_tree);
            debug_assert_eq!(br_ggsw.count, log_poly_size);

            let small_luts = PolynomialList::new(
                lut.into_data(),
                polynomial_size,
                1 << (log_lut_number - log_poly_size),
            );

            cmux_tree_memory_optimized(
                cmux_tree_lut_res.as_mut_view(),
                small_luts,
                cmux_ggsw,
                fft,
                stack.rb_mut(),
            );

            br_ggsw
        }
    };

    blind_rotate(
        cmux_tree_lut_res.as_mut_view(),
        br_ggsw,
        fft,
        stack.rb_mut(),
    );

    // sample extract of the RLWE of the Vertical packing
    cmux_tree_lut_res.as_view().sample_extract(lwe_out, 0);
}

pub fn blind_rotate_scratch(
    glwe_params: GlweParams,
    fft: FftView<'_>,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        StackReq::try_new_aligned::<u64>(
            (glwe_params.dimension + 1) * glwe_params.polynomial_size,
            CACHELINE_ALIGN,
        )?,
        cmux_scratch(glwe_params, fft)?,
    ])
}

pub fn blind_rotate(
    mut lut: GlweCiphertext<&mut [u64]>,
    ggsw_list: FourierGgswCiphertextListView<'_>,
    fft: FftView<'_>,
    mut stack: DynStack<'_>,
) {
    let mut monomial_degree = 1;

    for ggsw in ggsw_list.into_ggsw_iter().rev() {
        let ct_0 = lut.as_mut_view();
        let (mut ct1_data, stack) = stack
            .rb_mut()
            .collect_aligned(CACHELINE_ALIGN, ct_0.as_view().into_data().iter().copied());
        let mut ct_1 = GlweCiphertext::new(&mut *ct1_data, ct_0.glwe_params);

        for a in ct_1.as_mut_view().into_polynomial_list().iter_polynomial() {
            update_with_wrapping_unit_monomial_div(a, monomial_degree);
        }
        monomial_degree <<= 1;
        cmux(ct_0, ct_1, ggsw, fft, stack);
    }
}
