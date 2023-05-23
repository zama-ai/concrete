#![allow(clippy::too_many_arguments)]

use tfhe::core_crypto::fft_impl::fft64::math::fft::FftView;
use tfhe::core_crypto::prelude::*;
//use tfhe::core_crypto::fft_impl::fft64::crypto::ggsw::FourierGgswCiphertextListView;

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::{SizeOverflow, StackReq};

//use crate::implementation::zip_eq;

pub fn cmux_tree_memory_optimized_scratch(
    glwe_size: GlweSize,
    polynomial_size: PolynomialSize,
    nb_layer: usize,
    fft: FftView<'_>,
) -> Result<StackReq, SizeOverflow> {
    let t_scratch = StackReq::try_new_aligned::<u64>(
        glwe_size.0 * polynomial_size.0 * nb_layer,
        CACHELINE_ALIGN,
    )?;

    StackReq::try_all_of([
        t_scratch,                             // t_0
        t_scratch,                             // t_1
        StackReq::try_new::<usize>(nb_layer)?, // t_fill
        t_scratch,                             // diff
        add_external_product_assign_mem_optimized_requirement::<u64>(
            glwe_size,
            polynomial_size,
            fft,
        )?,
    ])
}

/// Performs a tree of cmux in a way that limits the total allocated memory to avoid issues for
/// bigger trees.
/*pub fn cmux_tree_memory_optimized(
    mut output_glwe: GlweCiphertext<&mut [u64]>,
    lut_per_layer: PolynomialList<&[u64]>,
    ggsw_list: FourierGgswCiphertextListView<'_>,
    fft: FftView<'_>,
    stack: PodStack<'_>,
) {
    debug_assert_eq!(lut_per_layer.polynomial_count().0, 1 << ggsw_list.count());
    debug_assert!(ggsw_list.count() > 0);

    let glwe_size = ggsw_list.glwe_size();
    let polynomial_size = ggsw_list.polynomial_size();
    let nb_layer = ggsw_list.count();

    debug_assert!(stack.can_hold(
        cmux_tree_memory_optimized_scratch(glwe_size, polynomial_size, nb_layer, fft).unwrap()
    ));

    // These are accumulator that will be used to propagate the result from layer to layer
    // At index 0 you have the lut that will be loaded, and then the result for each layer gets
    // computed at the next index, last layer result gets stored in `result`.
    // This allow to use memory space in C * nb_layer instead of C' * 2 ^ nb_layer
    let (mut t_0_data, stack) = stack.make_aligned_with(
        polynomial_size.0 * glwe_size.0 * nb_layer,
        CACHELINE_ALIGN,
        |_| 0_u64,
    );
    let (mut t_1_data, stack) = stack.make_aligned_with(
        polynomial_size.0 * glwe_size.0 * nb_layer,
        CACHELINE_ALIGN,
        |_| 0_u64,
    );

    let mut t_0 = GlweCiphertextList::from_container(core::slice::from_raw_parts_mut(
        t_0_data.as_mut(), polynomial_size.0 * glwe_size.0 * nb_layer),
    glwe_size, polynomial_size, CiphertextModulus::new_native());
    let mut t_1 = GlweCiphertextList::new(t_1_data.as_mut(), nb_layer, ggsw_list.glwe_params);

    let (mut t_fill, mut stack) = stack.make_with(nb_layer, |_| 0_usize);

    let mut lut_polynomial_iter = lut_per_layer.iter();
    loop {
        let even = lut_polynomial_iter.next();
        let odd = lut_polynomial_iter.next();

        let (lut_2i, lut_2i_plus_1) = match (even, odd) {
            (Some(even), Some(odd)) => (even, odd),
            _ => break,
        };

        let mut t_iter = zip_eq(
            t_0.iter_mut().into_glwe_iter(),
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
                    add_external_product_assign_mem_optimized(&mut output, ggsw, diff, fft, stack);
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
                    add_external_product_assign_mem_optimized(&mut output, ggsw, diff, fft, stack);
                }
            } else {
                break;
            }
        }
    }
}*/

fn print_ct(ct: u64) {
    print!("{}", (((ct >> 53) + 1) >> 1) % (1 << 10));
}

fn log2(a: usize) -> usize {
    let result = u64::BITS as usize - 1 - a.leading_zeros() as usize;

    debug_assert_eq!(a, 1 << result);

    result
}
