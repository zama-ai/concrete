use crate::implementation::convert::x86::*;
use core::mem::MaybeUninit;
use pulp::{as_arrays, as_arrays_mut};

/// # Postconditions
///
/// this function leaves all the elements of `output_fourier` in an initialized state.
///
/// # Safety
///
///  - if `is_output_uninit` is false, `output_fourier` must not hold any uninitialized values.
#[cfg(feature = "nightly")]
unsafe fn update_with_fmadd_avx512(
    simd: Avx512,
    output_fourier: &mut [MaybeUninit<f64>],
    ggsw_polynomial: &[f64],
    fourier: &[f64],
    is_output_uninit: bool,
) {
    use crate::implementation::assume_init_mut;

    let n = output_fourier.len();

    debug_assert_eq!(n, ggsw_polynomial.len());
    debug_assert_eq!(n, fourier.len());
    debug_assert_eq!(n % 8, 0);

    // 8×f64 => 4×c64
    let (ggsw_polynomial, _) = as_arrays::<8, _>(ggsw_polynomial);
    let (fourier, _) = as_arrays::<8, _>(fourier);

    simd.vectorize(|| {
        let simd = simd.avx512f;
        if is_output_uninit {
            let (output_fourier, _) = as_arrays_mut::<8, _>(output_fourier);
            for (out, lhs, rhs) in izip!(output_fourier, ggsw_polynomial, fourier) {
                let ab = simd_cast(*lhs);
                let xy = simd_cast(*rhs);
                let aa = simd._mm512_unpacklo_pd(ab, ab);
                let bb = simd._mm512_unpackhi_pd(ab, ab);
                let yx = simd._mm512_permute_pd::<0b01010101>(xy);
                *out = simd_cast(simd._mm512_fmaddsub_pd(aa, xy, simd._mm512_mul_pd(bb, yx)));
            }
        } else {
            let (output_fourier, _) =
                as_arrays_mut::<8, _>(unsafe { assume_init_mut(output_fourier) });
            for (out, lhs, rhs) in izip!(output_fourier, ggsw_polynomial, fourier) {
                let ab = simd_cast(*lhs);
                let xy = simd_cast(*rhs);
                let aa = simd._mm512_unpacklo_pd(ab, ab);
                let bb = simd._mm512_unpackhi_pd(ab, ab);
                let yx = simd._mm512_permute_pd::<0b01010101>(xy);
                *out = simd_cast(simd._mm512_fmaddsub_pd(
                    aa,
                    xy,
                    simd._mm512_fmaddsub_pd(bb, yx, simd_cast(*out)),
                ));
            }
        }
    });
}

/// # Postconditions
///
/// this function leaves all the elements of `output_fourier` in an initialized state.
///
/// # Safety
///
///  - if `is_output_uninit` is false, `output_fourier` must not hold any uninitialized values.
unsafe fn update_with_fmadd_fma(
    simd: FusedMulAdd,
    output_fourier: &mut [MaybeUninit<f64>],
    ggsw_polynomial: &[f64],
    fourier: &[f64],
    is_output_uninit: bool,
) {
    use crate::implementation::assume_init_mut;

    let n = output_fourier.len();

    debug_assert_eq!(n, ggsw_polynomial.len());
    debug_assert_eq!(n, fourier.len());
    debug_assert_eq!(n % 4, 0);

    // 8×f64 => 4×c64
    let (ggsw_polynomial, _) = as_arrays::<4, _>(ggsw_polynomial);
    let (fourier, _) = as_arrays::<4, _>(fourier);

    simd.vectorize(|| {
        let FusedMulAdd { avx, fma, .. } = simd;
        if is_output_uninit {
            let (output_fourier, _) = as_arrays_mut::<4, _>(output_fourier);
            for (out, lhs, rhs) in izip!(output_fourier, ggsw_polynomial, fourier) {
                let ab = simd_cast(*lhs);
                let xy = simd_cast(*rhs);
                let aa = avx._mm256_unpacklo_pd(ab, ab);
                let bb = avx._mm256_unpackhi_pd(ab, ab);
                let yx = avx._mm256_permute_pd::<0b0101>(xy);
                *out = simd_cast(fma._mm256_fmaddsub_pd(aa, xy, avx._mm256_mul_pd(bb, yx)));
            }
        } else {
            let (output_fourier, _) =
                as_arrays_mut::<4, _>(unsafe { assume_init_mut(output_fourier) });
            for (out, lhs, rhs) in izip!(output_fourier, ggsw_polynomial, fourier) {
                let ab = simd_cast(*lhs);
                let xy = simd_cast(*rhs);
                let aa = avx._mm256_unpacklo_pd(ab, ab);
                let bb = avx._mm256_unpackhi_pd(ab, ab);
                let yx = avx._mm256_permute_pd::<0b0101>(xy);
                *out = simd_cast(fma._mm256_fmaddsub_pd(
                    aa,
                    xy,
                    fma._mm256_fmaddsub_pd(bb, yx, simd_cast(*out)),
                ));
            }
        }
    });
}

pub unsafe fn update_with_fmadd(
    output_fourier: &mut [MaybeUninit<f64>],
    ggsw_polynomial: &[f64],
    fourier: &[f64],
    is_output_uninit: bool,
) {
    #[cfg(feature = "nightly")]
    if let Some(simd) = Avx512::try_new() {
        return unsafe {
            update_with_fmadd_avx512(
                simd,
                output_fourier,
                ggsw_polynomial,
                fourier,
                is_output_uninit,
            )
        };
    }
    if let Some(simd) = FusedMulAdd::try_new() {
        return unsafe {
            update_with_fmadd_fma(
                simd,
                output_fourier,
                ggsw_polynomial,
                fourier,
                is_output_uninit,
            )
        };
    }

    unsafe {
        super::update_with_fmadd_scalar(output_fourier, ggsw_polynomial, fourier, is_output_uninit)
    }
}
