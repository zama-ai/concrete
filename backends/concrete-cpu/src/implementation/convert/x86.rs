#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::mem::MaybeUninit;

use pulp::{as_arrays, as_arrays_mut, cast, simd_type};

use crate::implementation::fft::Twisties;

simd_type! {
    pub struct FusedMulAdd {
        pub sse2: "sse2",
        pub avx: "avx",
        pub avx2: "avx2",
        pub fma: "fma"
    }

    #[cfg(feature = "nightly")]
    pub struct Avx512 {
        pub avx512f: "avx512f",
        pub avx512dq: "avx512dq",
    }
}

pub unsafe trait CastInto<T: Copy>: Copy {
    #[inline(always)]
    fn transmute(self) -> T {
        debug_assert_eq!(core::mem::size_of::<T>(), core::mem::size_of::<Self>());
        unsafe { core::mem::transmute_copy(&self) }
    }
}

// anything can be cast into MaybeUninit
unsafe impl<const N: usize, T: Copy, U: Copy> CastInto<[MaybeUninit<T>; N]> for U {}

unsafe impl CastInto<[f64; 4]> for __m256d {}
unsafe impl CastInto<__m256d> for [f64; 4] {}

unsafe impl CastInto<[u64; 4]> for __m256i {}
unsafe impl CastInto<__m256i> for [u64; 4] {}

unsafe impl CastInto<[i64; 4]> for __m256i {}
unsafe impl CastInto<__m256i> for [i64; 4] {}

#[cfg(feature = "nightly")]
mod nightly_impls {
    use super::*;
    unsafe impl CastInto<[f64; 8]> for __m512d {}
    unsafe impl CastInto<__m512d> for [f64; 8] {}

    unsafe impl CastInto<[u64; 8]> for __m512i {}
    unsafe impl CastInto<__m512i> for [u64; 8] {}

    unsafe impl CastInto<[i64; 8]> for __m512i {}
    unsafe impl CastInto<__m512i> for [i64; 8] {}
}

#[inline(always)]
pub fn simd_cast<T: CastInto<U>, U: Copy>(t: T) -> U {
    t.transmute()
}

/// Converts a vector of f64 values to a vector of i64 values.
/// See `f64_to_i64_bit_twiddles` in `fft/tests.rs` for the scalar version.
#[inline(always)]
fn mm256_cvtpd_epi64(simd: FusedMulAdd, x: __m256d) -> __m256i {
    let FusedMulAdd { avx, avx2, .. } = simd;

    // reinterpret the bits as u64 values
    let bits = avx._mm256_castpd_si256(x);
    // mask that covers the first 52 bits
    let mantissa_mask = avx._mm256_set1_epi64x(0xFFFFFFFFFFFFF_u64 as i64);
    // mask that covers the 52nd bit
    let explicit_mantissa_bit = avx._mm256_set1_epi64x(0x10000000000000_u64 as i64);
    // mask that covers the first 11 bits
    let exp_mask = avx._mm256_set1_epi64x(0x7FF_u64 as i64);

    // extract the first 52 bits and add the implicit bit
    let mantissa = avx2._mm256_or_si256(
        avx2._mm256_and_si256(bits, mantissa_mask),
        explicit_mantissa_bit,
    );

    // extract the 52nd to 63rd (excluded) bits for the biased exponent
    let biased_exp = avx2._mm256_and_si256(avx2._mm256_srli_epi64::<52>(bits), exp_mask);

    // extract the 63rd sign bit
    let sign_is_negative_mask = avx2._mm256_sub_epi64(
        avx._mm256_setzero_si256(),
        avx2._mm256_srli_epi64::<63>(bits),
    );

    // we need to shift the mantissa by some value that may be negative, so we first shift
    // it to the left by the maximum amount, then shift it to the right by our
    // value plus the offset we just shifted by
    //
    // the 52nd bit is set to 1, so we shift to the left by 11 so the 63rd (last) bit is
    // set.
    let mantissa_lshift = avx2._mm256_slli_epi64::<11>(mantissa);

    // shift to the right and apply the exponent bias
    let mantissa_shift = avx2._mm256_srlv_epi64(
        mantissa_lshift,
        avx2._mm256_sub_epi64(avx._mm256_set1_epi64x(1086), biased_exp),
    );

    // if the sign bit is unset, we keep our result
    let value_if_positive = mantissa_shift;
    // otherwise, we negate it
    let value_if_negative = avx2._mm256_sub_epi64(avx._mm256_setzero_si256(), value_if_positive);

    // if the biased exponent is all zeros, we have a subnormal value (or zero)

    // if it is not subnormal, we keep our results
    let value_if_non_subnormal =
        avx2._mm256_blendv_epi8(value_if_positive, value_if_negative, sign_is_negative_mask);

    // if it is subnormal, the conversion to i64 (rounding towards zero) returns zero
    let value_if_subnormal = avx._mm256_setzero_si256();

    // compare the biased exponent to a zero value
    let is_subnormal = avx2._mm256_cmpeq_epi64(biased_exp, avx._mm256_setzero_si256());

    // choose the result depending on subnormalness
    avx2._mm256_blendv_epi8(value_if_non_subnormal, value_if_subnormal, is_subnormal)
}

/// Converts a vector of f64 values to a vector of i64 values.
/// See `f64_to_i64_bit_twiddles` in `fft/tests.rs` for the scalar version.
#[cfg(feature = "nightly")]
#[inline(always)]
fn mm512_cvtpd_epi64(simd: Avx512, x: __m512d) -> __m512i {
    let simd = simd.avx512f;

    // reinterpret the bits as u64 values
    let bits = simd._mm512_castpd_si512(x);
    // mask that covers the first 52 bits
    let mantissa_mask = simd._mm512_set1_epi64(0xFFFFFFFFFFFFF_u64 as i64);
    // mask that covers the 53rd bit
    let explicit_mantissa_bit = simd._mm512_set1_epi64(0x10000000000000_u64 as i64);
    // mask that covers the first 11 bits
    let exp_mask = simd._mm512_set1_epi64(0x7FF_u64 as i64);

    // extract the first 52 bits and add the implicit bit
    let mantissa = simd._mm512_or_si512(
        simd._mm512_and_si512(bits, mantissa_mask),
        explicit_mantissa_bit,
    );

    // extract the 52nd to 63rd (excluded) bits for the biased exponent
    let biased_exp = simd._mm512_and_si512(simd._mm512_srli_epi64::<52>(bits), exp_mask);

    // extract the 63rd sign bit
    let sign_is_negative_mask = simd._mm512_cmpneq_epi64_mask(
        simd._mm512_srli_epi64::<63>(bits),
        simd._mm512_set1_epi64(1),
    );

    // we need to shift the mantissa by some value that may be negative, so we first shift it to
    // the left by the maximum amount, then shift it to the right by our value plus the offset we
    // just shifted by
    //
    // the 53rd bit is set to 1, so we shift to the left by 10 so the 63rd (last) bit is set.
    let mantissa_lshift = simd._mm512_slli_epi64::<11>(mantissa);

    // shift to the right and apply the exponent bias
    let mantissa_shift = simd._mm512_srlv_epi64(
        mantissa_lshift,
        simd._mm512_sub_epi64(simd._mm512_set1_epi64(1086), biased_exp),
    );

    // if the sign bit is unset, we keep our result
    let value_if_positive = mantissa_shift;
    // otherwise, we negate it
    let value_if_negative = simd._mm512_sub_epi64(simd._mm512_setzero_si512(), value_if_positive);

    // if the biased exponent is all zeros, we have a subnormal value (or zero)

    // if it is not subnormal, we keep our results
    let value_if_non_subnormal =
        simd._mm512_mask_blend_epi64(sign_is_negative_mask, value_if_positive, value_if_negative);

    // if it is subnormal, the conversion to i64 (rounding towards zero) returns zero
    let value_if_subnormal = simd._mm512_setzero_si512();

    // compare the biased exponent to a zero value
    let is_subnormal = simd._mm512_cmpeq_epi64_mask(biased_exp, simd._mm512_setzero_si512());

    // choose the result depending on subnormalness
    simd._mm512_mask_blend_epi64(is_subnormal, value_if_non_subnormal, value_if_subnormal)
}

/// Converts a vector of i64 values to a vector of f64 values. Not sure how it works.
/// Ported from <https://stackoverflow.com/a/41148578>.
#[inline(always)]
fn mm256_cvtepi64_pd(simd: FusedMulAdd, x: __m256i) -> __m256d {
    let FusedMulAdd { avx, avx2, .. } = simd;

    let mut x_hi = avx2._mm256_srai_epi32::<16>(x);
    x_hi = avx2._mm256_blend_epi16::<0x33>(x_hi, avx._mm256_setzero_si256());
    x_hi = avx2._mm256_add_epi64(
        x_hi,
        avx._mm256_castpd_si256(avx._mm256_set1_pd(442721857769029238784.0)), // 3*2^67
    );
    let x_lo = avx2._mm256_blend_epi16::<0x88>(
        x,
        avx._mm256_castpd_si256(avx._mm256_set1_pd(4503599627370496.0)),
    ); // 2^52

    let f = avx._mm256_sub_pd(
        avx._mm256_castsi256_pd(x_hi),
        avx._mm256_set1_pd(442726361368656609280.0), // 3*2^67 + 2^52
    );

    avx._mm256_add_pd(f, avx._mm256_castsi256_pd(x_lo))
}

/// Converts a vector of i64 values to a vector of f64 values.
#[cfg(feature = "nightly")]
#[inline(always)]
fn mm512_cvtepi64_pd(simd: Avx512, x: __m512i) -> __m512d {
    // hopefully this compiles to vcvtqq2pd
    simd.vectorize(
        #[inline(always)]
        || {
            let i64x8: [i64; 8] = simd_cast(x);
            let as_f64x8 = [
                i64x8[0] as f64,
                i64x8[1] as f64,
                i64x8[2] as f64,
                i64x8[3] as f64,
                i64x8[4] as f64,
                i64x8[5] as f64,
                i64x8[6] as f64,
                i64x8[7] as f64,
            ];
            simd_cast(as_f64x8)
        },
    )
}

#[cfg(feature = "nightly")]
fn convert_forward_integer_u64_avx512(
    simd: Avx512,
    out: &mut [MaybeUninit<f64>],
    in_re: &[u64],
    in_im: &[u64],
    twisties: Twisties<&[f64]>,
) {
    let n = in_re.len();
    debug_assert_eq!(n % 8, 0);

    debug_assert_eq!(2 * n, out.len());
    debug_assert_eq!(n, in_re.len());
    debug_assert_eq!(n, in_im.len());
    debug_assert_eq!(n, twisties.re.len());
    debug_assert_eq!(n, twisties.im.len());

    let (out, _) = as_arrays_mut::<16, _>(out);
    let (in_re, _) = as_arrays::<8, _>(in_re);
    let (in_im, _) = as_arrays::<8, _>(in_im);
    let (w_re, _) = as_arrays::<8, _>(twisties.re);
    let (w_im, _) = as_arrays::<8, _>(twisties.im);

    simd.vectorize(
        #[inline(always)]
        || {
            for (out, in_re, in_im, w_re, w_im) in izip!(
                out,
                in_re.iter().copied(),
                in_im.iter().copied(),
                w_re.iter().copied(),
                w_im.iter().copied(),
            ) {
                // convert to i64, then to f64
                // the intermediate conversion to i64 can reduce noise by up to 10 bits.
                let in_re = mm512_cvtepi64_pd(simd, simd_cast(in_re));
                let in_im = mm512_cvtepi64_pd(simd, simd_cast(in_im));

                let w_re = simd_cast(w_re);
                let w_im = simd_cast(w_im);

                let simd = simd.avx512f;

                // perform complex multiplication
                let out_re = simd._mm512_fmsub_pd(in_re, w_re, simd._mm512_mul_pd(in_im, w_im));
                let out_im = simd._mm512_fmadd_pd(in_re, w_im, simd._mm512_mul_pd(in_im, w_re));

                // we have
                // x0 x1 x2 x3 x4 x5 x6 x7
                // y0 y1 y2 y3 y4 y5 y6 y7
                //
                // we want
                // x0 y0 x1 y1 x2 y2 x3 y3
                // x4 y4 x5 y5 x6 y6 x7 y7

                // interleave real part and imaginary part
                {
                    let idx0 = simd._mm512_setr_epi64(
                        0b0000, 0b1000, 0b0001, 0b1001, 0b0010, 0b1010, 0b0011, 0b1011,
                    );
                    let idx1 = simd._mm512_setr_epi64(
                        0b0100, 0b1100, 0b0101, 0b1101, 0b0110, 0b1110, 0b0111, 0b1111,
                    );

                    let out0 = simd._mm512_permutex2var_pd(out_re, idx0, out_im);
                    let out1 = simd._mm512_permutex2var_pd(out_re, idx1, out_im);

                    // store c64 values
                    *out = simd_cast([out0, out1]);
                }
            }
        },
    );
}

fn convert_forward_integer_u64_fma(
    simd: FusedMulAdd,
    out: &mut [MaybeUninit<f64>],
    in_re: &[u64],
    in_im: &[u64],
    twisties: Twisties<&[f64]>,
) {
    let n = in_re.len();
    debug_assert_eq!(n % 4, 0);

    debug_assert_eq!(2 * n, out.len());
    debug_assert_eq!(n, in_re.len());
    debug_assert_eq!(n, in_im.len());
    debug_assert_eq!(n, twisties.re.len());
    debug_assert_eq!(n, twisties.im.len());

    let (out, _) = as_arrays_mut::<8, _>(out);
    let (in_re, _) = as_arrays::<4, _>(in_re);
    let (in_im, _) = as_arrays::<4, _>(in_im);
    let (w_re, _) = as_arrays::<4, _>(twisties.re);
    let (w_im, _) = as_arrays::<4, _>(twisties.im);

    simd.vectorize(
        #[inline(always)]
        move || {
            for (out, in_re, in_im, w_re, w_im) in izip!(
                out,
                in_re.iter().copied(),
                in_im.iter().copied(),
                w_re.iter().copied(),
                w_im.iter().copied(),
            ) {
                // convert to i64, then to f64
                // the intermediate conversion to i64 can reduce noise by up to 10 bits.
                let in_re = mm256_cvtepi64_pd(simd, cast(in_re));
                let in_im = mm256_cvtepi64_pd(simd, cast(in_im));

                let w_re = cast(w_re);
                let w_im = cast(w_im);

                let FusedMulAdd { avx, fma, .. } = simd;

                // perform complex multiplication
                let out_re = fma._mm256_fmsub_pd(in_re, w_re, avx._mm256_mul_pd(in_im, w_im));
                let out_im = fma._mm256_fmadd_pd(in_re, w_im, avx._mm256_mul_pd(in_im, w_re));

                // we have
                // x0 x1 x2 x3
                // y0 y1 y2 y3
                //
                // we want
                // x0 y0 x1 y1
                // x2 y2 x3 y3

                // interleave real part and imaginary part

                // unpacklo/unpackhi
                // x0 y0 x2 y2
                // x1 y1 x3 y3
                let lo = avx._mm256_unpacklo_pd(out_re, out_im);
                let hi = avx._mm256_unpackhi_pd(out_re, out_im);

                let out0 = avx._mm256_permute2f128_pd::<0b00100000>(lo, hi);
                let out1 = avx._mm256_permute2f128_pd::<0b00110001>(lo, hi);

                // store c64 values
                *out = simd_cast([out0, out1]);
            }
        },
    );
}

/// Performs common work for `u32` and `u64`, used by the backward torus transformation.
///
/// This deinterleaves two vectors of c64 values into two vectors of real part and imaginary part,
/// then rounds to the nearest integer.
#[cfg(feature = "nightly")]
#[inline(always)]
fn convert_torus_prologue_avx512f(
    simd: Avx512,
    normalization: __m512d,
    w_re: __m512d,
    w_im: __m512d,
    input0: __m512d,
    input1: __m512d,
    scaling: __m512d,
) -> (__m512d, __m512d) {
    let simd = simd.avx512f;

    let w_re = simd._mm512_mul_pd(normalization, w_re);
    let w_im = simd._mm512_mul_pd(normalization, w_im);

    // real indices
    let idx0 = simd._mm512_setr_epi64(
        0b0000, 0b0010, 0b0100, 0b0110, 0b1000, 0b1010, 0b1100, 0b1110,
    );
    // imaginary indices
    let idx1 = simd._mm512_setr_epi64(
        0b0001, 0b0011, 0b0101, 0b0111, 0b1001, 0b1011, 0b1101, 0b1111,
    );

    // re0 re1 re2 re3 re4 re5 re6 re7
    let inp_re = simd._mm512_permutex2var_pd(input0, idx0, input1);
    // im0 im1 im2 im3 im4 im5 im6 im7
    let inp_im = simd._mm512_permutex2var_pd(input0, idx1, input1);

    // perform complex multiplication with conj(w)
    let mul_re = simd._mm512_fmadd_pd(inp_re, w_re, simd._mm512_mul_pd(inp_im, w_im));
    let mul_im = simd._mm512_fnmadd_pd(inp_re, w_im, simd._mm512_mul_pd(inp_im, w_re));

    // round to nearest integer and suppress exceptions
    const ROUNDING: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    // get the fractional part (centered around zero) by subtracting rounded value
    let fract_re = simd._mm512_sub_pd(mul_re, simd._mm512_roundscale_pd::<ROUNDING>(mul_re));
    let fract_im = simd._mm512_sub_pd(mul_im, simd._mm512_roundscale_pd::<ROUNDING>(mul_im));
    // scale fractional part and round
    let fract_re = simd._mm512_roundscale_pd::<ROUNDING>(simd._mm512_mul_pd(scaling, fract_re));
    let fract_im = simd._mm512_roundscale_pd::<ROUNDING>(simd._mm512_mul_pd(scaling, fract_im));

    (fract_re, fract_im)
}

/// Performs common work for `u32` and `u64`, used by the backward torus transformation.
///
/// This deinterleaves two vectors of c64 values into two vectors of real part and imaginary part,
/// then rounds to the nearest integer.
#[inline(always)]
fn convert_torus_prologue_fma(
    simd: FusedMulAdd,
    normalization: __m256d,
    w_re: __m256d,
    w_im: __m256d,
    input0: __m256d,
    input1: __m256d,
    scaling: __m256d,
) -> (__m256d, __m256d) {
    let FusedMulAdd { avx, fma, sse2, .. } = simd;

    let w_re = avx._mm256_mul_pd(normalization, w_re);
    let w_im = avx._mm256_mul_pd(normalization, w_im);

    // re0 im0
    // re1 im1
    let [inp0, inp1] = cast::<__m256d, [__m128d; 2]>(input0);
    // re2 im2
    // re3 im3
    let [inp2, inp3] = cast::<__m256d, [__m128d; 2]>(input1);

    // re0 re1
    let inp_re01 = sse2._mm_unpacklo_pd(inp0, inp1);
    // im0 im1
    let inp_im01 = sse2._mm_unpackhi_pd(inp0, inp1);
    // re2 re3
    let inp_re23 = sse2._mm_unpacklo_pd(inp2, inp3);
    // im2 im3
    let inp_im23 = sse2._mm_unpackhi_pd(inp2, inp3);

    // re0 re1 re2 re3
    let inp_re = avx._mm256_insertf128_pd::<0b1>(avx._mm256_castpd128_pd256(inp_re01), inp_re23);
    // im0 im1 im2 im3
    let inp_im = avx._mm256_insertf128_pd::<0b1>(avx._mm256_castpd128_pd256(inp_im01), inp_im23);

    // perform complex multiplication with conj(w)
    let mul_re = fma._mm256_fmadd_pd(inp_re, w_re, avx._mm256_mul_pd(inp_im, w_im));
    let mul_im = fma._mm256_fnmadd_pd(inp_re, w_im, avx._mm256_mul_pd(inp_im, w_re));

    // round to nearest integer and suppress exceptions
    const ROUNDING: i32 = _MM_FROUND_NINT | _MM_FROUND_NO_EXC;

    // get the fractional part (centered around zero) by subtracting rounded value
    let fract_re = avx._mm256_sub_pd(mul_re, avx._mm256_round_pd::<ROUNDING>(mul_re));
    let fract_im = avx._mm256_sub_pd(mul_im, avx._mm256_round_pd::<ROUNDING>(mul_im));
    // scale fractional part and round
    let fract_re = avx._mm256_round_pd::<ROUNDING>(avx._mm256_mul_pd(scaling, fract_re));
    let fract_im = avx._mm256_round_pd::<ROUNDING>(avx._mm256_mul_pd(scaling, fract_im));

    (fract_re, fract_im)
}

#[cfg(feature = "nightly")]
fn convert_add_backward_torus_u64_avx512f(
    simd: Avx512,
    out_re: &mut [u64],
    out_im: &mut [u64],
    input: &[f64],
    twisties: Twisties<&[f64]>,
) {
    let n = out_re.len();
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(n, out_re.len());
    debug_assert_eq!(n, out_im.len());
    debug_assert_eq!(2 * n, input.len());
    debug_assert_eq!(n, twisties.re.len());
    debug_assert_eq!(n, twisties.im.len());

    let (out_re, _) = as_arrays_mut::<8, _>(out_re);
    let (out_im, _) = as_arrays_mut::<8, _>(out_im);
    let (inp, _) = as_arrays::<16, _>(input);
    let (w_re, _) = as_arrays::<8, _>(twisties.re);
    let (w_im, _) = as_arrays::<8, _>(twisties.im);

    simd.vectorize(
        #[inline(always)]
        || {
            let normalization = simd.avx512f._mm512_set1_pd(1.0 / n as f64);
            let scaling = simd.avx512f._mm512_set1_pd(2.0_f64.powi(u64::BITS as i32));
            for (out_re, out_im, inp, w_re, w_im) in izip!(
                out_re,
                out_im,
                inp.iter().copied(),
                w_re.iter().copied(),
                w_im.iter().copied(),
            ) {
                let [input0, input1]: [[f64; 8]; 2] = cast(inp);
                let (fract_re, fract_im) = convert_torus_prologue_avx512f(
                    simd,
                    normalization,
                    simd_cast(w_re),
                    simd_cast(w_im),
                    simd_cast(input0),
                    simd_cast(input1),
                    scaling,
                );

                // convert f64 to i64
                let fract_re = mm512_cvtpd_epi64(simd, fract_re);
                let fract_im = mm512_cvtpd_epi64(simd, fract_im);

                // add to input and store
                *out_re = simd_cast(simd.avx512f._mm512_add_epi64(fract_re, simd_cast(*out_re)));
                *out_im = simd_cast(simd.avx512f._mm512_add_epi64(fract_im, simd_cast(*out_im)));
            }
        },
    );
}

fn convert_add_backward_torus_u64_fma(
    simd: FusedMulAdd,
    out_re: &mut [u64],
    out_im: &mut [u64],
    input: &[f64],
    twisties: Twisties<&[f64]>,
) {
    let n = out_re.len();
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(n, out_re.len());
    debug_assert_eq!(n, out_im.len());
    debug_assert_eq!(2 * n, input.len());
    debug_assert_eq!(n, twisties.re.len());
    debug_assert_eq!(n, twisties.im.len());

    let (out_re, _) = as_arrays_mut::<4, _>(out_re);
    let (out_im, _) = as_arrays_mut::<4, _>(out_im);
    let (inp, _) = as_arrays::<8, _>(input);
    let (w_re, _) = as_arrays::<4, _>(twisties.re);
    let (w_im, _) = as_arrays::<4, _>(twisties.im);

    simd.vectorize(
        #[inline(always)]
        || {
            let normalization = simd.avx._mm256_set1_pd(1.0 / n as f64);
            let scaling = simd.avx._mm256_set1_pd(2.0_f64.powi(u64::BITS as i32));
            for (out_re, out_im, inp, w_re, w_im) in izip!(
                out_re,
                out_im,
                inp.iter().copied(),
                w_re.iter().copied(),
                w_im.iter().copied(),
            ) {
                let [input0, input1]: [[f64; 4]; 2] = cast(inp);
                let (fract_re, fract_im) = convert_torus_prologue_fma(
                    simd,
                    normalization,
                    simd_cast(w_re),
                    simd_cast(w_im),
                    simd_cast(input0),
                    simd_cast(input1),
                    scaling,
                );

                // convert f64 to i64
                let fract_re = mm256_cvtpd_epi64(simd, fract_re);
                let fract_im = mm256_cvtpd_epi64(simd, fract_im);

                // add to input and store
                *out_re = simd_cast(simd.avx2._mm256_add_epi64(fract_re, simd_cast(*out_re)));
                *out_im = simd_cast(simd.avx2._mm256_add_epi64(fract_im, simd_cast(*out_im)));
            }
        },
    );
}

pub fn convert_forward_integer_u64(
    out: &mut [MaybeUninit<f64>],
    in_re: &[u64],
    in_im: &[u64],
    twisties: Twisties<&[f64]>,
) {
    #[cfg(feature = "nightly")]
    if let Some(simd) = Avx512::try_new() {
        return convert_forward_integer_u64_avx512(simd, out, in_re, in_im, twisties);
    }
    if let Some(simd) = FusedMulAdd::try_new() {
        return convert_forward_integer_u64_fma(simd, out, in_re, in_im, twisties);
    }

    super::convert_forward_integer_u64_scalar(out, in_re, in_im, twisties);
}

pub fn convert_add_backward_torus_u64(
    out_re: &mut [u64],
    out_im: &mut [u64],
    inp: &[f64],
    twisties: Twisties<&[f64]>,
) {
    #[cfg(feature = "nightly")]
    if let Some(simd) = Avx512::try_new() {
        return convert_add_backward_torus_u64_avx512f(simd, out_re, out_im, inp, twisties);
    }
    if let Some(simd) = FusedMulAdd::try_new() {
        return convert_add_backward_torus_u64_fma(simd, out_re, out_im, inp, twisties);
    }

    super::convert_add_backward_torus_u64_scalar(out_re, out_im, inp, twisties);
}

#[cfg(test)]
mod tests {
    #[test]
    fn f64_to_i64_bit_twiddles() {
        for x in [
            0.0,
            -0.0,
            37.1242161_f64,
            -37.1242161_f64,
            0.1,
            -0.1,
            1.0,
            -1.0,
            0.9,
            -0.9,
            2.0,
            -2.0,
            1e-310,
            -1e-310,
            2.0_f64.powi(62),
            -(2.0_f64.powi(62)),
            1.1 * 2.0_f64.powi(62),
            1.1 * -(2.0_f64.powi(62)),
            -(2.0_f64.powi(63)),
        ] {
            // this test checks the correctness of converting from f64 to i64 by manipulating the
            // bits of the ieee754 representation of the floating point values.
            //
            // if the value is not representable as an i64, the result is unspecified.
            //
            // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
            let bits = x.to_bits();
            let implicit_mantissa = bits & 0xFFFFFFFFFFFFF;
            let explicit_mantissa = implicit_mantissa | 0x10000000000000;
            let biased_exp = ((bits >> 52) & 0x7FF) as i64;
            let sign = bits >> 63;

            let explicit_mantissa_lshift = explicit_mantissa << 11;

            // equivalent to:
            //
            // let exp = biased_exp - 1023;
            // let explicit_mantissa_shift = explicit_mantissa_lshift >> (63 - exp.max(0));
            let right_shift_amount = (1086 - biased_exp) as u64;

            let explicit_mantissa_shift = if right_shift_amount < 64 {
                explicit_mantissa_lshift >> right_shift_amount
            } else {
                0
            };

            let value = if sign == 0 {
                explicit_mantissa_shift as i64
            } else {
                (explicit_mantissa_shift as i64).wrapping_neg()
            };

            let value = if biased_exp == 0 { 0 } else { value };
            debug_assert_eq!(value, x as i64);
        }
    }
}
