use crate::implementation::{assume_init_mut, from_torus};

use super::as_mut_uninit;
use super::fft::{FftView, Twisties};
use bytemuck::cast_slice_mut;
use concrete_fft::c64;
use core::mem::MaybeUninit;
use dyn_stack::{DynStack, SizeOverflow, StackReq};
use pulp::{as_arrays, as_arrays_mut};

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod x86;

fn convert_forward_integer_u64_scalar(
    out: &mut [MaybeUninit<f64>],
    in_re: &[u64],
    in_im: &[u64],
    twisties: Twisties<&[f64]>,
) {
    debug_assert_eq!(out.len(), in_re.len() * 2);
    let (out, _) = as_arrays_mut::<2, _>(out);
    for (out, in_re, in_im, w_re, w_im) in izip!(out, in_re, in_im, twisties.re, twisties.im) {
        // Don't remove the cast to i64. It can reduce the noise by up to 10 bits.
        let in_re: f64 = *in_re as i64 as f64;
        let in_im: f64 = *in_im as i64 as f64;
        out[0].write(in_re * w_re - in_im * w_im);
        out[1].write(in_re * w_im + in_im * w_re);
    }
}

fn convert_add_backward_torus_u64_scalar(
    out_re: &mut [u64],
    out_im: &mut [u64],
    inp: &[f64],
    twisties: Twisties<&[f64]>,
) {
    debug_assert_eq!(inp.len(), out_re.len() * 2);
    let (inp, _) = as_arrays::<2, _>(inp);

    let normalization = 1.0 / inp.len() as f64;
    for (out_re, out_im, &inp, &w_re, &w_im) in izip!(out_re, out_im, inp, twisties.re, twisties.im)
    {
        let w_re = w_re * normalization;
        let w_im = w_im * normalization;

        let tmp_re = inp[0] * w_re + inp[1] * w_im;
        let tmp_im = inp[1] * w_re - inp[0] * w_im;

        *out_re = out_re.wrapping_add(from_torus(tmp_re));
        *out_im = out_im.wrapping_add(from_torus(tmp_im));
    }
}

fn convert_forward_torus_u64(
    out: &mut [MaybeUninit<f64>],
    in_re: &[u64],
    in_im: &[u64],
    twisties: Twisties<&[f64]>,
) {
    debug_assert_eq!(out.len(), in_re.len() * 2);

    let normalization = 2.0_f64.powi(-(u64::BITS as i32));
    let (out, _) = as_arrays_mut::<2, _>(out);
    for (out, in_re, in_im, w_re, w_im) in izip!(out, in_re, in_im, twisties.re, twisties.im) {
        // Don't remove the cast to i64. It can reduce the noise by up to 10 bits.
        let in_re: f64 = *in_re as i64 as f64 * normalization;
        let in_im: f64 = *in_im as i64 as f64 * normalization;
        out[0].write(in_re * w_re - in_im * w_im);
        out[1].write(in_re * w_im + in_im * w_re);
    }
}

fn convert_backward_torus_u64(
    out_re: &mut [MaybeUninit<u64>],
    out_im: &mut [MaybeUninit<u64>],
    inp: &[f64],
    twisties: Twisties<&[f64]>,
) {
    debug_assert_eq!(inp.len(), out_re.len() * 2);
    let (inp, _) = as_arrays::<2, _>(inp);

    let normalization = 1.0 / inp.len() as f64;
    for (out_re, out_im, inp, w_re, w_im) in izip!(out_re, out_im, inp, twisties.re, twisties.im) {
        let w_re = w_re * normalization;
        let w_im = w_im * normalization;

        let tmp_re = inp[0] * w_re + inp[1] * w_im;
        let tmp_im = inp[1] * w_re - inp[0] * w_im;

        out_re.write(from_torus(tmp_re));
        out_im.write(from_torus(tmp_im));
    }
}

fn convert_forward_integer_u64(
    out: &mut [MaybeUninit<f64>],
    in_re: &[u64],
    in_im: &[u64],
    twisties: Twisties<&[f64]>,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    x86::convert_forward_integer_u64(out, in_re, in_im, twisties);

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    convert_forward_integer_u64_scalar(out, in_re, in_im, twisties);
}

fn convert_add_backward_torus_u64(
    out_re: &mut [u64],
    out_im: &mut [u64],
    inp: &[f64],
    twisties: Twisties<&[f64]>,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    x86::convert_add_backward_torus_u64(out_re, out_im, inp, twisties);

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    convert_add_backward_torus_u64_scalar(out_re, out_im, inp, twisties);
}

impl FftView<'_> {
    /// Returns the polynomial size that this FFT was made for.
    pub fn polynomial_size(self) -> usize {
        2 * self.plan.fft_size()
    }

    /// Returns the memory required for a forward negacyclic FFT.
    pub fn forward_scratch(self) -> Result<StackReq, SizeOverflow> {
        self.plan.fft_scratch()
    }

    /// Returns the memory required for a backward negacyclic FFT.
    pub fn backward_scratch(self) -> Result<StackReq, SizeOverflow> {
        self.plan
            .fft_scratch()?
            .try_and(StackReq::try_new_aligned::<c64>(
                self.polynomial_size() / 2,
                aligned_vec::CACHELINE_ALIGN,
            )?)
    }

    /// Performs a negacyclic real FFT of `standard`, viewed as torus elements, and stores the
    /// result in `fourier`.
    ///
    /// # Postconditions
    ///
    /// this function leaves all the elements of `fourier` in an initialized state.
    ///
    /// # Panics
    ///
    /// Panics if `standard`, `fourier` and `self` have different polynomial sizes.
    pub fn forward_as_torus(
        self,
        fourier: &mut [MaybeUninit<f64>],
        standard: &[u64],
        stack: DynStack<'_>,
    ) {
        // SAFETY: `convert_forward_torus` initializes the output slice that is passed to it
        unsafe { self.forward_with_conv(fourier, standard, convert_forward_torus_u64, stack) }
    }

    /// Performs a negacyclic real FFT of `standard`, viewed as integers, and stores the
    /// result in `fourier`.
    ///
    /// # Postconditions
    ///
    /// this function leaves all the elements of `fourier` in an initialized state.
    ///
    /// # Panics
    ///
    /// Panics if `standard`, `fourier` and `self` have different polynomial sizes.
    pub fn forward_as_integer(
        self,
        fourier: &mut [MaybeUninit<f64>],
        standard: &[u64],
        stack: DynStack<'_>,
    ) {
        // SAFETY: `convert_forward_torus` initializes the output slice that is passed to it
        unsafe { self.forward_with_conv(fourier, standard, convert_forward_integer_u64, stack) }
    }

    /// Performs an inverse negacyclic real FFT of `fourier` and stores the result in `standard`,
    /// viewed as torus elements.
    ///
    /// # Postconditions
    ///
    /// this function leaves all the elements of `standard` in an initialized state.
    ///
    /// # Panics
    ///
    /// Panics if `standard`, `fourier` and `self` have different polynomial sizes.
    pub fn backward_as_torus(
        self,
        standard: &mut [MaybeUninit<u64>],
        fourier: &[f64],
        stack: DynStack<'_>,
    ) {
        // SAFETY: `convert_backward_torus` initializes the output slices that are passed to it
        unsafe { self.backward_with_conv(standard, fourier, convert_backward_torus_u64, stack) }
    }

    /// Performs an inverse negacyclic real FFT of `fourier` and adds the result to `standard`,
    /// viewed as torus elements.
    ///
    /// # Panics
    ///
    /// Panics if `standard`, `fourier` and `self` have different polynomial sizes.
    pub fn add_backward_as_torus(self, standard: &mut [u64], fourier: &[f64], stack: DynStack<'_>) {
        // SAFETY: `convert_add_backward_torus` initializes the output slices that are passed to it
        unsafe {
            self.backward_with_conv(
                as_mut_uninit(standard),
                fourier,
                |out_re, out_im, inp, twisties| {
                    convert_add_backward_torus_u64(
                        assume_init_mut(out_re),
                        assume_init_mut(out_im),
                        inp,
                        twisties,
                    )
                },
                stack,
            )
        }
    }

    /// # Safety
    ///
    /// `conv_fn` must initialize the entirety of the mutable slice that it receives.
    unsafe fn forward_with_conv(
        self,
        fourier: &mut [MaybeUninit<f64>],
        standard: &[u64],
        conv_fn: impl Fn(&mut [MaybeUninit<f64>], &[u64], &[u64], Twisties<&[f64]>),
        stack: DynStack<'_>,
    ) {
        let n = standard.len();
        debug_assert_eq!(n, fourier.len());
        let (standard_re, standard_im) = standard.split_at(n / 2);
        conv_fn(fourier, standard_re, standard_im, self.twisties);

        let fourier = cast_slice_mut(unsafe { assume_init_mut(fourier) });

        self.plan.fwd(fourier, stack);
    }

    /// # Safety
    ///
    /// `conv_fn` must initialize the entirety of the mutable slices that it receives.
    unsafe fn backward_with_conv(
        self,
        standard: &mut [MaybeUninit<u64>],
        fourier: &[f64],
        conv_fn: impl Fn(&mut [MaybeUninit<u64>], &mut [MaybeUninit<u64>], &[f64], Twisties<&[f64]>),
        stack: DynStack<'_>,
    ) {
        let n = standard.len();
        debug_assert_eq!(n, fourier.len());
        let (mut tmp, stack) =
            stack.collect_aligned(aligned_vec::CACHELINE_ALIGN, fourier.iter().copied());
        self.plan.inv(cast_slice_mut(&mut tmp), stack);

        let (standard_re, standard_im) = standard.split_at_mut(n / 2);
        conv_fn(standard_re, standard_im, &tmp, self.twisties);
    }
}
