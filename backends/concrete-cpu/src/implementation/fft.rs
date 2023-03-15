use alloc::sync::Arc;

use aligned_vec::{avec, ABox};
use concrete_fft::unordered::Plan;

use crate::implementation::zip_eq;

use super::Container;

/// Twisting factors from the paper:
/// [Fast and Error-Free Negacyclic Integer Convolution using Extended Fourier Transform][paper]
///
/// The real and imaginary parts form (the first `N/2`) `2N`-th roots of unity.
///
/// [paper]: https://eprint.iacr.org/2021/480
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct Twisties<C: Container<Item = f64>> {
    pub re: C,
    pub im: C,
}

impl<C: Container<Item = f64>> Twisties<C> {
    pub fn as_view(&self) -> Twisties<&[f64]> {
        Twisties {
            re: self.re.as_ref(),
            im: self.im.as_ref(),
        }
    }
}

impl Twisties<ABox<[f64]>> {
    /// Creates a new [`Twisties`] containing the `2N`-th roots of unity with `n = N/2`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is not a power of two.
    pub fn new(n: usize) -> Self {
        debug_assert!(n.is_power_of_two());
        let mut re = avec![0.0; n].into_boxed_slice();
        let mut im = avec![0.0; n].into_boxed_slice();

        let unit = core::f64::consts::PI / (2.0 * n as f64);
        for (i, (re, im)) in zip_eq(&mut *re, &mut *im).enumerate() {
            (*im, *re) = (i as f64 * unit).sin_cos();
        }

        Twisties { re, im }
    }
}

/// Negacyclic Fast Fourier Transform. See [`FftView`] for transform functions.
///
/// This structure contains the twisting factors as well as the
/// FFT plan needed for the negacyclic convolution over the reals.
#[derive(Clone, Debug)]
pub struct Fft {
    plan: Arc<(Twisties<ABox<[f64]>>, Plan)>,
}

/// View type for [`Fft`].
#[derive(Clone, Copy, Debug)]
#[readonly::make]
pub struct FftView<'a> {
    pub plan: &'a Plan,
    pub twisties: Twisties<&'a [f64]>,
}

impl Fft {
    #[inline]
    pub fn as_view(&self) -> FftView<'_> {
        FftView {
            plan: &self.plan.1,
            twisties: self.plan.0.as_view(),
        }
    }
}

#[cfg(feature = "std")]
mod std_only {
    use super::*;
    use concrete_fft::unordered::Method;
    use core::time::Duration;
    use once_cell::sync::OnceCell;
    use std::collections::hash_map::Entry;
    use std::collections::HashMap;
    use std::sync::RwLock;

    type PlanMap = RwLock<HashMap<usize, Arc<OnceCell<Arc<(Twisties<ABox<[f64]>>, Plan)>>>>>;
    static PLANS: OnceCell<PlanMap> = OnceCell::new();
    fn plans() -> &'static PlanMap {
        PLANS.get_or_init(|| RwLock::new(HashMap::new()))
    }

    impl Fft {
        /// Real polynomial of size `size`.
        pub fn new(size: usize) -> Self {
            let global_plans = plans();

            let n = size;
            let get_plan = || {
                let plans = global_plans.read().unwrap();
                let plan = plans.get(&n).cloned();
                drop(plans);

                plan.map(|p| {
                    p.get_or_init(|| {
                        Arc::new((
                            Twisties::new(n / 2),
                            Plan::new(n / 2, Method::Measure(Duration::from_millis(10))),
                        ))
                    })
                    .clone()
                })
            };

            // could not find a plan of the given size, we lock the map again and try to insert it
            let mut plans = global_plans.write().unwrap();
            if let Entry::Vacant(v) = plans.entry(n) {
                v.insert(Arc::new(OnceCell::new()));
            }

            drop(plans);

            Self {
                plan: get_plan().unwrap(),
            }
        }
    }
}

#[cfg(not(feature = "std"))]
mod no_std {
    use concrete_fft::ordered::FftAlgo;
    use concrete_fft::unordered::Method;

    use super::*;

    impl Fft {
        /// Real polynomial of size `size`.
        pub fn new(size: usize) -> Self {
            let n = size.0;
            Self {
                plan: Arc::new((
                    Twisties::new(n / 2),
                    Plan::new(
                        n / 2,
                        Method::UserProvided {
                            base_algo: FftAlgo::Dif4,
                            base_n: 512,
                        },
                    ),
                )),
            }
        }
    }
}
