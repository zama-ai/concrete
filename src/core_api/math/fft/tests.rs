macro_rules! fft_test_mod {
    ($T:ty,$MN:ident) => {
        mod $MN {
            use crate::core_api::math::FFT;
            use crate::types::{C2CPlanTorus, CTorus};
            use fftw::array::AlignedVec;
            use fftw::plan::*;
            use fftw::types::*;

            type Torus = $T;

            #[test]
            fn test_put_2_in_coeff_torus() {
                use rand::Rng;
                let big_n = 512;
                let mut a_1: Vec<Torus> = vec![0; big_n];
                let mut b_1: Vec<Torus> = vec![0; big_n];
                let mut a_2: Vec<Torus> = vec![0; big_n];
                let mut b_2: Vec<Torus> = vec![0; big_n];
                let mut rng = rand::thread_rng();

                // fill ai and b1 with random numbers < 1024
                for i in 0..big_n {
                    a_1[i] = rng.gen::<Torus>().rem_euclid(1024);
                    b_1[i] = rng.gen::<Torus>().rem_euclid(1024);
                }

                let mut a_fft_tmp: AlignedVec<CTorus> = AlignedVec::new(big_n);
                let mut b_fft_tmp: AlignedVec<CTorus> = AlignedVec::new(big_n);
                let mut fft: C2CPlanTorus =
                    C2CPlan::aligned(&[big_n], Sign::Forward, Flag::Measure).unwrap();
                let mut ifft: C2CPlanTorus =
                    C2CPlan::aligned(&[big_n], Sign::Backward, Flag::Measure).unwrap();

                FFT::put_2_in_fft_domain_torus(
                    &mut a_fft_tmp,
                    &mut b_fft_tmp,
                    &a_1,
                    &b_1,
                    TWIDDLES_TORUS!(big_n),
                    &mut fft,
                );

                FFT::put_2_in_coeff_domain(
                    &mut a_2,
                    &mut b_2,
                    &mut a_fft_tmp,
                    &mut b_fft_tmp,
                    INVERSE_TWIDDLES_TORUS!(big_n),
                    &mut ifft,
                );
                for (a, b) in a_1.iter().zip(a_2.iter()) {
                    let tmp = if a.wrapping_sub(*b) < b.wrapping_sub(*a) {
                        a.wrapping_sub(*b)
                    } else {
                        b.wrapping_sub(*a)
                    };
                    if tmp > 10 {
                        panic!("panic1: {} != {}", *a, *b);
                    }
                }
                for (a, b) in b_1.iter().zip(b_2.iter()) {
                    let tmp = if a.wrapping_sub(*b) < b.wrapping_sub(*a) {
                        a.wrapping_sub(*b)
                    } else {
                        b.wrapping_sub(*a)
                    };
                    if tmp > 10 {
                        panic!("panic2: {} != {}", *a, *b);
                    }
                }
            }

            #[test]
            fn test_put_fft_domain() {
                use crate::core_api::math::Tensor;

                let big_n: usize = 1024;
                let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
                let mut tmp: AlignedVec<CTorus> = AlignedVec::new(big_n);

                let mut fft: C2CPlanTorus =
                    C2CPlan::aligned(&[big_n], Sign::Forward, Flag::Measure).unwrap();

                let mut coeff_b: Vec<Torus> = vec![0; big_n];

                Tensor::uniform_random_default(&mut coeff_b);

                FFT::put_in_fft_domain_torus(
                    &mut fft_b,
                    &mut tmp,
                    &coeff_b,
                    TWIDDLES_TORUS!(big_n),
                    &mut fft,
                );
                // use crate::TWIDDLES_TORUS;
            }
        }
    };
}

fft_test_mod!(u32, tests_u32);
fft_test_mod!(u64, tests_u64);
