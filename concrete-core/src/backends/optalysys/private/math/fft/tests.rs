use crate::backends::core::private::math::fft::twiddles::{BackwardCorrector, ForwardCorrector};
#[cfg(feature = "serde_serialize")]
use crate::backends::core::private::math::fft::SerializableComplex64;
use crate::backends::core::private::math::fft::{Complex64, Fft, FourierPolynomial};
use crate::backends::core::private::math::polynomial::Polynomial;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::PolynomialSize;
use concrete_fftw::array::AlignedVec;
#[cfg(feature = "serde_serialize")]
use serde_test::{assert_tokens, Token};

#[test]
fn test_single_forward_backward() {
    fn fw_conv(
        out: &mut FourierPolynomial<AlignedVec<Complex64>>,
        inp: &Polynomial<Vec<f64>>,
        corr: &ForwardCorrector<&'static [Complex64]>,
    ) {
        for (input, (corrector, output)) in inp
            .as_tensor()
            .iter()
            .zip(corr.as_tensor().iter().zip(out.as_mut_tensor().iter_mut()))
        {
            *output = Complex64::new(*input, 0.) * corrector;
        }
    }
    fn bw_conv(
        out: &mut Polynomial<Vec<f64>>,
        inp: &FourierPolynomial<AlignedVec<Complex64>>,
        corr: &BackwardCorrector<&'static [Complex64]>,
    ) {
        for (input, (corrector, output)) in inp
            .as_tensor()
            .iter()
            .zip(corr.as_tensor().iter().zip(out.as_mut_tensor().iter_mut()))
        {
            *output = (input * corrector).re;
        }
    }
    let mut generator = RandomGenerator::new(None);
    for _ in 0..100 {
        for size in &[128, 256, 512, 1024, 2048, 4096, 8192, 16384] {
            let fft = Fft::new(PolynomialSize(*size));
            let mut poly = Polynomial::allocate(f64::ZERO, PolynomialSize(*size));
            generator.fill_tensor_with_random_gaussian(&mut poly, 0., 1.);
            let mut fourier_poly =
                FourierPolynomial::allocate(Complex64::new(0., 0.), PolynomialSize(*size));
            fft.forward(&mut fourier_poly, &poly, fw_conv);
            let mut out = Polynomial::allocate(f64::ZERO, PolynomialSize(*size));
            fft.backward(&mut out, &mut fourier_poly, bw_conv);
            poly.as_tensor()
                .iter()
                .zip(out.as_tensor().iter())
                .for_each(|(exp, out)| assert!((exp - out).abs() < 1e-12f64))
        }
    }
}

#[test]
fn test_two_forward_backward() {
    fn fw_conv(
        out: &mut FourierPolynomial<AlignedVec<Complex64>>,
        inp1: &Polynomial<Vec<f64>>,
        inp2: &Polynomial<Vec<f64>>,
        corr: &ForwardCorrector<&'static [Complex64]>,
    ) {
        for (input_1, (input_2, (corrector, output))) in inp1.as_tensor().iter().zip(
            inp2.as_tensor()
                .iter()
                .zip(corr.as_tensor().iter().zip(out.as_mut_tensor().iter_mut())),
        ) {
            *output = Complex64::new(*input_1, *input_2) * corrector;
        }
    }
    fn bw_conv(
        out1: &mut Polynomial<Vec<f64>>,
        out2: &mut Polynomial<Vec<f64>>,
        inp: &FourierPolynomial<AlignedVec<Complex64>>,
        corr: &BackwardCorrector<&'static [Complex64]>,
    ) {
        for (input, (corrector, (output1, output2))) in inp.as_tensor().iter().zip(
            corr.as_tensor().iter().zip(
                out1.as_mut_tensor()
                    .iter_mut()
                    .zip(out2.as_mut_tensor().iter_mut()),
            ),
        ) {
            let interm = input * corrector;
            *output1 = interm.re;
            *output2 = interm.im;
        }
    }
    let mut generator = RandomGenerator::new(None);
    for _ in 0..100 {
        for size in &[128, 256, 512, 1024, 2048, 4096, 8192, 16384] {
            let fft = Fft::new(PolynomialSize(*size));
            let mut poly1 = Polynomial::allocate(f64::ZERO, PolynomialSize(*size));
            generator.fill_tensor_with_random_gaussian(&mut poly1, 0., 1.);
            let mut poly2 = Polynomial::allocate(f64::ZERO, PolynomialSize(*size));
            generator.fill_tensor_with_random_gaussian(&mut poly2, 0., 1.);
            let mut fourier_poly_1 =
                FourierPolynomial::allocate(Complex64::new(0., 0.), PolynomialSize(*size));
            let mut fourier_poly_2 =
                FourierPolynomial::allocate(Complex64::new(0., 0.), PolynomialSize(*size));
            fft.forward_two(
                &mut fourier_poly_1,
                &mut fourier_poly_2,
                &poly1,
                &poly2,
                fw_conv,
            );
            let mut out1 = Polynomial::allocate(f64::ZERO, PolynomialSize(*size));
            let mut out2 = Polynomial::allocate(f64::ZERO, PolynomialSize(*size));
            fft.backward_two(
                &mut out1,
                &mut out2,
                &mut fourier_poly_1,
                &mut fourier_poly_2,
                bw_conv,
            );
            poly1
                .as_tensor()
                .iter()
                .zip(out1.as_tensor().iter())
                .for_each(|(exp, out)| assert!((exp - out).abs() < 1e-12f64));
            poly2
                .as_tensor()
                .iter()
                .zip(out2.as_tensor().iter())
                .for_each(|(exp, out)| assert!((exp - out).abs() < 1e-12f64));
        }
    }
}

#[cfg(feature = "serde_serialize")]
#[test]
fn test_ser_de_complex64() {
    let x = SerializableComplex64(Complex64 {
        re: 1.234,
        im: 5.678,
    });

    assert_tokens(
        &x,
        &[
            Token::Tuple { len: 2 },
            Token::F64(1.234),
            Token::F64(5.678),
            Token::TupleEnd,
        ],
    );
}
