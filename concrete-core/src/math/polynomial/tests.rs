use crate::crypto::UnsignedTorus;
use crate::math::polynomial::{MonomialDegree, Polynomial};
use crate::math::random;
use rand::Rng;

fn test_multiply_divide_unit_monomial<T: UnsignedTorus>() {
    //! tests if multiply_by_monomial and divide_by_monomial cancel each other
    let mut rng = rand::thread_rng();

    // settings
    let polynomial_size = (rng.gen::<usize>() % 2048) + 1;

    // generates a random Torus polynomial
    let mut poly = Polynomial::from_container(
        random::random_uniform_tensor::<T>(polynomial_size).into_container(),
    );

    // copy this polynomial
    let ground_truth = poly.clone();

    // generates a random r
    let mut r: usize = rng.gen();
    r %= polynomial_size;

    // multiply by X^r and then divides by X^r
    poly.update_with_wrapping_monic_monomial_mul(MonomialDegree(r));
    poly.update_with_wrapping_unit_monomial_div(MonomialDegree(r));

    // test
    assert_eq!(&poly, &ground_truth);

    // generates a random r_big
    let mut r_big: usize = rng.gen();
    r_big = r_big % polynomial_size + 2048;

    // multiply by X^r_big and then divides by X^r_big
    poly.update_with_wrapping_monic_monomial_mul(MonomialDegree(r_big));
    poly.update_with_wrapping_unit_monomial_div(MonomialDegree(r_big));

    // test
    assert_eq!(&poly, &ground_truth);

    // divides by X^r_big and then multiply by X^r_big
    poly.update_with_wrapping_monic_monomial_mul(MonomialDegree(r_big));
    poly.update_with_wrapping_unit_monomial_div(MonomialDegree(r_big));

    // test
    assert_eq!(&poly, &ground_truth);
}

#[test]
pub fn test_multiply_divide_unit_monomial_u32() {
    test_multiply_divide_unit_monomial::<u32>()
}

#[test]
pub fn test_multiply_divide_unit_monomial_u64() {
    test_multiply_divide_unit_monomial::<u64>()
}
