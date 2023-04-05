#[cfg(target_arch = "x86_64")]
use concrete_ntt::native_binary64::Plan32;

#[cfg(target_arch = "x86_64")]
use crate::implementation::zip_eq;

use super::types::polynomial::Polynomial;

pub fn update_with_wrapping_unit_monomial_div(
    mut polynomial: Polynomial<&mut [u64]>,
    monomial_degree: usize,
) {
    let full_cycles_count = monomial_degree / polynomial.len();
    if full_cycles_count % 2 != 0 {
        for a in polynomial.as_mut_view().into_data().iter_mut() {
            *a = a.wrapping_neg()
        }
    }
    let remaining_degree = monomial_degree % polynomial.len();
    polynomial
        .as_mut_view()
        .into_data()
        .rotate_left(remaining_degree);

    for a in polynomial
        .into_data()
        .iter_mut()
        .rev()
        .take(remaining_degree)
    {
        *a = a.wrapping_neg()
    }
}

pub fn update_with_wrapping_monic_monomial_mul(
    polynomial: Polynomial<&mut [u64]>,
    monomial_degree: usize,
) {
    let polynomial = polynomial.into_data();

    let full_cycles_count = monomial_degree / polynomial.len();
    if full_cycles_count % 2 != 0 {
        for a in polynomial.iter_mut() {
            *a = a.wrapping_neg()
        }
    }
    let remaining_degree = monomial_degree % polynomial.len();
    polynomial.rotate_right(remaining_degree);
    for a in polynomial.iter_mut().take(remaining_degree) {
        *a = a.wrapping_neg()
    }
}

pub fn update_with_wrapping_add_mul(
    polynomial: Polynomial<&mut [u64]>,
    lhs_polynomial: Polynomial<&[u64]>,
    rhs_bin_polynomial: Polynomial<&[u64]>,
) {
    debug_assert_eq!(polynomial.len(), lhs_polynomial.len());
    debug_assert_eq!(polynomial.len(), rhs_bin_polynomial.len());

    let polynomial = polynomial.into_data();

    // TODO: optimize performance, while keeping constant time, so as not to leak information about
    // the secret key.
    let dim = polynomial.len();
    for (i, lhs) in lhs_polynomial.iter().enumerate() {
        let lhs = *lhs;

        for (j, rhs) in rhs_bin_polynomial.as_ref().iter().enumerate() {
            let target_degree = i + j;

            if target_degree < dim {
                let update = polynomial[target_degree].wrapping_add(lhs * *rhs);
                polynomial[target_degree] = update;
            } else {
                let update = polynomial[target_degree - dim].wrapping_sub(lhs * *rhs);
                polynomial[target_degree - dim] = update;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub fn update_with_wrapping_add_mul_ntt(
    plan: &Plan32,
    polynomial: Polynomial<&mut [u64]>,
    lhs_polynomial: Polynomial<&[u64]>,
    rhs_bin_polynomial: Polynomial<&[u64]>,
    mut buffer: Polynomial<&mut [u64]>,
) {
    debug_assert_eq!(polynomial.len(), lhs_polynomial.len());
    debug_assert_eq!(polynomial.len(), rhs_bin_polynomial.len());

    plan.negacyclic_polymul(
        buffer.as_mut_view().into_data(),
        lhs_polynomial.into_data(),
        rhs_bin_polynomial.into_data(),
    );

    for (p, b) in zip_eq(polynomial.into_data(), buffer.into_data()) {
        *p = p.wrapping_add(*b);
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::implementation::types::polynomial::Polynomial;

    #[test]
    fn ntt_correct() {
        let ps: usize = 1024;

        let a: Vec<u64> = (1..ps as u64 + 1).collect();
        let a = Polynomial::new(a.as_slice(), ps);

        let b: Vec<u64> = (0..ps / 2).flat_map(|_| [0, 1].into_iter()).collect();

        let b = Polynomial::new(b.as_slice(), ps);

        let mut c = vec![1; ps];
        let mut c = Polynomial::new(c.as_mut_slice(), ps);

        let mut d = vec![1; ps];
        let mut d = Polynomial::new(d.as_mut_slice(), ps);

        let mut buffer = vec![0; ps];
        let buffer = Polynomial::new(buffer.as_mut_slice(), ps);

        let plan = Plan32::try_new(ps).unwrap();

        update_with_wrapping_add_mul(c.as_mut_view(), a, b);
        update_with_wrapping_add_mul_ntt(&plan, d.as_mut_view(), a, b, buffer);

        assert_eq!(c.into_data(), d.into_data());
    }
}

pub fn update_with_wrapping_sub_mul(
    polynomial: Polynomial<&mut [u64]>,
    lhs_polynomial: Polynomial<&[u64]>,
    rhs_bin_polynomial: Polynomial<&[u64]>,
) {
    debug_assert_eq!(polynomial.len(), lhs_polynomial.len());
    debug_assert_eq!(polynomial.len(), rhs_bin_polynomial.len());

    let polynomial = polynomial.into_data();

    // TODO: optimize performance, while keeping constant time, so as not to leak information about
    // the secret key.
    let dim = polynomial.len();
    for (i, lhs) in lhs_polynomial.iter().enumerate() {
        let lhs = *lhs;

        for (j, rhs) in rhs_bin_polynomial.as_ref().iter().enumerate() {
            let target_degree = i + j;

            if target_degree < dim {
                let update = polynomial[target_degree].wrapping_sub(lhs * *rhs);
                polynomial[target_degree] = update;
            } else {
                let update = polynomial[target_degree - dim].wrapping_add(lhs * *rhs);
                polynomial[target_degree - dim] = update;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub fn update_with_wrapping_sub_mul_ntt(
    plan: &Plan32,
    polynomial: Polynomial<&mut [u64]>,
    lhs_polynomial: Polynomial<&[u64]>,
    rhs_bin_polynomial: Polynomial<&[u64]>,
    mut buffer: Polynomial<&mut [u64]>,
) {
    debug_assert_eq!(polynomial.len(), lhs_polynomial.len());
    debug_assert_eq!(polynomial.len(), rhs_bin_polynomial.len());

    plan.negacyclic_polymul(
        buffer.as_mut_view().into_data(),
        lhs_polynomial.into_data(),
        rhs_bin_polynomial.into_data(),
    );

    for (p, b) in zip_eq(polynomial.into_data(), buffer.into_data()) {
        *p = p.wrapping_sub(*b);
    }
}
