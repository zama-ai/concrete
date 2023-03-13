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
