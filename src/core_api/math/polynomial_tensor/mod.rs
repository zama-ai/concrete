//! Polynomial Tensor Operations
//! * Contains every function related to polynomial torus tensors

#[cfg(test)]
mod tests;

use crate::core_api::crypto::SecretKey;
use crate::Types;

pub trait PolynomialTensor: Sized {
    fn compute_binary_multisum(
        t_res: &mut [Self],
        t_torus: &[Self],
        t_bool_key: &[Self],
        polynomial_size: usize,
    );
    fn sub_binary_multisum(
        t_res: &mut [Self],
        t_torus: &[Self],
        t_bool: &[Self],
        polynomial_size: usize,
    );
    fn add_binary_multisum(
        t_res: &mut [Self],
        t_torus: &[Self],
        t_bool: &[Self],
        polynomial_size: usize,
    );
    fn compute_binary_multisum_monome(
        t_torus: &[Self],
        t_bool_key: &[Self],
        polynomial_size: usize,
        dimension: usize,
        monomial: usize,
    ) -> Self;
    fn multiply_by_monomial(res: &mut [Self], monomial_degree: usize, polynomial_size: usize);
    fn divide_by_monomial(res: &mut [Self], monomial_degree: usize, polynomial_size: usize);
    fn print_torus(poly: &[Self], polynomial_size: usize);
    fn to_string_torus(poly: &[Self], polynomial_size: usize) -> std::string::String;
    fn to_string_binary_representation(
        poly: &[Self],
        polynomial_size: usize,
    ) -> std::string::String;
    fn to_string_(poly: &[Self], polynomial_size: usize) -> std::string::String;
    fn signed_decompose_one_level(
        sign_decomp: &mut [Self],
        carries: &mut [Self],
        polynomial: &[Self],
        base_log: usize,
        dec_level: usize,
    );
}

macro_rules! impl_trait_polynomial_tensor {
    ($T:ty,$DOC:expr) => {
        impl PolynomialTensor for $T {
            /// Compute the multisum between 2 Torus tensors but one is viewed as a boolean tensor
            /// # Arguments
            /// * `t_res` - a torus slice (output)
            /// * `t_torus` - a torus slice
            /// * `t_bool_key` - a torus slice viewed as a boolean slice
            /// * `polynomial_size` - the size of the polynomials
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::PolynomialTensor ;
            /// use concrete_lib::core_api::crypto::SecretKey ;
            #[doc = $DOC]
            /// let n: usize = 100 ;
            /// let polynomial_size: usize = 1024 ;
            /// let mut t_torus: Vec<Torus> = vec![0; n * polynomial_size] ;
            /// // fill t_torus
            /// // ...
            /// let size_bool_key = <Torus as SecretKey>::get_secret_key_length(n, polynomial_size) ;
            /// let mut t_bool_key = vec![0; size_bool_key] ;
            /// // fill t_bool_key
            /// // ...
            /// // allocation for the result
            /// let mut t_res = vec![0; polynomial_size ] ;
            ///
            /// <Torus as PolynomialTensor>::compute_binary_multisum(&mut t_res, &t_torus, &t_bool_key, polynomial_size ) ;
            /// ```
            #[allow(dead_code)]
            fn compute_binary_multisum(
                t_res: &mut [$T],
                t_torus: &[$T],
                t_bool_key: &[$T],
                polynomial_size: usize,
            ) {
                // test if there is enough bit in t_bool_key to do the multisum
                debug_assert!(t_bool_key.len() * <$T as Types>::TORUS_BIT >= t_torus.len(),
                    "The key is too short :  {} bits for a mask of {} Torus elements.",
                    t_bool_key.len() * <$T as Types>::TORUS_BIT, t_torus.len()) ;
                let mut k: i32;
                let mut bit: bool;
                t_res.iter_mut().for_each(|m| *m = 0);
                for (i, c) in t_torus.iter().enumerate() {
                    for (j, res) in t_res.iter_mut().enumerate() {
                        k = ((j as i32) - (i as i32)) % (polynomial_size as i32);
                        if k < 0 {
                            k += polynomial_size as i32;
                        }
                        bit = SecretKey::get_bit_monomial(
                            t_bool_key,
                            polynomial_size,
                            i / polynomial_size,
                            k as usize,
                        );
                        if bit {
                            if (i % polynomial_size) + (k as usize) >= polynomial_size {
                                *res = res.wrapping_sub(*c);
                            } else {
                                *res = res.wrapping_add(*c);
                            }
                        }
                    }
                }
            }

            /// Subtract the multisum between 2 Torus tensors but one is viewed as a boolean tensor
            /// res = res - \Sigma_i t_torus_i * t_bool_i
            /// # Arguments
            /// * `t_res` - the result (output)
            /// * `t_torus` - a Torus tensor
            /// * `t_bool` - a Torus tensor viewed as a boolean tensor
            /// * `polynomial_size` - the size of the polynomials
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::SecretKey;
            /// use concrete_lib::core_api::math::PolynomialTensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 256;
            /// let dimension: usize = 10;
            ///
            /// // allocate a tensor of polynomials
            /// let mut t_polynomials: Vec<Torus> = vec![0; polynomial_size * dimension];
            ///
            /// // ... (fill the tensor)
            ///
            /// // allocate a tensor of boolean polynomials
            /// let mut t_bool: Vec<Torus> =
            ///     vec![0; <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size)];
            ///
            /// // ... (fill the tensor)
            ///
            /// // allocate the result
            /// let mut t_res: Vec<Torus> = vec![0; polynomial_size * dimension];
            ///
            /// // subtract a binary sum
            /// PolynomialTensor::sub_binary_multisum(
            ///     &mut t_res,
            ///     &t_polynomials,
            ///     &t_bool,
            ///     polynomial_size,
            /// );
            /// ```
            fn sub_binary_multisum(
                t_res: &mut [$T],
                t_torus: &[$T],
                t_bool: &[$T],
                polynomial_size: usize,
            ) {
                // test if there is enough bit in t_bool_key to do the multisum
                debug_assert!(t_bool.len() * <$T as Types>::TORUS_BIT >= t_torus.len(),
                    "The key is too short :  {} bits for a mask of {} Torus elements.",
                t_bool.len() * <$T as Types>::TORUS_BIT, t_torus.len()) ;
                let mut k: i32;
                let mut bit: bool;
                for (i, c) in t_torus.iter().enumerate() {
                    for (j, res) in t_res.iter_mut().enumerate() {
                        k = ((j as i32) - (i as i32)) % (polynomial_size as i32);
                        if k < 0 {
                            k += polynomial_size as i32;
                        }
                        bit = SecretKey::get_bit_monomial(
                            t_bool,
                            polynomial_size,
                            i / polynomial_size,
                            k as usize,
                        );
                        if bit {
                            if (i % polynomial_size) + (k as usize) >= polynomial_size {
                                *res = res.wrapping_add(*c);
                            } else {
                                *res = res.wrapping_sub(*c);
                            }
                        }
                    }
                }
            }

            /// Add the multisum between 2 Torus tensors but one is viewed as a boolean tensor
            /// res = res + \Sigma_i t_torus_i * t_bool_i
            /// # Arguments
            /// * `t_res` - the result (output)
            /// * `t_torus` - a Torus tensor
            /// * `t_bool` - a Torus tensor viewed as a boolean tensor
            /// * `polynomial_size` - the size of the polynomials
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::SecretKey;
            /// use concrete_lib::core_api::math::PolynomialTensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 256;
            /// let dimension: usize = 10;
            ///
            /// // allocate a tensor of polynomials
            /// let mut t_polynomials: Vec<Torus> = vec![0; polynomial_size * dimension];
            ///
            /// // ... (fill the tensor)
            ///
            /// // allocate a tensor of boolean polynomials
            /// let mut t_bool: Vec<Torus> =
            ///     vec![0; <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size)];
            ///
            /// // ... (fill the tensor)
            ///
            /// // allocate the result
            /// let mut t_res: Vec<Torus> = vec![0; polynomial_size * dimension];
            ///
            /// // add a binary sum
            /// PolynomialTensor::add_binary_multisum(
            ///     &mut t_res,
            ///     &t_polynomials,
            ///     &t_bool,
            ///     polynomial_size,
            /// );
            /// ```
            fn add_binary_multisum(
                t_res: &mut [$T],
                t_torus: &[$T],
                t_bool: &[$T],
                polynomial_size: usize,
            ) {
                // test if there is enough bit in t_bool_key to do the multisum
                debug_assert!(t_bool.len() * <$T as Types>::TORUS_BIT >= t_torus.len(),
                    "The key is too short :  {} bits for a mask of {} Torus elements.",
                    t_bool.len() * <$T as Types>::TORUS_BIT, t_torus.len()) ;
                let mut k: i32;
                let mut bit: bool;
                for (i, c) in t_torus.iter().enumerate() {
                    for (j, res) in t_res.iter_mut().enumerate() {
                        k = ((j as i32) - (i as i32)) % (polynomial_size as i32);
                        if k < 0 {
                            k += polynomial_size as i32;
                        }
                        bit = SecretKey::get_bit_monomial(
                            t_bool,
                            polynomial_size,
                            i / polynomial_size,
                            k as usize,
                        );
                        if bit {
                            if (i % polynomial_size) + (k as usize) >= polynomial_size {
                                *res = res.wrapping_sub(*c);
                            } else {
                                *res = res.wrapping_add(*c);
                            }
                        }
                    }
                }
            }

            /// Compute one monomial of a resulting multisum between 2 Torus tensors but one is viewed as a boolean tensor
            /// # Arguments
            /// * `t_torus` - a torus slice
            /// * `t_bool_key` - a torus slice viewed as a boolean slice
            /// * `polynomial_size` - the size of the polynomials
            /// * `dimension` - size of the mask
            /// * `monomial` - the index of the desired monomial
            /// # Output
            /// the coefficient of the desired monomial
            #[allow(dead_code)]
            fn compute_binary_multisum_monome(
                t_torus: &[$T],
                t_bool_key: &[$T],
                polynomial_size: usize,
                dimension: usize,
                monomial: usize,
            ) -> $T {
                let mut res: $T = 0;
                for i in 0..polynomial_size {
                    // set the index of the X^j monomial from the i-th polynomial of the mask
                    let pow_masque: usize = i;
                    // set the index of the complementary monomial of the i-th polynomial of the secret key such that their product is a monomial of degree monomial
                    let mut pow_key: usize = monomial;
                    if monomial < i {
                        pow_key += polynomial_size;
                    }
                    pow_key -= i;

                    for j in 0..dimension {
                        // j goes through the mask
                        if !SecretKey::get_bit_monomial(
                            t_bool_key,
                            polynomial_size,
                            j as usize,
                            pow_key,
                        ) {
                            // the key bit is 0
                            continue;
                        }
                        let index: usize = convert_to_index(polynomial_size, j, pow_masque);
                        if pow_key + pow_masque >= polynomial_size {
                            // multiply by -1
                            res = res.wrapping_sub(t_torus[index]);
                        } else {
                            res = res.wrapping_add(t_torus[index]);
                        }
                    }
                }
                return res;
            }

            /// Multiply several polynomial by a monomial in the ring Torus\[X\] / (X^polynomial_size + 1)
            /// Warning : this function is inplace (overwrite)
            /// # Arguments
            /// * `res` - polynomials to multiply (output)
            /// * `monomial_degree` - degree of the monomial
            /// * `polynomial_size` - nnumber of coefficients of the polynomials
            ///  REFACTOR
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::PolynomialTensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 256;
            /// let dimension: usize = 10;
            ///
            /// // allocate a tensor of polynomials
            /// let mut t_polynomials: Vec<Torus> = vec![0; polynomial_size * dimension];
            ///
            /// // ... (fill the tensor)
            ///
            /// // define a monomial_degree
            /// let monomial_degree: usize = 5;
            ///
            /// // allocate the result
            /// let mut t_res: Vec<Torus> = vec![0; polynomial_size * dimension];
            ///
            /// // add a binary sum
            /// PolynomialTensor::multiply_by_monomial(&mut t_res, monomial_degree, polynomial_size);
            /// ```
            fn multiply_by_monomial(
                res: &mut [$T],
                monomial_degree: usize,
                polynomial_size: usize,
            ) {
                let mut n_rotation: usize = monomial_degree % (2 * polynomial_size);

                if n_rotation > polynomial_size {
                    n_rotation -= polynomial_size;
                    for ref_res in res.iter_mut() {
                        *ref_res = (0 as $T).wrapping_sub(*ref_res);
                    }
                }

                for polynomial in res.chunks_mut(polynomial_size) {
                    polynomial.rotate_right(n_rotation);
                    for coeff in polynomial[..n_rotation].iter_mut() {
                        *coeff = (0 as $T).wrapping_sub(*coeff);
                    }
                }
            }

            /// Divide several polynomial by a monomial in the ring Torus\[X\] / (X^polynomial_size + 1)
            /// Warning : this function is inplace (overwrite)
            /// # Arguments
            /// * `res` - polynomials to multiply (output)
            /// * `monomial_degree` - degree of the monomial
            /// * `polynomial_size` - nnumber of coefficients of the polynomials
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::PolynomialTensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 256;
            /// let dimension: usize = 10;
            ///
            /// // allocate a tensor of polynomials
            /// let mut t_polynomials: Vec<Torus> = vec![0; polynomial_size * dimension];
            ///
            /// // ... (fill the tensor)
            ///
            /// // define a monomial_degree
            /// let monomial_degree: usize = 5;
            ///
            /// // allocate the result
            /// let mut t_res: Vec<Torus> = vec![0; polynomial_size * dimension];
            ///
            /// // add a binary sum
            /// PolynomialTensor::divide_by_monomial(&mut t_res, monomial_degree, polynomial_size);
            /// ```
            fn divide_by_monomial(res: &mut [$T], monomial_degree: usize, polynomial_size: usize) {
                let mut n_rotation: usize = monomial_degree % (2 * polynomial_size);
                if n_rotation > polynomial_size {
                    n_rotation -= polynomial_size;
                    for ref_res in res.iter_mut() {
                        *ref_res = (0 as $T).wrapping_sub(*ref_res);
                    }
                }
                let n_polynomial: usize = res.len() / polynomial_size;
                for i in 0..n_polynomial {
                    res[i * polynomial_size..(i + 1) * polynomial_size].rotate_left(n_rotation);
                    for j in 0..n_rotation {
                        res[i * polynomial_size + polynomial_size - 1 - j] = (0 as $T)
                            .wrapping_sub(res[i * polynomial_size + polynomial_size - 1 - j]);
                    }
                }
            }

            /// Prints in a pretty way a tensor of polynomials over the Torus
            /// Torus elements are between 0 and TORUS_MAX
            /// # Arguments
            /// * `poly` - Torus slice
            /// * `polynomial_size` - the size of the polynomials
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::{PolynomialTensor, Tensor};
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 6;
            ///
            /// // creation of a random polynomial
            /// let mut polynomial: Vec<Torus> = vec![0; polynomial_size];
            /// Tensor::uniform_random_default(&mut polynomial);
            ///
            /// // print
            /// PolynomialTensor::print_torus(&polynomial, polynomial_size);
            ///
            /// // stdout:
            /// // [POLYNOMIAL] poly: 0011 1011 0110 1010 1000 0011 1100 0000 + 0011 0100 1011 1101 1100 1110 0010 0011 X + 1010 0001 1101 0110 0001 1101 1100 1000 X^2 + 1110 1101 0100 1110 0100 1101 0100 1000 X^3 + 1100 1111 1111 1001 0000 1111 1101 1011 X^4 + 1101 0000 0100 1010 0111 0000 1110 1100 X^5
            /// ```
            fn print_torus(poly: &[$T], polynomial_size: usize) {
                print!("[POLYNOMIAL] poly:");
                let s = Self::to_string_binary_representation(poly, polynomial_size);
                println!("{}", s.replace("{}", " "));
            }

            /// Generate a pretty string representing a tensor of polynomials over the Torus
            /// Torus elements are between 0 and TORUS_MAX
            /// # Arguments
            /// * `poly` - Torus slice
            /// * `polynomial_size` - the size of the polynomials
            /// # Arguments
            /// * `poly` - Torus slice
            /// * `polynomial_size` - the size of the polynomials
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::{PolynomialTensor, Tensor};
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 6;
            ///
            /// // creation of a random polynomial
            /// let mut polynomial: Vec<Torus> = vec![0; polynomial_size];
            /// Tensor::uniform_random_default(&mut polynomial);
            ///
            /// // to string
            /// let s = PolynomialTensor::to_string_torus(&polynomial, polynomial_size);
            ///
            /// // output:
            /// // s = "{}0.39626873028464615 + 0.8058619236107916 X + 0.8790576986502856 X^2 + 0.6840280669275671 X^3 + 0.5027693663723767 X^4 + 0.4896212250459939 X^5"
            /// ```
            fn to_string_torus(poly: &[$T], polynomial_size: usize) -> std::string::String {
                let mut res: std::string::String = "{}".to_string();
                for poly_ith in poly.chunks(polynomial_size) {
                    for j in 0..polynomial_size {
                        res = format!("{}{}", res, Types::torus_to_f64(poly_ith[j]));
                        if j > 0 {
                            res = format!("{} X", res);
                        }
                        if j > 1 {
                            res = format!("{}^{}", res, j);
                        }
                        if j != polynomial_size - 1 {
                            res = format!("{} + ", res);
                        }
                    }
                    res = format!("{}\n", res);
                }
                return res;
            }

            /// Generate a pretty string representing a tensor of polynomials over the Torus
            /// Torus elements are displayed with their binary representations
            /// # Arguments
            /// * `poly` - Torus slice
            /// * `polynomial_size` - the size of the polynomials
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::{PolynomialTensor, Tensor};
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 6;
            ///
            /// // creation of a random polynomial
            /// let mut polynomial: Vec<Torus> = vec![0; polynomial_size];
            /// Tensor::uniform_random_default(&mut polynomial);
            ///
            /// // to string
            /// let s = PolynomialTensor::to_string_binary_representation(&polynomial, polynomial_size);
            ///
            /// // output:
            /// // s = "{}1110 1000 0100 1011 1011 1000 1011 1010 + 0110 0010 0010 1100 1011 1011 1011 0110 X + 0101 1010 1011 0111 0010 0111 1011 0101 X^2 + 1011 0101 1111 0100 1000 1001 1110 1100 X^3 + 1010 1000 0000 1010 0000 1100 0100 1100 X^4 + 1011 1010 0001 1101 1110 0100 0010 0001 X^5"
            /// ```
            fn to_string_binary_representation(
                poly: &[$T],
                polynomial_size: usize,
            ) -> std::string::String {
                let mut res: std::string::String = "{}".to_string();
                for poly_ith in poly.chunks(polynomial_size) {
                    for j in 0..polynomial_size {
                        res = format!(
                            "{}{}",
                            res,
                            Types::torus_binary_representation(poly_ith[j], 4)
                        );
                        if j > 0 {
                            res = format!("{} X", res);
                        }
                        if j > 1 {
                            res = format!("{}^{}", res, j);
                        }
                        if j != polynomial_size - 1 {
                            res = format!("{} + ", res);
                        }
                    }
                    res = format!("{}\n", res);
                }
                return res;
            }

            /// Generate a pretty string representing a tensor of polynomials over the Torus
            /// Torus elements are between -TORUS_MAX/2 and TORUS_MAX/2
            /// # Arguments
            /// * `poly` - Torus slice
            /// * `polynomial_size` - the size of the polynomials
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::{PolynomialTensor, Tensor};
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 6;
            ///
            /// // creation of a random polynomial
            /// let mut polynomial: Vec<Torus> = vec![0; polynomial_size];
            /// Tensor::uniform_random_default(&mut polynomial);
            ///
            /// // to string
            /// let s = PolynomialTensor::to_string_(&polynomial, polynomial_size);
            ///
            /// // output:
            /// // s = "{} - 327668133 - 1690174369 X - 1471456801 X^2 - 1820608313 X^3 - 688390573 X^4 + 281745273 X^5"
            /// ```
            fn to_string_(poly: &[$T], polynomial_size: usize) -> std::string::String {
                let mut res: std::string::String = "{}".to_string();
                for j in 0..polynomial_size {
                    let mut tmp = poly[j];
                    if tmp >= 1 << (<$T as Types>::TORUS_BIT - 1) {
                        tmp = <$T as Types>::TORUS_MAX - tmp;
                        tmp += 1;
                        res = format!("{} - ", res);
                    } else {
                        res = format!("{} + ", res);
                    }
                    res = format!("{}{}", res, tmp);
                    if j > 0 {
                        res = format!("{} X", res);
                    }
                    if j > 1 {
                        res = format!("{}^{}", res, j);
                    }
                }
                return res;
            }

            /// Compute the dec_level-th element of the signed decomposition of a Torus polynomials according to some parameters with the previous level's carries
            /// # Arguments
            /// * `sign_decomp` - a Torus slice containing signed values of the decomposition (output)
            /// * `carries` - a Torus slice containing the previous level carries, and will contain the next carries (input AND output)
            /// * `polynomial` - a Torus slice containing a Torus polynomial to be decomposed
            /// * `base_log` - number of bits for the base B (B=2^base_log)
            /// * `dec_level` - level of the decomposition wanted
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::PolynomialTensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 256;
            /// let base_log: usize = 4;
            ///
            /// // allocate a polynomial to decompose
            /// let mut polynomial: Vec<Torus> = vec![0; polynomial_size];
            ///
            /// // ... (fill the polynomial)
            ///
            /// // allocate the carry tensor
            /// let mut carries: Vec<Torus> = vec![0; polynomial_size];
            ///
            /// // ...
            ///
            /// // allocate the output
            /// let mut sign_decomp: Vec<Torus> = vec![0; polynomial_size];
            ///
            /// PolynomialTensor::signed_decompose_one_level(
            ///     &mut sign_decomp,
            ///     &mut carries,
            ///     &polynomial,
            ///     base_log,
            ///     2,
            /// );
            /// ```
            fn signed_decompose_one_level(
                sign_decomp: &mut [$T],
                carries: &mut [$T],
                polynomial: &[$T],
                base_log: usize,
                dec_level: usize,
            ) {
                // loop over the coefficients of the polynomial
                for (carry, (decomp, value)) in carries
                    .iter_mut()
                    .zip(sign_decomp.iter_mut().zip(polynomial.iter()))
                {
                    let pair =
                        Types::signed_decompose_one_level(*value, *carry, base_log, dec_level);
                    *decomp = pair.0;
                    *carry = pair.1;
                }
            }
        }
    };
}

impl_trait_polynomial_tensor!(u32, "type Torus = u32;");
impl_trait_polynomial_tensor!(u64, "type Torus = u64;");

/// Convert indexes (mask and monomial) into a flatten index
/// # Argument
/// * `polynomial_size` - size of the mask
/// * `mask_index` - the index of the wanted mask element
/// * `monomial_index` - the index of the wanted monomial
/// # Output
/// * the wanted index
#[allow(dead_code)]
fn convert_to_index(polynomial_size: usize, mask_index: usize, monomial_index: usize) -> usize {
    mask_index * polynomial_size + monomial_index
}
