//! This trait contains functions related to the dispersion of secret key coefficients, and
//! operations related to the secret keys (e.g., products of secret keys).

use concrete_commons::dispersion::*;
use concrete_commons::parameters::PolynomialSize;

use super::*;
use concrete_commons::key_kinds::*;
use concrete_commons::numeric::UnsignedInteger;

// The Gaussian secret keys have modular standard deviation set to 3.2 by default.
const GAUSSIAN_MODULAR_STDEV: f64 = 3.2;

/// This trait contains functions related to the dispersion of secret key coefficients, and
/// operations related to the secret keys (e.g., products of secret keys).
pub trait KeyDispersion: KeyKind {
    /// Returns the variance of key coefficients.
    /// # Example
    ///```rust
    /// use concrete_commons::dispersion::*;
    /// use concrete_commons::key_kinds::*;
    /// use concrete_npe::*;
    ///
    /// type ui = u64;
    ///
    /// let var_out_1 =
    ///     Variance::get_modular_variance::<ui>(&GaussianKeyKind::variance_key_coefficient::<ui>());
    /// let expected_var_out_1 = 10.24;
    /// println!("{}", var_out_1);
    /// assert!((expected_var_out_1 - var_out_1).abs() < 0.0001);
    /// ```
    fn variance_key_coefficient<T: UnsignedInteger>() -> Variance;

    /// Returns the expectation of key coefficients.
    /// # Example
    ///```rust
    /// use concrete_commons::dispersion::*;
    /// use concrete_commons::key_kinds::*;
    /// use concrete_npe::*;
    ///
    /// type ui = u64;
    ///
    /// let expect_out_1 = BinaryKeyKind::expectation_key_coefficient();
    /// let expected_expect_out_1 = 0.5;
    /// println!("{}", expect_out_1);
    /// assert!((expected_expect_out_1 - expect_out_1).abs() < 0.0001);
    /// ```
    fn expectation_key_coefficient() -> f64;

    /// Returns the variance of the squared key coefficients.
    /// # Example
    ///```rust
    /// use concrete_commons::dispersion::*;
    /// use concrete_commons::key_kinds::*;
    /// use concrete_npe::*;
    ///
    /// type ui = u64;
    ///
    /// let var_out_2 = Variance::get_modular_variance::<ui>(
    ///     &TernaryKeyKind::variance_key_coefficient_squared::<ui>(),
    /// );
    /// let expected_var_out_2 = 0.2222;
    /// println!("{}", var_out_2);
    /// assert!((expected_var_out_2 - var_out_2).abs() < 0.0001);
    /// ```
    fn variance_key_coefficient_squared<T: UnsignedInteger>() -> Variance;

    /// Returns the expectation of the squared key coefficients.
    /// # Example
    ///```rust
    /// use concrete_commons::dispersion::*;
    /// use concrete_commons::key_kinds::*;
    /// use concrete_npe::*;
    ///
    /// type ui = u64;
    ///
    /// let expect_out_2 = GaussianKeyKind::expectation_key_coefficient_squared::<ui>();
    /// let expected_expect_out_2 = 10.24;
    /// println!("{}", expect_out_2);
    /// assert!((expected_expect_out_2 - expect_out_2).abs() < 0.0001);
    /// ```
    fn expectation_key_coefficient_squared<T: UnsignedInteger>() -> f64;

    /// Returns the variance of the odd coefficients of a polynomial key to the square.
    /// # Example
    ///```rust
    /// use concrete_commons::dispersion::*;
    /// use concrete_commons::key_kinds::*;
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_npe::*;
    ///
    /// type ui = u64;
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// let var_odd_out_3 = Variance::get_modular_variance::<ui>(
    ///     &TernaryKeyKind::variance_odd_coefficient_in_polynomial_key_squared::<ui>(polynomial_size),
    /// );
    /// let expected_var_odd_out_3 = 910.2222;
    /// println!("{}", var_odd_out_3);
    /// assert!((expected_var_odd_out_3 - var_odd_out_3).abs() < 0.0001);
    /// ```
    fn variance_odd_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance;

    /// Returns the variance of the even coefficients of a polynomial key to the square
    /// # Example
    ///```rust
    /// use concrete_commons::dispersion::*;
    /// use concrete_commons::key_kinds::*;
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_npe::*;
    ///
    /// type ui = u64;
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// let var_even_out_3 = Variance::get_modular_variance::<ui>(
    ///     &BinaryKeyKind::variance_even_coefficient_in_polynomial_key_squared::<ui>(polynomial_size),
    /// );
    /// let expected_var_even_out_3 = 383.75;
    /// println!("{}", var_even_out_3);
    /// assert!((expected_var_even_out_3 - var_even_out_3).abs() < 0.0001);
    /// ```
    fn variance_even_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance;

    /// Returns the mean expectation of the coefficients of a polynomial key to the square.
    /// # Example
    ///```rust
    /// use concrete_commons::dispersion::*;
    /// use concrete_commons::key_kinds::*;
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_npe::*;
    ///
    /// type ui = u64;
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// let expect_out_3 =
    ///     GaussianKeyKind::squared_expectation_mean_in_polynomial_key_squared::<ui>(polynomial_size);
    /// let expected_expect_out_3 = 0.0;
    /// println!("{}", expect_out_3);
    /// assert!((expected_expect_out_3 - expect_out_3).abs() < 0.0001);
    /// ```
    fn squared_expectation_mean_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> f64;

    /// Returns the variance of the
    /// coefficients of a polynomial key resulting from the multiplication of two polynomial keys
    /// of the same key kind ($S_i \cdot S_j$ with $i,j$ different).
    /// # Example
    ///```rust
    /// use concrete_commons::dispersion::*;
    /// use concrete_commons::key_kinds::*;
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_npe::*;
    ///
    /// type ui = u64;
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// let var_out_4 = Variance::get_modular_variance::<ui>(
    ///     &ZeroKeyKind::variance_coefficient_in_polynomial_key_times_key::<ui>(polynomial_size),
    /// );
    /// let expected_var_out_4 = 0.0;
    /// println!("{}", var_out_4);
    /// assert!((expected_var_out_4 - var_out_4).abs() < 0.0001);
    /// ```
    fn variance_coefficient_in_polynomial_key_times_key<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance;

    /// Returns the mean expectation of
    /// the coefficients of a polynomial key resulting from the multiplication of two polynomial
    /// keys of the same key kind ($S_i \cdot S_j$ with $i,j$ different).
    /// # Example
    ///```rust
    /// use concrete_commons::dispersion::*;
    /// use concrete_commons::key_kinds::*;
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_npe::*;
    ///
    /// type ui = u64;
    /// let polynomial_size = PolynomialSize(2048);
    ///
    /// let expect_out_4 =
    ///     BinaryKeyKind::square_expectation_mean_in_polynomial_key_times_key(polynomial_size);
    /// let expected_expect_out_4 = 87381.375;
    /// println!("{}", expect_out_4);
    /// assert!((expected_expect_out_4 - expect_out_4).abs() < 0.0001);
    /// ```
    fn square_expectation_mean_in_polynomial_key_times_key(poly_size: PolynomialSize) -> f64;
}

/// Implementations are provided for binary, ternary and Gaussian key kinds.
/// The ZeroKeyKind is only for debug purposes.
impl KeyDispersion for BinaryKeyKind {
    fn variance_key_coefficient<T: UnsignedInteger>() -> Variance {
        Variance::from_modular_variance::<T>(1. / 4.)
    }
    fn expectation_key_coefficient() -> f64 {
        1. / 2.
    }
    fn variance_key_coefficient_squared<T: UnsignedInteger>() -> Variance {
        Variance::from_modular_variance::<T>(1. / 4.)
    }
    fn expectation_key_coefficient_squared<T: UnsignedInteger>() -> f64 {
        1. / 2.
    }
    fn variance_odd_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance {
        if poly_size.0 == 1 {
            Variance::from_modular_variance::<T>(0.)
        } else {
            Variance::from_modular_variance::<T>(3. * (poly_size.0 as f64) / 8.)
        }
    }
    fn variance_even_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance {
        if poly_size.0 == 1 {
            Variance::from_modular_variance::<T>(
                2. * Variance::get_modular_variance::<T>(
                    &BinaryKeyKind::variance_key_coefficient_squared::<T>(),
                ),
            )
        } else {
            Variance::from_modular_variance::<T>(((3 * poly_size.0 - 2) as f64) / 8.)
        }
    }
    fn squared_expectation_mean_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> f64 {
        if poly_size.0 == 1 {
            square(BinaryKeyKind::expectation_key_coefficient_squared::<T>())
        } else {
            (square(poly_size.0 as f64) + 2.) / 48.
        }
    }
    fn variance_coefficient_in_polynomial_key_times_key<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance {
        Variance::from_modular_variance::<T>(3. * (poly_size.0 as f64) / 16.)
    }
    fn square_expectation_mean_in_polynomial_key_times_key(poly_size: PolynomialSize) -> f64 {
        (square(poly_size.0 as f64) + 2.) / 48.
    }
}

impl KeyDispersion for TernaryKeyKind {
    fn variance_key_coefficient<T: UnsignedInteger>() -> Variance {
        Variance::from_modular_variance::<T>(2. / 3.)
    }
    fn expectation_key_coefficient() -> f64 {
        0.
    }
    fn variance_key_coefficient_squared<T: UnsignedInteger>() -> Variance {
        Variance::from_modular_variance::<T>(2. / 9.)
    }
    fn expectation_key_coefficient_squared<T: UnsignedInteger>() -> f64 {
        2. / 3.
    }
    fn variance_odd_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance {
        if poly_size.0 == 1 {
            Variance::from_modular_variance::<T>(0.)
        } else {
            Variance::from_modular_variance::<T>(8. * (poly_size.0 as f64) / 9.)
        }
    }
    fn variance_even_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance {
        if poly_size.0 == 1 {
            Variance::from_modular_variance::<T>(
                2. * Variance::get_modular_variance::<T>(
                    &TernaryKeyKind::variance_key_coefficient_squared::<T>(),
                ),
            )
        } else {
            Variance::from_modular_variance::<T>(4. * ((2 * poly_size.0 - 3) as f64) / 9.)
        }
    }
    fn squared_expectation_mean_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> f64 {
        if poly_size.0 == 1 {
            square(TernaryKeyKind::expectation_key_coefficient_squared::<T>())
        } else {
            0.
        }
    }
    fn variance_coefficient_in_polynomial_key_times_key<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance {
        Variance::from_modular_variance::<T>(4. * (poly_size.0 as f64) / 9.)
    }
    fn square_expectation_mean_in_polynomial_key_times_key(_poly_size: PolynomialSize) -> f64 {
        0.
    }
}

impl KeyDispersion for GaussianKeyKind {
    fn variance_key_coefficient<T: UnsignedInteger>() -> Variance {
        Variance::from_modular_variance::<T>(square(GAUSSIAN_MODULAR_STDEV))
    }
    fn expectation_key_coefficient() -> f64 {
        0.
    }
    fn variance_key_coefficient_squared<T: UnsignedInteger>() -> Variance {
        Variance::from_modular_variance::<T>(
            2. * square(Variance::get_modular_variance::<T>(
                &GaussianKeyKind::variance_key_coefficient::<T>(),
            )),
        )
    }
    fn expectation_key_coefficient_squared<T: UnsignedInteger>() -> f64 {
        Variance::get_modular_variance::<T>(&GaussianKeyKind::variance_key_coefficient::<T>())
    }
    fn variance_odd_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance {
        if poly_size.0 == 1 {
            Variance::from_modular_variance::<T>(0.)
        } else {
            Variance::from_modular_variance::<T>(
                2. * (poly_size.0 as f64)
                    * square(Variance::get_modular_variance::<T>(
                        &GaussianKeyKind::variance_key_coefficient::<T>(),
                    )),
            )
        }
    }
    fn variance_even_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance {
        if poly_size.0 == 1 {
            Variance::from_modular_variance::<T>(
                2. * Variance::get_modular_variance::<T>(
                    &GaussianKeyKind::variance_key_coefficient_squared::<T>(),
                ),
            )
        } else {
            Variance::from_modular_variance::<T>(
                2. * (poly_size.0 as f64)
                    * square(Variance::get_modular_variance::<T>(
                        &GaussianKeyKind::variance_key_coefficient::<T>(),
                    )),
            )
        }
    }
    fn squared_expectation_mean_in_polynomial_key_squared<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> f64 {
        if poly_size.0 == 1 {
            square(GaussianKeyKind::expectation_key_coefficient_squared::<T>())
        } else {
            0.
        }
    }
    fn variance_coefficient_in_polynomial_key_times_key<T: UnsignedInteger>(
        poly_size: PolynomialSize,
    ) -> Variance {
        Variance::from_modular_variance::<T>(
            square(Variance::get_modular_variance::<T>(
                &GaussianKeyKind::variance_key_coefficient::<T>(),
            )) * (poly_size.0 as f64),
        )
    }
    fn square_expectation_mean_in_polynomial_key_times_key(_poly_size: PolynomialSize) -> f64 {
        0.
    }
}

impl KeyDispersion for ZeroKeyKind {
    fn variance_key_coefficient<T: UnsignedInteger>() -> Variance {
        Variance::from_modular_variance::<T>(0.)
    }
    fn expectation_key_coefficient() -> f64 {
        0.
    }
    fn variance_key_coefficient_squared<T: UnsignedInteger>() -> Variance {
        Variance::from_modular_variance::<T>(0.)
    }
    fn expectation_key_coefficient_squared<T: UnsignedInteger>() -> f64 {
        0.
    }
    fn variance_odd_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        _poly_size: PolynomialSize,
    ) -> Variance {
        Variance::from_modular_variance::<T>(0.)
    }
    fn variance_even_coefficient_in_polynomial_key_squared<T: UnsignedInteger>(
        _poly_size: PolynomialSize,
    ) -> Variance {
        Variance::from_modular_variance::<T>(0.)
    }
    fn squared_expectation_mean_in_polynomial_key_squared<T: UnsignedInteger>(
        _poly_size: PolynomialSize,
    ) -> f64 {
        0.
    }
    fn variance_coefficient_in_polynomial_key_times_key<T: UnsignedInteger>(
        _poly_size: PolynomialSize,
    ) -> Variance {
        Variance::from_modular_variance::<T>(0.)
    }
    fn square_expectation_mean_in_polynomial_key_times_key(_poly_size: PolynomialSize) -> f64 {
        0.
    }
}
