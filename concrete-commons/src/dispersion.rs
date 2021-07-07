//! Noise distribution
//!
//! When dealing with noise, we tend to use different representation for the same value. In
//! general, the noise is specified by the standard deviation of a gaussian distribution, which
//! is of the form $\sigma = 2^p$, with $p$ a negative integer. Depending on the use case though,
//! we rely on different representations for this quantity:
//!
//! + $\sigma$ can be encoded in the [`StandardDev`] type.
//! + $p$ can be encoded in the [`LogStandardDev`] type.
//! + $\sigma^2$ can be encoded in the [`Variance`] type.
//!
//! In any of those cases, the corresponding type implements the `DispersionParameter` trait,
//! which makes if possible to use any of those representations generically when noise must be
//! defined.

use serde::{Deserialize, Serialize};

use super::UnsignedInteger;

/// A trait for types representing distribution parameters, for a given unsigned integer type.
pub trait DispersionParameter: Clone {
    /// Returns the standard deviation of the distribution, i.e. $\sigma = 2^p$.
    fn get_standard_dev(&self) -> f64;
    /// Returns the variance of the distribution, i.e. $\sigma^2 = 2^{2p}$.
    fn get_variance(&self) -> f64;
    /// Returns base 2 logarithm of the standard deviation of the distribution, i.e.
    /// $\log_2(\sigma)=p$
    fn get_log_standard_dev(&self) -> f64;
    /// For a `Uint` type representing $\mathbb{Z}/2^q\mathbb{Z}$, we return $2^{q-p}$.
    fn get_modular_standard_dev<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger;
    /// For a `Uint` type representing $\mathbb{Z}/2^q\mathbb{Z}$, we return $2^{2(q-p)}$.
    fn get_modular_variance<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger;
    /// For a `Uint` type representing $\mathbb{Z}/2^q\mathbb{Z}$, we return $q-p$.
    fn get_modular_log_standard_dev<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger;
}

/// A distribution parameter that uses the base-2 logarithm of the standard deviation as
/// representation.
///
/// # Example:
///
/// ```
/// use concrete_commons::{DispersionParameter, LogStandardDev};
/// let params = LogStandardDev::from_log_standard_dev(-25.);
/// assert_eq!(params.get_standard_dev(), 2_f64.powf(-25.));
/// assert_eq!(params.get_log_standard_dev(), -25.);
/// assert_eq!(params.get_variance(), 2_f64.powf(-25.).powi(2));
/// assert_eq!(
///     params.get_modular_standard_dev::<u32>(),
///     2_f64.powf(32. - 25.)
/// );
/// assert_eq!(params.get_modular_log_standard_dev::<u32>(), 32. - 25.);
/// assert_eq!(
///     params.get_modular_variance::<u32>(),
///     2_f64.powf(32. - 25.).powi(2)
/// );
///
/// let modular_params = LogStandardDev::from_modular_log_standard_dev::<u32>(22.);
/// assert_eq!(modular_params.get_standard_dev(), 2_f64.powf(-10.));
/// ```
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct LogStandardDev(pub f64);

impl LogStandardDev {
    pub fn from_log_standard_dev(log_std: f64) -> LogStandardDev {
        LogStandardDev(log_std)
    }

    pub fn from_modular_log_standard_dev<Uint>(log_std: f64) -> LogStandardDev
    where
        Uint: UnsignedInteger,
    {
        LogStandardDev(log_std - Uint::BITS as f64)
    }
}

impl DispersionParameter for LogStandardDev {
    fn get_standard_dev(&self) -> f64 {
        f64::powf(2., self.0)
    }
    fn get_variance(&self) -> f64 {
        f64::powf(2., self.0 * 2.)
    }
    fn get_log_standard_dev(&self) -> f64 {
        self.0
    }
    fn get_modular_standard_dev<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger,
    {
        f64::powf(2., Uint::BITS as f64 + self.0)
    }
    fn get_modular_variance<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger,
    {
        f64::powf(2., (Uint::BITS as f64 + self.0) * 2.)
    }
    fn get_modular_log_standard_dev<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger,
    {
        Uint::BITS as f64 + self.0
    }
}

/// A distribution parameter that uses the standard deviation as representation.
///
/// # Example:
///
/// ```
/// use concrete_commons::{DispersionParameter, StandardDev};
/// let params = StandardDev::from_standard_dev(2_f64.powf(-25.));
/// assert_eq!(params.get_standard_dev(), 2_f64.powf(-25.));
/// assert_eq!(params.get_log_standard_dev(), -25.);
/// assert_eq!(params.get_variance(), 2_f64.powf(-25.).powi(2));
/// assert_eq!(
///     params.get_modular_standard_dev::<u32>(),
///     2_f64.powf(32. - 25.)
/// );
/// assert_eq!(params.get_modular_log_standard_dev::<u32>(), 32. - 25.);
/// assert_eq!(
///     params.get_modular_variance::<u32>(),
///     2_f64.powf(32. - 25.).powi(2)
/// );
/// ```
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct StandardDev(pub f64);

impl StandardDev {
    pub fn from_standard_dev(std: f64) -> StandardDev {
        StandardDev(std)
    }

    pub fn from_modular_standard_dev<Uint>(std: f64) -> StandardDev
    where
        Uint: UnsignedInteger,
    {
        StandardDev(std / 2_f64.powf(Uint::BITS as f64))
    }
}

impl DispersionParameter for StandardDev {
    fn get_standard_dev(&self) -> f64 {
        self.0
    }
    fn get_variance(&self) -> f64 {
        self.0.powi(2)
    }
    fn get_log_standard_dev(&self) -> f64 {
        self.0.log2()
    }
    fn get_modular_standard_dev<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger,
    {
        2_f64.powf(Uint::BITS as f64 + self.0.log2())
    }
    fn get_modular_variance<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger,
    {
        2_f64.powf(2. * (Uint::BITS as f64 + self.0.log2()))
    }
    fn get_modular_log_standard_dev<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger,
    {
        Uint::BITS as f64 + self.0.log2()
    }
}

/// A distribution parameter that uses the variance as representation
///
/// # Example:
///
/// ```
/// use concrete_commons::{DispersionParameter, Variance};
/// let params = Variance::from_variance(2_f64.powi(-50));
/// assert_eq!(params.get_standard_dev(), 2_f64.powf(-25.));
/// assert_eq!(params.get_log_standard_dev(), -25.);
/// assert_eq!(params.get_variance(), 2_f64.powf(-25.).powi(2));
/// assert_eq!(
///     params.get_modular_standard_dev::<u32>(),
///     2_f64.powf(32. - 25.)
/// );
/// assert_eq!(params.get_modular_log_standard_dev::<u32>(), 32. - 25.);
/// assert_eq!(
///     params.get_modular_variance::<u32>(),
///     2_f64.powf(32. - 25.).powi(2)
/// );
/// ```
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Variance(pub f64);

impl Variance {
    pub fn from_variance(var: f64) -> Variance {
        Variance(var)
    }

    pub fn from_modular_variance<Uint>(var: f64) -> Variance
    where
        Uint: UnsignedInteger,
    {
        Variance(var / 2_f64.powf(Uint::BITS as f64 * 2.))
    }
}

impl DispersionParameter for Variance {
    fn get_standard_dev(&self) -> f64 {
        self.0.sqrt()
    }
    fn get_variance(&self) -> f64 {
        self.0
    }
    fn get_log_standard_dev(&self) -> f64 {
        self.0.sqrt().log2()
    }
    fn get_modular_standard_dev<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger,
    {
        2_f64.powf(Uint::BITS as f64 + self.0.sqrt().log2())
    }
    fn get_modular_variance<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger,
    {
        2_f64.powf(2. * (Uint::BITS as f64 + self.0.sqrt().log2()))
    }
    fn get_modular_log_standard_dev<Uint>(&self) -> f64
    where
        Uint: UnsignedInteger,
    {
        Uint::BITS as f64 + self.0.sqrt().log2()
    }
}
