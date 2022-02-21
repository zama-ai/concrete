#![deny(rustdoc::broken_intra_doc_links)]
//! Welcome the the `concrete-npe` documentation!
//!
//! # Description
//! This library makes it possible to estimate the noise propagation after homomorphic operations.
//! It makes it possible to obtain characteristics of the output distribution of the noise, that we
//! call **dispersion**, which regroups the
//! variance and expectation. This is particularly useful to track the noise growth during the
//! homomorphic evaluation of a circuit. The explanations and the proofs of these formula can be
//! found in the appendices of the article [Improved Programmable Bootstrapping with Larger
//! Precision
//! and Efficient Arithmetic Circuits for TFHE]([https://eprint.iacr.org/2021/729]) by *Ilaria
//! Chillotti, Damien Ligier, Jean-Baptiste Orfila and Samuel Tap*.
//!
//! # Quick Example
//! The following piece of code shows how to obtain the variance $\sigma_{add}$ of the noise
//! after a simulated homomorphic addition between two ciphertexts which have variances
//! $\sigma_{ct_1}$ and $\sigma_{ct_2}$, respectively.
//!
//! # Example:
//! ```rust
//! use concrete_commons::dispersion::{DispersionParameter, Variance};
//! use concrete_npe::estimate_addition_noise;
//! //We suppose that the two ciphertexts have the same variance.
//! let var1 = Variance(2_f64.powf(-25.));
//! let var2 = Variance(2_f64.powf(-25.));
//!
//! //We call the npe to estimate characteristics of the noise after an addition
//! //between these two variances.
//! //Here, we assume that ciphertexts are encoded over 64 bits.
//! let var_out = estimate_addition_noise::<u64, _, _>(var1, var2);
//! println!("Expect Variance (2^24) =  {}", f64::powi(2., -24));
//! println!("Output Variance {}", var_out.get_variance());
//! assert!((f64::powi(2., -24) - var_out.get_variance()).abs() < 0.0001);
//! ```

#![allow(clippy::upper_case_acronyms)]

mod key_dispersion;
mod operators;
mod tools;

pub use key_dispersion::*;
pub use operators::*;
pub use tools::*;
