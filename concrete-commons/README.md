# Concrete Commons

This crate contains types and traits to manipulate numeric types in a generic manner in the 
[`concrete`](https://crates.io/crates/concrete) library.
It also contains traits and structures to handle the computation of variance, standard deviation, etc.

## Numeric types

For instance, in the standard library, the `f32` and `f64` trait share a lot of methods of the
same name and same semantics. Still, it is not possible to use them generically. This module
provides the [`FloatingPoint`] trait, implemented by both of those type, to remedy the
situation. It also provides the [`SignedInteger`] and [`UnsignedInteger`] traits.

### Note

The current implementation of those traits does not strive to be general, in the sense that
not all the common methods of the same kind of types are exposed. Only were included the ones
that are used in the rest of the library.

## Dispersion

The dispersion module deals with noise distribution.
When dealing with noise, we tend to use different representation for the same value. In
general, the noise is specified by the standard deviation of a gaussian distribution, which
is of the form $\sigma = 2^p$, with $p$ a negative integer. Depending on the use case though,
we rely on different representations for this quantity:

+ $\sigma$ can be encoded in the [`StandardDev`] type.
+ $p$ can be encoded in the [`LogStandardDev`] type.
+ $\sigma^2$ can be encoded in the [`Variance`] type.

In any of those cases, the corresponding type implements the `DispersionParameter` trait,
which makes if possible to use any of those representations generically when noise must be
defined.

## Key kinds

This module contains types to manage the different kinds of secret keys.

## Parameters

This module contains structures that wrap unsigned integer parameters of
concrete, like the ciphertext dimension or the polynomial degree.

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions,
please contact us at `hello@zama.ai`.
