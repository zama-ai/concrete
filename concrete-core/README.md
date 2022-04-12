# Concrete Core

This crate contains low-level implementations of homomorphic operators used in the
[`concrete`](https://crates.io/crates/concrete) library.

## ⚠ Warning ⚠

This crate assumes that the user is comfortable with the theory behind FHE. If you prefer to use a
simpler API, that will perform sanity checks on your behalf, the higher-level `concrete`
crate should have your back.

## Example

Here is a small example of how one could use `concrete-core` to perform a simple operation
homomorphically:

```rust 
// This examples shows how to multiply a secret value by a public one homomorphically. First
// we import the proper symbols:
use concrete_core::crypto::encoding::{RealEncoder, Cleartext, Encoder, Plaintext};
use concrete_core::crypto::secret::LweSecretKey;
use concrete_core::crypto::LweDimension;
use concrete_core::crypto::lwe::LweCiphertext;
use concrete_core::math::dispersion::LogStandardDev;

// We initialize an encoder that will allow us to turn cleartext values into plaintexts.
let encoder = RealEncoder{offset: 0., delta: 100.};
// Our secret value will be 10.,
let cleartext = Cleartext(10.);
let public_multiplier = Cleartext(5);
// We encode our cleartext
let plaintext = encoder.encode(cleartext);

// We generate a new secret key which is used to encrypt the message
let secret_key_size = LweDimension(710);
let secret_key = LweSecretKey::generate(secret_key_size);

// We allocate a ciphertext and encrypt the plaintext with a secure parameter
let mut ciphertext = LweCiphertext::allocate(0u32, secret_key_size.to_lwe_size());
secret_key.encrypt_lwe(
    &mut ciphertext,
    &plaintext,
    LogStandardDev::from_log_standard_dev(-17.)
);

// We perform the homomorphic operation:
ciphertext.update_with_scalar_mul(public_multiplier);

// We decrypt the message
let mut output_plaintext = Plaintext(0u32);
secret_key.decrypt_lwe(&mut output_plaintext, &ciphertext);
let output_cleartext = encoder.decode(output_plaintext);

// We check that the result is as expected !
assert_eq!((output_cleartext.0 - 50.).abs() < 0.01);
```

## Backends

Two backend are currently implemented. 

### `core` backend

This is the default backend, using the FFTW library for the Fourier transforms.

### `optalysys` backend

This backend is designed to use the Optalysys optical technology for the Fourier transforms. It currently makes use of the Optalysys simulator, and will be updated to use the optical hardware as soon as it is publicly available.

## Links

- [TFHE](https://eprint.iacr.org/2018/421.pdf)
- [concrete-core-1.0.0-alpha release](https://community.zama.ai/t/concrete-core-v1-0-0-alpha/120)
- [Optalysys](https://optalysys.com/)

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions,
please contact us at `hello@zama.ai`.
