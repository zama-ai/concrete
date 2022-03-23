# Introduction

Welcome to the `concrete-core` guide !

This library contains a set of fast, low-level primitives which can be used to implement *Fully
Homomorphic Encryption* (FHE) programs. In a nutshell, FHE makes it possible to perform arbitrary
computations over encrypted data. With FHE, you can perform computations without putting your trust
on third-party computation providers.

## Quick start

Here is a quick example of how the library can be used to encrypt an integer and decrypt it:

```rust
extern crate concrete_core;
extern crate concrete_commons;

use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::*;

// DISCLAIMER: the parameters used here are only for demo purposes, and are not secure.
let lwe_dimension = LweDimension(630);
// Here a hard-set encoding is applied (shift by 20 bits)
let input = 3_u32 << 20;
let noise = Variance(2_f64.powf(-25.));

let mut engine = CoreEngine::new().unwrap();
let key: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dimension).unwrap();
let plaintext = engine.create_plaintext(&input).unwrap();
let ciphertext = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise).unwrap();

let decrypted_plaintext = engine.decrypt_lwe_ciphertext(&key, &ciphertext).unwrap();

engine.destroy(key).unwrap();
engine.destroy(plaintext).unwrap();
engine.destroy(ciphertext).unwrap();
engine.destroy(decrypted_plaintext).unwrap();
```

## Audience

The goal of `concrete-core` is to be the intermediate layer where library writers and compiler
makers, can access a breadth of fast primitives developed by researchers and hardware designers. As
such, this guide contains two kinds of tutorials geared towards this two categories of users.

### Contributors to the public API

If you are a researcher working on a fast FHE algorithm, or a hardware designer who wants to
accelerate an existing algorithm, you can contribute your code to `concrete-core`. What this gives
you is:

+ The `concrete-benchmark` application, which allows to benchmark your implementation in minutes,
  and compare the results with the existing implementations.
+ The `concrete-core-test` application, which allows to easily test your implementation, to verify
  that it is correct.
+ Exposure to external users, which can easily access your implementation, and integrate it within
  their library or compiler.

If you are interested in contributing new algorithms, or new hardware accelerations
to `concrete-core`, proceed to the [Contributor Guide](../dev/index.rst) section !

### Consumers of the public API

If you are writing a library or a compiler for FHE, you can rely on the public API
of `concrete-core` to access a large set of homomorphic operators. To properly use those operators
though, you have to know your way around FHE. The API gives you the freedom to control multiple
parameters, which can lead to less than 128 bits of security if chosen incorrectly. As a rule of
thumb, if you did not carefully study the mathematical foundations of FHE, `concrete-core` is
probably too low-level for you to use. Fortunately, we propose multiple libraries that build on top
of `concrete-core` and which propose a safer API. To see which one suits your needs best, see
the [concrete homepage](https://zama.ai/concrete).

If you are interested in building a higher level abstraction using the `concrete-core` API, proceed
to the
[API](../api/backends.md) section !
