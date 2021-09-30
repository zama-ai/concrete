# Concrete CSPRNG

This crate contains a fast *Cryptographically Secure Pseudoramdon Number Generator*, used in the
['concrete'](https://crates.io/crates/concrete) library.

The implementation is based on the AES blockcipher used in CTR mode, as described in the ISO/IEC
18033-4 standard.

The current implementation uses special instructions existing on modern *intel* cpus. We may add a
generic implementation in the future.

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions,
please contact us at `hello@zama.ai`.
