# Concrete CSPRNG

This crate contains a fast *Cryptographically Secure Pseudoramdon Number Generator*, used in the
['concrete'](https://crates.io/crates/concrete) library.

The implementation is based on the AES blockcipher used in CTR mode, as described in the ISO/IEC
18033-4 standard.

The current implementation uses special instructions existing on modern *intel* cpus. We may add a
generic implementation in the future.

## License

Concrete-csprng is licensed under AGPLv3. If this does not fit your requirements, get in touch
at hello@zama.ai for other licensing options
