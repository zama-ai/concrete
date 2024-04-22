# Concrete backends

The Concrete backends are implementations of the cryptographic primitives of the Zama variant of [TFHE](https://eprint.iacr.org/2018/421.pdf).
The compiler emits code which combines call into these backends to perform more complex homomorphic operations.

There are client and server features.

Client features are:
 - private (G)LWE key generation (currently random bits)
 - encryption of ciphertexts using a private key
 - public key generation from private keys for keyswitch, bootstrap or private packing
 - (de)serialization of ciphertexts and public keys (also needed server side)

Server features are homomorphic operations on ciphertexts:
 - linear operations (multisums with plain weights)
 - keyswitch
 - simple PBS
 - WoP PBS


There are currently 2 backends:
- `concrete-cpu` which implements both client and server features targeting the CPU.
- `concrete-cuda` which implements only server features targeting GPUs to accelerate homomorphic circuit evaluation.


The compiler uses `concrete-cpu` for the client and can use either `concrete-cpu` or `concrete-cuda` for the server.
