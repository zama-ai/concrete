# Adding a new backend to the compiler


## Context

The Concrete backends are implementations of the cryptographic primitives of the Zama variant of [TFHE](https://eprint.iacr.org/2018/421.pdf).

There are client features (private and public key generation, encryption and decryption) and server features (homomorphic operations on ciphertexts using public keys).

Considering that
- performance improvements are mostly beneficial for the server operations
- the client needs to be portable for the variety of clients that may exist, 
we expect mostly server backend to be added to the compiler to improve performance (e.g. by using specialized hardware)

## What is needed in the server backend

The server backend should expose C or C++ functions to do TFHE operations using the current ciphertext and key memory representation (or functions to change representation).
A backend can support only a subset of the current TFHE operations.

The most common operations one would be expected to add are WP-PBS (standard TFHE programmable bootstrap), keyswitch and WoP (without padding bootsrap).

Linear operations may also be supported but may need more work since their introduction may interfere with other compilation passes.
The following example does not include this.

## Concrete-cuda example

We will detail how `concrete-cuda` is integrated in the compiler.
Adding a new server feature backend (for non linear operations) should be quite similar.
However, if you want to integrate a backend but it does not fit with this description, please open an issue or contact us to discuss the integration. 


In `compilers/concrete-compiler/Makefile`
- the variable `CUDA_SUPPORT` has been added and set to `OFF` (`CUDA_SUPPORT?=OFF`) by default
- the variables `CUDA_SUPPORT` and `CUDA_PATH` are passed to CMake

```
-DCONCRETELANG_CUDA_SUPPORT=${CUDA_SUPPORT}
-DCUDAToolkit_ROOT=$(CUDA_PATH)
```


In `compilers/concrete-compiler/compiler/include/concretelang/Runtime/context.h`, the `RuntimeContext` struct is enriched with state to manage the backend ressources (behind a `#ifdef CONCRETELANG_CUDA_SUPPORT`).

In `compilers/concrete-compiler/compiler/lib/Runtime/wrappers.cpp`, the cuda backend server functions are added (behind a `#ifdef CONCRETELANG_CUDA_SUPPORT`)

The pass `ConcreteToCAPI` is modified to have a flag to insert calls to these new wrappers instead of the cpu ones (the code calling this pass is modified accordingly).

It may be possible to replace the cpu wrappers (with a compilation flag) instead of adding new ones to avoid having to change the pass.

In `compilers/concrete-compiler/CMakeLists.txt` a Section `#Concrete Cuda Configuration` has been added
Other `CMakeLists.txt` have also been modified (or added) with `if(CONCRETELANG_CUDA_SUPPORT)` guard to handle header includes, linking...


