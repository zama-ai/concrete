# Concrete

The `concrete` project is a set of crates that implements Zama's variant of
[TFHE](https://eprint.iacr.org/2018/421.pdf) and make it easy to use. In a nutshell,
[fully homomorphic encryption (FHE)](https://en.wikipedia.org/wiki/Homomorphic_encryption), allows
you to perform computations over encrypted data, allowing you to implement Zero Trust services.

Concrete is based on the
[Learning With Errors (LWE)](https://cims.nyu.edu/~regev/papers/lwesurvey.pdf) and the
[Ring Learning With Errors (RLWE)](https://eprint.iacr.org/2012/230.pdf) problems, which are well
studied cryptographic hardness assumptions believed to be secure even against quantum computers.


## Project layout

The `concrete` project is a set of several modules which are high-level frontends, compilers, backends and side tools.

- The `frontends` directory contains a `python` frontend.
- The `compilers` directory contains the `concrete-compiler` and `concrete-optimizer` modules. The `concrete-compiler` is a compiler that synthetize a FHE computation dag expressed as a [MLIR](https://mlir.llvm.org/) dialect, compile to a set of artifacts, and provide tools to manipulate those artifacts at runtime. The `concrete-optimizer` is a specific module used by the compiler to find the best, secure and accurate set of crypto parameters for a given dag.
- The `backends` directory contains implementations of cryptographic primitives on different computation unit, used by the `concrete-compiler` runtime. The `concrete-cpu` module provides CPU implementation, while `concrete-cuda` module provides GPU implementation using the CUDA platform.
- The `tools` directory contains side tools used by the rest of the project.
