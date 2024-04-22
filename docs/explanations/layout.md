# Project layout

## Concrete layout

Concrete is a modular framework composed by sub-projects using different technologies, all having theirs own build system and test suite. Each sub-project have is own README that explain how to setup the developer environment, how to build it and how to run tests commands.

Concrete is made of 4 main categories of sub-project that are organized in subdirectories from the root of the Concrete repo:

* `frontends` contains **high-level transpilers** that target end users developers who want to use the Concrete stack easily from their usual environment. There are for now only one frontend provided by the Concrete project: a Python frontend named `concrete-python`.
* `compilers` contains the sub-projects in charge of actually solving the compilation problem of an high-level abstraction of FHE to an actual executable. `concrete-optimizer` is a Rust based project that solves the optimization problems of an FHE dag to a TFHE dag and `concrete-compiler` which use `concrete-optimizer` is an end-to-end MLIR-based compiler that takes a crypto free FHE dialect and generates compilation artifacts both for the client and the server. `concrete-compiler` project provide in addition of the compilation engine, a client and server library in order to easily play with the compilation artifacts to implement a client and server protocol.
* `backends` contains CAPI that can be called by the `concrete-compiler` runtime to perform the cryptographic operations. There are currently two backends:
  * `concrete-cpu`, using TFHE-rs that implement the fastest implementation of TFHE on CPU.
  * `concrete-cuda` that provides a GPU acceleration of TFHE primitives.
* `tools` are basically every other sub-projects that cannot be classified in the three previous categories and which are used as a common support by the others.

## Concrete Python layout

The module structure of **Concrete Python**. You are encouraged to check individual `.py` files to learn more.

* concrete
  * fhe
    * **dtypes:** data type specifications (e.g., int4, uint5, float32)
    * **values:** value specifications (i.e., data type + shape + encryption status)
    * **representation:** representation of computation (e.g., computation graphs, nodes)
    * **tracing:** tracing of python functions
    * **extensions:** custom functionality (see [Extensions](../core-features/extensions.md))
    * **mlir:** computation graph to mlir conversion
    * **compilation:** configuration, compiler, artifacts, circuit, client/server, and anything else related to compilation
