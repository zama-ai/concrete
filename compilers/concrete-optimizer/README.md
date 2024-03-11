# Concrete optimizer

Concrete Optimizer is a Rust library that find the best cryptographic parameters for a given TFHE homomorphic circuit.
The goal if to minimize computation time under security and error constraints.
Its main client is Concrete Compiler.
It is implemented in Rust and offers a C++ API.
It also provides a CLI tool to provide parameters for simplified circuits.

# Prerequisite

- Have a rust toolchain installed (the last stable version is supported) https://www.rust-lang.org/tools/install

# Build

Concrete Optimizer and its C++ interface are built automatically by Concrete Compiler.

To build Concrete Optimizer manually, run:
```
cargo build --release
```

# Test

To run the tests, run:
```
cargo test --release
```

# CLI tool Usage

Running `./optimizer` prints a [table of cryptographic parameters](./v0-parameters/README.md#v0-parameters) for different precisions and `log2(norm2)`.

The tools accepts different [options](./v0-parameters/README.md#usage) that can be recalled by:
```
./optimizer --help
```
