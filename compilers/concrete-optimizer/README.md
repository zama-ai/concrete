# Concrete optimizer

Concrete Optimizer is a Rust library that find the best cryptographic parameters for a given TFHE homomorphic circuit.
The goal if to minimize computation time under security and error constaints.
Its main client is Concrete Compiler.
It is implemented in Rust and offers a C++ API.
It also provides a CLI tool to provide parameters for simplified circuits.

# Prerequisite

- Have a rust toolchain installed (the last stable version is supported) https://www.rust-lang.org/tools/install

# Build

Concrete Optimizer and its C++ interface are built automatically by Concrete Compiler.

To build Concrete Optimizer manually, run:
```
cd concrete-optimizer
cargo build --release
```

# Test

To run the tests, run:
```
cargo test
```

# CLI tool Usage

Running `cargo run --release --bin v0-parameters` prints a table of cryptographic parameters for different precisions and log2_norm2

For a given (`precision`, `log2_norm2`), these parameters can be used in a TFHE integer circuit where the maximal integer precision is `precision` and the maximal norm2 between table lookups is `2^log2_norm2`.
They guarantee the given security and probability of error.
The norm2 is the sum of the square of weights in multisum between table lookups or graph inputs (weights on the same input must first be combined as a single weight).
The probablity of error is the maximal acceptable probability of error of each table lookup.

Running `cargo run --release --bin v0-parameters -- --help` shows all available optimization parameters.

For example, `cargo run --release --bin v0-parameters -- --min-precision 3  --max-precision 3 --security-level 192 --p-error 0.01`
