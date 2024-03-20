# Concrete Cpu

The `concrete-cpu` project is a Rust cpu-based implementation of the cryptographic primitives of the Zama variant of TFHE. This implementation aims to use moderns cpu features to run as fast as possible on recent CPUs.

In order to be integrated in a C-based project like the `concrete-compiler` runtime, `concrete-cpu` also offer a C-API.

## Getting started

### Prerequisites

The `concrete-cpu` project is implemented thanks Rust, thus as the main prerequisite the rust toolchain must be installed. You can install from the official [Install Rust instructions](https://www.rust-lang.org/tools/install). Some of features like the use of avx512 instructions are available only with a nightly rust toolchain so if you want to use it you need to install it following those [instructions](https://rust-lang.github.io/rustup/concepts/channels.html).

### Setting RUSTFLAGS

As mentioned before the `concrete-cpu` project aims to use moderns CPU features, to be sure to activate all that is available in your machine you can export the following rust flags:

```
export RUSTFLAGS="-C target-cpu=native"
```

Or you can use the following set of CPU features that should be available on most of the modern CPUs:

```
-Ctarget-feature=+aes,+sse2,+avx,+avx2
```

### Build

Finally you can build using the `stable` Rust toolchain in release mode using

```
cargo build --release
```

Or to enable avx512 support which will detected at runtime you can build with the `nightly` toolchain using

```
cargo +nightly build --release --features=nightly
```

Once the build is done you can link your project with the static library located at `target/release/libconcrete_cpu.a` with the corresponding C header that is located at `include/concrete-cpu.h`.

## Testing

Run basic Rust tests:
```
cargo test
```

Run C-API tests:
Prerequisite: zig version 0.10 installed
```
cd test
make test
```
