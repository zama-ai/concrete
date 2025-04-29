# Concrete Compiler

The Concrete Compiler is a set of tools that allows the compilation and from an high-level and crypto free representation of an arithmetic circuit of operations on encrypted integers.
This compiler is based on the [MLIR project](https://mlir.llvm.org/) it use the framework, the standard dialects exposed by MLIR and define new fhe specific dialects and passes to lower the high-level fhe dialects to standard MLIR dialects.

## Getting started

The source of the project is located in the `compiler` directory.

```sh
cd compiler
```

### Prerequisite: Install build tools

If you intend to build the Compiler from source, you should make sure to have `build-essential` (or equivalent if not on Ubuntu), [CMake](https://cmake.org/), and [Ninja](https://ninja-build.org/) installed. Optionally, you can also install [Ccache](https://ccache.dev/) to speedup compilations after small changes.

### Prerequisite: Install Rust

The Compiler depends on some Rust libraries, including the optimizer and the concrete-cpu backend. You can install it from [rustup](https://rustup.rs/).

**Note:** some crates use Rust nightly currently, and it might be required to install both stable and nightly distributions.

### Prerequisite: Enable dataflow parallelism (optional)

In order to implement the dataflow parallelism and the distribution of the computation we use the [HPX Standard Library](https://hpx-docs.stellar-group.org/). The build script takes care of installing HPX, but it may rely on a local installation of BOOST.

Once you have a proper installation of BOOST you can enable the dataflow parallelism set the `DATAFLOW_EXECUTION_ENABLED=ON`.

### Prerequisite: Fetch git submodules

This project rely on `llvm-project` and `concrete-optimizer` as git submodules so you need to initialize and update the git submodules.

```sh
git submodule init
git submodule update
```

### Prerequisite: python packages

Install MLIR python requirements in your dev python environment:

```bash
# From repo root
pip install -r ./llvm-project/mlir/python/requirements.txt
# From compiler dir
pip install -r ../llvm-project/mlir/python/requirements.txt
```

You should also have the python development package installed.

### Prerequisite: nightly rust toolchain

If you see a build error like

```
error: toolchain 'nightly-x86_64-unknown-linux-gnu' is not installed
```

it means you need to install the nightly rust toolchain

```bash
rustup toolchain install nightly
```

### Build from source

We use cmake as the main build system but in order to initialize the build system and define straightforward target for the main artifacts of the project. You can initialize and build all the main artifacts thanks the following command:

```sh
make all
```

or in several steps:

Generate the compiler build system, in a `build-*` directory

```sh
make build-initialized
```

Build the compiler

```sh
make concretecompiler
```

Run the compiler

```sh
./build-Release/bin/concretecompiler
```

#### Debug build and custom linker

To build a debug version of the project, you can set `BUILD_TYPE=Debug` in the `Makefile`. In `Debug`
the build system will detect if the `lld` linker is installed on the system and use it. `lld` is much faster
than the default `ld` linker. Release builds with `lld` can also be enabled by modifying the `Makefile`.

### Installation from source

You can install libs, bins, and include files into a specific directory by running:

```sh
make INSTALL_PREFIX=/your/directory install
```

You will then find `lib`, `bin`, and `include` under `/your/directory/concretecompiler`.

### Tests

You can build all the tests with the following command:

```sh
make build-tests
```

and run them with:

```sh
make run-tests
```

### Benchmarks

You can build all the benchmarks with the following command:

```sh
make build-benchmarks
```

and run them with:

```sh
make run-benchmarks
```

## Build releases

### Build tarball

You can create a tarball containing libs, bins, and include files for the tools of the compiler, by following previous steps of [installation from source](#installation-from-source), then creating a tar archive from the installation directory.

### Build the Python Package

> [!IMPORTANT]  
> The wheel built in the following steps is for `concrete-compiler` (which doesn't have the frontend layer) and not `concrete-python`. If you are interested in the `concrete-python` package, then you should build it from [here](https://github.com/zama-ai/concrete/tree/main/frontends/concrete-python) instead.

Currently supported platforms:
- Linux x86_64 for python 3.9, 3.10, 3.11, and 3.12

pybind11 is required to build the python package, you can install it in your current environment with:

```bash
$ pip install pybind11
```

To specify which python executable to target you can specify the `Python3_EXECUTABLE` environment variable.

#### Build wheels in your environment

Building the wheels is actually simple.

```bash
$ pip wheel --no-deps -w ../wheels .
```

Depending on the platform you are using (specially Linux), you might need to use `auditwheel` to specify the platform this wheel is targeting. For example, in our build of the package for Linux x86_64 and GLIBC 2.24, we also run:

```bash
$ auditwheel repair ../wheels/*.whl --plat manylinux_2_24_x86_64 -w ../wheels
```
