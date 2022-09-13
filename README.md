# Concrete Compiler

The Concrete Compiler is a set of tools that allows the compilation and from an high-level and crypto free representation of an arithmetic circuit of operations on encrypted integers.
This compiler is based on the [MLIR project](https://mlir.llvm.org/) it use the framework, the standard dialects exposed by MLIR and define new fhe specific dialects and passes to lower the high-level fhe dialects to standard MLIR dialects.

## Getting started

The source of the project is located in the `compiler` directory.

```sh
cd compiler
```

### Prerequisite: Building HPX and enable dataflow parallelism (optional)

In order to implement the dataflow parallelism and the distribution of the computation we use the [HPX Standard Library](https://hpx-docs.stellar-group.org/). You can else use your own HPX installation by set the `HPX_INSTALL_DIR` environment variable or you can install HPX on the default path of our build system thanks the following command:

```sh
make install-hpx-from-source
```

This may fail on some systems when dependences are missing. Some recent packages required are Cmake, HWLOC and BOOST. For full details see [HPX Quickstart guide](https://hpx-docs.stellar-group.org/tags/1.7.1/html/quickstart.html).
Once you have a proper installation of HPX to enable the dataflow parallelism set the `DATAFLOW_EXECUTION_ENABLED=ON`.

### Prerequisite: Fetch git submodules

This project rely on `llvm-project` and `concrete-optimizer` as git submodules so you need to initialize and update the git submodules.

```sh
git submodules init
git submodules update
```

### Build from source

We use cmake as the main build system but in order to initialize the build system and define straightforward target for the main artifacts of the project. You can initialize and build all the main artifacts thanks the following command:

```sh
make all
```

### Tests

You can build all the tests with the following command:

```
make run-tests
```

### Benchmarks

You can build all the benchmarks with the following command:

```
make run-benchmarks
```

## Build releases

### Build tarball

The final tarball contains intallation instructions. We only support Linux x86_64 for the moment. You can find the output tarball under `/tarballs`.

```bash
$ cd compiler
$ make release-tarballs
```

### Build the Python Package

Currently supported platforms:
- Linux x86_64 for python 3.7, 3.8, 3.9, and 3.10

#### Linux

We use the [manylinux](https://github.com/pypa/manylinux) docker images for building python packages for Linux. Those packages should work on distributions that have GLIBC >= 2.24.

You can use Make to build the python wheels using these docker images:

```bash
$ cd compiler
$ make package_py38  # package_py39 package_py310
```

This will build the image for the appropriate python version then copy the wheels out under `/wheels`

#### Build wheels in your environment

Building the wheels is actually simple.

```bash
$ pip wheel --no-deps -w ../wheels .
```

Depending on the platform you are using (specially Linux), you might need to use `auditwheel` to specify the platform this wheel is targeting. For example, in our build of the package for Linux x86_64 and GLIBC 2.24, we also run:

```bash
$ auditwheel repair ../wheels/*.whl --plat manylinux_2_24_x86_64 -w ../wheels
```
