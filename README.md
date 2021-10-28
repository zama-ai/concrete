# Homomorphizer

The homomorphizer is a compiler that takes a high level computation model and produces a programs that evaluate the model in an homomorphic way.

## Build tarball

The final tarball contains intallation instructions. We only support Linux x86_64 for the moment. You can find the output tarball under `/tarballs`.

```bash
$ cd compiler
$ make release_tarballs
```

## Build the Python Package

Currently supported platforms:
- Linux x86_64 for python 3.8, 3.9, and 3.10

### Linux

We use the [manylinux](https://github.com/pypa/manylinux) docker images for building python packages for Linux. Those packages should work on distributions that have GLIBC >= 2.24.

You can use Make to build the python wheels using these docker images:

```bash
$ cd compiler
$ make package_py38  # package_py39 package_py310
```

This will build the image for the appropriate python version then copy the wheels out under `/wheels`

### Build wheels in your environment

#### Temporary MLIR issue

Due to an issue with MLIR, you will need to manually add `__init__.py` files to the `mlir` python package after the build.

```bash
$ make python-bindings
$ touch build/tools/zamalang/python_packages/zamalang_core/mlir/__init__.py
$ touch build/tools/zamalang/python_packages/zamalang_core/mlir/dialects/__init__.py
```

#### Build wheel

Building the wheels is actually simple.

```bash
$ pip wheel --no-deps -w ../wheels .
```

Depending on the platform you are using (specially Linux), you might need to use `auditwheel` to specify the platform this wheel is targeting. For example, in our build of the package for Linux x86_64 and GLIBC 2.24, we also run:

```bash
$ auditwheel repair ../wheels/*.whl --plat manylinux_2_24_x86_64 -w ../wheels
```
