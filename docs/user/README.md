# What is **Concrete**

## Introduction

**Concrete Framework**, or **Concrete** for short, is an open-source framework which aims to simplify the use of so-called fully homomorphic encryption (FHE) for data scientists.

FHE is a new powerful cryptographic tool, which allows e.g. servers to perform computations directly on encrypted data, without needing to decrypt first. With FHE, privacy is at the center, and one can build services which ensure full privacy of the user and are the perfect equivalent of their unsecure counterpart.

FHE is also a killer feature regarding data breaches: as anything done on the server is done over encrypted data, even if the server is compromised, there is in the end no leak of any kind of useful data.

**Concrete** is made of several parts:
- a library, called concrete-lib, which contains the core cryptographic API's for computing with FHE
- a compiler, called concrete-compiler, which allows to turn an MLIR program into an FHE program, on the top of concrete-lib
- some frontends, which convert different langages to MLIR, to finally be compiled.

```{important}
In the first version of Concrete, there is a single frontend, called homomorphic numpy (or hnp), which is the equivalent of numpy. With our toolchain, a data scientist can convert a numpy program into an FHE program, without any a-priori knowledge on cryptography.
```

```{note}
On top of the numpy frontend, we are adding an alpha-version of a torch compiler, which basically transforms a subset of torch modules into numpy, and then use numpy frontend and the compiler. This is an early version of a more stable torch compiler which will be released later in the year.
```

## Organization of the documentation

Basically, we have divided our documentation into several parts:
- one about basic elements, notably description of the installation, that you are currently reading
- one dedicated to _users_ of **Concrete**, with tutorials, how-to's and deeper explanations
- one detailing the API's of the different functions of the frontend, directly done by parsing its source code
- and finally, one dedicated to _developers_ of **Concrete**, who could be internal or external contributors to the framework

## A work in progress

```{note}
Concrete is a work in progress, and is currently limited to a certain number of operators and features. In the future, there will be improvements as described in this [section](explanation/FUTURE_FEATURES.md).
```

The main _current_ limits are:
- **Concrete** is only supporting unsigned integers
- **Concrete** needs the integer to be less than 7 bits (included)

These limits can be taken care of with the use of quantization, as explained a bit further in [this](explanation/QUANTIZATION.md) and [this](howto/REDUCE_NEEDED_PRECISION.md) parts of the documentation.

```{warning}
FIXME(Jordan): speak about our quantization framework
```

```{warning}
FIXME(Jordan/Andrei): add an .md about the repository of FHE-friendly models, and ideally .ipynb's
```
