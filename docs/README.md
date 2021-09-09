# What is ConcreteFHE

## Introduction

ConcreteFHE, or Concrete for short, is a python package which aims to simplify the use of so-called fully homomorphic encryption (FHE) for datascientists. FHE is a new powerful cryptographic tool, which allows e.g. servers to perform computations directly on encrypted data, without needing any kind of secret key. With FHE, privacy is at the center, and one can build services which ensure full privacy of the user and are the perfect equivalent of their unsecure counterpart. FHE is also a killer feature regarding data breaches: as anything done on the server is done over encrypted data, even if the server is compromised, there is in the end no leak of any kind of useful data.

Concrete framework is made of several parts:
- a library, called concrete-lib, which contains the core cryptographic API's for computing with FHE
- a compiler, called concrete-compiler, which allows to turn an MLIR program into an FHE program, on the top of concrete-lib
- some frontends, which convert different langages to MLIR, to finally be compiled.

In the first version of Concrete framework, there is a single frontend, called concrete-hnp, which is the equivalent of numpy. With our toolchain, a data scientist can convert a numpy program into an FHE program, without any a-priori knowledge on cryptography.

## Organization of the Documentation

Basically, we have divided our documentation into several parts:
- one about basic elements, notably description of the installation, that you are currently reading
- one dedicated to _users_ of Concrete, with tutorials, how-to's and deeper explanations
- and finally, one dedicated to _developpers_ of Concrete, who could be internal or external contributors to the framework

