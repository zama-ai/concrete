<div align="center">
  <h1>Concrete Numpy</h1>
</div>

<p align="center">
<!-- Version badge using shields.io -->
  <a href="https://github.com/zama-ai/concrete-numpy/releases">
    <img src="https://img.shields.io/github/v/release/zama-ai/concrete-numpy?style=flat-square">
  </a>
<!-- Link to docs badge using shields.io -->
  <a href="https://docs.zama.ai/concrete-numpy/">
    <img src="https://img.shields.io/badge/read-documentations-yellow?style=flat-square">
  </a>
<!-- Community forum badge using shields.io -->
  <a href="https://community.zama.ai/c/concrete-numpy">
    <img src="https://img.shields.io/badge/community%20forum-available-brightgreen?style=flat-square">
  </a>
<!-- Open source badge using shields.io -->
  <a href="https://docs.zama.ai/concrete-numpy/developer/contributing">
    <img src="https://img.shields.io/badge/we're%20open%20source-contributing.md-blue?style=flat-square">
  </a>
<!-- Follow on twitter badge using shields.io -->
  <a href="https://twitter.com/zama_fhe">
    <img src="https://img.shields.io/twitter/follow/zama_fhe?color=blue&style=flat-square">
  </a>
</p>

## Table of contents

- [Introduction](#Introduction)
- [Installation](#Installation)
- [Getting started](#getting-started)
- [License](#license)

## Introduction

**Concrete Numpy** is an open-source library which simplifies the use of fully homomorphic encryption (FHE).

FHE is a powerful cryptographic tool, which allows computation to be performed directly on encrypted data without needing to decrypt it first.

With FHE, you can build services that preserve the privacy of the users. FHE is also great against data breaches as everything is done on encrypted data. Even if the server is compromised, in the end no sensitive data is leaked.

## Installation

The preferred way to install concrete-numpy is through PyPI. To install Concrete Numpy from PyPi, run the following:

```shell
pip install concrete-numpy
```

You can get the concrete-numpy docker image by  pulling the latest docker image:

```shell
docker pull zamafhe/concrete-numpy:v0.6.0
```

You can find more detailed installation instructions in [installing.md](docs/getting-started/installing.md)

## Getting started

```python
import concrete.numpy as cnp

@cnp.compiler({"x": "encrypted", "y": "encrypted"})
def add(x, y):
    return x + y

inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]

print(f"Compiling...")
circuit = add.compile(inputset)

examples = [(3, 4), (1, 2), (7, 7), (0, 0)]
for example in examples:
    result = circuit.encrypt_run_decrypt(*example)
    print(f"Evaluation of {' + '.join(map(str, example))} homomorphically = {result}")
```

if you have a function object that you cannot decorate, you can use the explicit `Compiler` API instead

```python
import concrete.numpy as cnp

def add(x, y):
    return x + y

compiler = cnp.Compiler(add, {"x": "encrypted", "y": "encrypted"})
inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]

print(f"Compiling...")
circuit = compiler.compile(inputset)

examples = [(3, 4), (1, 2), (7, 7), (0, 0)]
for example in examples:
    result = circuit.encrypt_run_decrypt(*example)
    print(f"Evaluation of {' + '.join(map(str, example))} homomorphically = {result}")
```

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
