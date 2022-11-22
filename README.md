<p align="center">
<!-- product name logo -->
  <img width=600 src="https://user-images.githubusercontent.com/5758427/193612313-6b1124c7-8e3e-4e23-8b8c-57fd43b17d4f.png">
</p>

<p align="center">
<!-- Version badge using shields.io -->
  <a href="https://github.com/zama-ai/concrete-numpy/releases">
    <img src="https://img.shields.io/github/v/release/zama-ai/concrete-numpy?style=flat-square">
  </a>
<!-- Link to docs badge using shields.io -->
  <a href="https://docs.zama.ai/concrete-numpy/">
    <img src="https://img.shields.io/badge/read-documentation-yellow?style=flat-square">
  </a>
<!-- Community forum badge using shields.io -->
  <a href="https://community.zama.ai/c/concrete-numpy">
    <img src="https://img.shields.io/badge/community%20forum-online-brightgreen?style=flat-square">
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


**Concrete Numpy** is an open-source library which simplifies the use of fully homomorphic encryption (FHE).

FHE is a powerful cryptographic tool, which allows computation to be performed directly on encrypted data without needing to decrypt it first.

With FHE, you can build services that preserve the privacy of the users. FHE is also great against data breaches as everything is done on encrypted data. Even if the server is compromised, in the end no sensitive data is leaked.

## Main features.

- Ability to compile Python functions (that may use NumPy within) to their FHE equivalents, to operate on encrypted data
- Support for [large collection of operators](https://docs.zama.ai/concrete-numpy/getting-started/compatibility)
- Partial support for floating points
- Support for table lookups on integers
- Support for integration with Client / Server architectures

## Installation.

|               OS / HW                | Available on Docker | Available on PyPI |
| :----------------------------------: | :-----------------: | :--------------: |
|                Linux                 |         Yes         |       Yes        |
|               Windows                |         Yes         |   Coming soon    |
|     Windows Subsystem for Linux      |         Yes         |       Yes        |
|            macOS (Intel)             |         Yes         |       Yes        |
| macOS (Apple Silicon, ie M1, M2 etc) |    Yes (Rosetta)    |   Coming soon    |


The preferred way to install Concrete Numpy is through PyPI:

```shell
pip install concrete-numpy
```

You can get the concrete-numpy docker image by  pulling the latest docker image:

```shell
docker pull zamafhe/concrete-numpy:v0.9.0
```

You can find more detailed installation instructions in [installing.md](docs/getting-started/installing.md)

## Getting started.

```python
import concrete.numpy as cnp

def add(x, y):
    return x + y

compiler = cnp.Compiler(add, {"x": "encrypted", "y": "encrypted"})
inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]

print(f"Compiling...")
circuit = compiler.compile(inputset)

print(f"Generating keys...")
circuit.keygen()

examples = [(3, 4), (1, 2), (7, 7), (0, 0)]
for example in examples:
    encrypted_example = circuit.encrypt(*example)
    encrypted_result = circuit.run(encrypted_example)
    result = circuit.decrypt(encrypted_result)
    print(f"Evaluation of {' + '.join(map(str, example))} homomorphically = {result}")
```

or if you have a simple function that you can decorate, and you don't care about explicit steps of key generation, encryption, evaluation and decryption:

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

## Documentation.

Full, comprehensive documentation is available at [https://docs.zama.ai/concrete-numpy](https://docs.zama.ai/concrete-numpy).

## Target audience.

Concrete Numpy is a generic library that supports variety of use cases. Because of this flexibility,
it doesn't provide primitives for specific use cases.

If you have a specific use case, or a specific field of computation, you may want to build abstractions on top of Concrete Numpy.

One such example is [Concrete ML](https://github.com/zama-ai/concrete-ml), which is built on top of Concrete Numpy to simplify Machine Learning oriented use cases.

## Tutorials.

Various tutorials are proposed in the documentation to help you start writing homomorphic programs:

- How to use Concrete Numpy with [Decorators](https://docs.zama.ai/concrete-numpy/tutorials/decorator)
- Partial support of [Floating Points](https://docs.zama.ai/concrete-numpy/tutorials/floating_points)
- How to perform [Table Lookup](https://docs.zama.ai/concrete-numpy/tutorials/table_lookup)

More generally, if you have built awesome projects using Concrete Numpy, feel free to let us know and we'll link to it!

## Need support?

<a target="_blank" href="https://community.zama.ai">
  <img src="https://user-images.githubusercontent.com/5758427/191792238-b132e413-05f9-4fee-bee3-1371f3d81c28.png">
</a>

## License.

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
