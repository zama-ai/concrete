<p align="center">
<!-- product name logo -->
  <img width=600 src="https://user-images.githubusercontent.com/5758427/231207493-62676aea-4cb9-4bb4-92b0-20309c8a933a.png">
</p>
<hr/>
<p align="center">
  <a href="https://docs.zama.ai/concrete"> 📒 Read documentation</a> | <a href="https://zama.ai/community"> 💛 Community support</a>
</p>
<p align="center">
<!-- Version badge using shields.io -->
  <a href="https://github.com/zama-ai/concrete/releases">
    <img src="https://img.shields.io/github/v/release/zama-ai/concrete?style=flat-square">
  </a>
<!-- Zama Bounty Program -->
  <a href="https://github.com/zama-ai/bounty-program">
    <img src="https://img.shields.io/badge/Contribute-Zama%20Bounty%20Program-yellow?style=flat-square">
  </a>
</p>
<hr/>

:warning: Starting from v1, Concrete Rust Libraries are now deprecated and replaced by [TFHE-rs](https://github.com/zama-ai/tfhe-rs), Concrete
is now, exclusively, Zama TFHE Compiler. Read full announcement [here](https://www.zama.ai/post/announcing-concrete-v1-0-0)

**Concrete** is an open-source FHE Compiler which simplifies the use of fully homomorphic encryption (FHE).

FHE is a powerful cryptographic tool, which allows computation to be performed directly on encrypted data without needing to decrypt it first. With FHE, you can build services that preserve privacy for all users. FHE is also great against data breaches as everything is done on encrypted data. Even if the server is compromised, in the end no sensitive data is leaked.

Since writing FHE programs can be difficult, Concrete, based on LLVM, make this process easier for developers.

## Main features

- Ability to compile Python functions (that may include NumPy) to their FHE equivalents, to operate on encrypted data
- Support for [large collection of operators](https://docs.zama.ai/concrete/getting-started/compatibility)
- Partial support for floating points
- Support for table lookups on integers
- Support for integration with Client / Server architectures

## Installation

|               OS / HW                | Available on Docker | Available on PyPI |
| :----------------------------------: | :-----------------: | :--------------: |
|                Linux                 |         Yes         |       Yes        |
|               Windows                |         Yes         |        No        |
|     Windows Subsystem for Linux      |         Yes         |       Yes        |
|            macOS (Intel)             |         Yes         |       Yes        |
|            macOS (Apple Silicon)     |         Yes         |       Yes        |


The preferred way to install Concrete is through PyPI:

```shell
pip install concrete-python
```

You can get the concrete-python docker image by pulling the latest docker image:

```shell
docker pull zamafhe/concrete-python:v1.0.0
```

You can find more detailed installation instructions in [installing.md](docs/getting-started/installing.md)

## Getting started

```python
from concrete import fhe

def add(x, y):
    return x + y

compiler = fhe.Compiler(add, {"x": "encrypted", "y": "encrypted"})
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
from concrete import fhe

@fhe.compiler({"x": "encrypted", "y": "encrypted"})
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

## Documentation

Full, comprehensive documentation is available at [https://docs.zama.ai/concrete](https://docs.zama.ai/concrete).

## Target users

Concrete is a generic library that supports a variety of use cases. Because of this flexibility,
it doesn't provide primitives for specific use cases.

If you have a specific use case, or a specific field of computation, you may want to build abstractions on top of Concrete.

One such example is [Concrete ML](https://github.com/zama-ai/concrete-ml), which is built on top of Concrete to simplify Machine Learning oriented use cases.

## Tutorials

Various tutorials are proposed in the documentation to help you start writing homomorphic programs:

- How to use Concrete with [Decorators](https://docs.zama.ai/concrete/tutorials/decorator)
- Partial support of [Floating Points](https://docs.zama.ai/concrete/tutorials/floating_points)
- How to perform [Table Lookup](https://docs.zama.ai/concrete/tutorials/table_lookups)

If you have built awesome projects using Concrete, feel free to let us know and we'll link to it.


## Project layout

`concrete` project is a set of several modules which are high-level frontends, compilers, backends and side tools.
- `frontends` directory contains a `python` frontend.
- `compilers` directory contains the `concrete-compiler` and `concrete-optimizer` modules. `concrete-compiler` is a compiler that:
  - synthetize a FHE computation dag expressed as a [MLIR](https://mlir.llvm.org/) dialect
  - compile to a set of artifacts
  - and provide tools to manipulate those artifacts at runtime.
`concrete-optimizer` is a specific module used by the compiler to find the best, secure and accurate set of cryptographic parameters for a given dag.
- The `backends` directory contains implementations of cryptographic primitives on different computation unit, used by  `concrete-compiler` runtime. `concrete-cpu` module provides CPU implementation, while `concrete-cuda` module provides GPU implementation using the CUDA platform.
- The `tools` directory contains side tools used by the rest of the project.

## Need support?

<a target="_blank" href="https://community.zama.ai">
  <img src="https://user-images.githubusercontent.com/5758427/231145251-9cb3f03f-3e0e-4750-afb8-2e6cf391fa43.png">
</a>

## Citing Concrete
To cite Concrete in academic papers, please use the following entry:

```text
@Misc{Concrete,
  title={{Concrete: TFHE Compiler that converts python programs into FHE equivalent}},
  author={Zama},
  year={2022},
  note={\url{https://github.com/zama-ai/concrete}},
}
```

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
