
<p align="center">
  <img src="docs/_static/CN_logo.png">
</p>

Concrete Numpy is an open-source set of tools which aims to simplify the use of fully homomorphic encryption (FHE) for data scientists.

With Concrete Numpy, data scientists can implement machine learning models using a subset of numpy that compiles to FHE. They will be able to train models with popular machine learning libraries and then convert the prediction functions of these models to FHE with concrete-numpy.

<!-- TOC -->

- [concrete-numpy](#concrete-numpy)
    - [Links](#links)
    - [For end users](#for-end-users)
        - [Installation](#Installation)
        - [A simple example](#a-simple-example-numpy-addition-in-fhe)
    - [For developers](#for-developers)
        - [Project setup](#project-setup)
        - [Documenting](#documenting)
        - [Developing](#developing)
        - [Contributing](#contributing)
    - [License](#license)

<!-- /TOC -->
# Links

- [documentation](https://docs.zama.ai/concrete-numpy/main/)
- [community website](https://community.zama.ai/c/concrete-numpy/7)
- [machine learning examples](https://docs.zama.ai/concrete-numpy/main/user/advanced_examples/index.html)
# For end users

## Installation

The preferred way to use concrete-numpy is through docker. You can get the concrete-numpy docker image by  pulling the latest docker image:

`docker pull zamafhe/concrete-numpy:latest`

To install Concrete Numpy from PyPi, run the following:

`pip install concrete-numpy`

You can find more detailed installation instructions in [installing.md](docs/user/basics/installing.md)


## A simple example: numpy addition in FHE

```python
import concrete.numpy as hnp

def add(x, y):
    return x + y

inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]
compiler = hnp.NPFHECompiler(add, {"x": "encrypted", "y": "encrypted"})

print(f"Compiling...")
circuit = compiler.compile_on_inputset(inputset)

examples = [(3, 4), (1, 2), (7, 7), (0, 0)]
for example in examples:
    result = circuit.run(*example)
    print(f"Evaluation of {' + '.join(map(str, example))} homomorphically = {result}")
```

# For developers

### Project setup

Installation steps are described in [project_setup.md](docs/dev/howto/project_setup.md).
Information about how to use Docker for development are available in [docker.md](docs/dev/howto/docker.md).

### Documenting

Some information about how to build the documentation of `concrete-numpy` are available in [documenting.md](docs/dev/howto/documenting.md). Notably, our documentation is pushed to [https://docs.zama.ai/concrete-numpy/](https://docs.zama.ai/concrete-numpy/).

### Developing

Some information about our terminology and the infrastructure of `concrete-numpy` are available in [terminology_and_structure.md](docs/dev/explanation/terminology_and_structure.md). An in-depth look at what is done in `concrete-numpy` is available in [compilation.md](docs/dev/explanation/compilation.md).

### Contributing

Information about how to contribute are available in [contributing.md](docs/dev/howto/contributing.md).


# License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
