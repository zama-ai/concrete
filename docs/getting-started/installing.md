# Installation

Concrete is natively supported on Linux and macOS from Python 3.8 to 3.11 inclusive. If you have Docker in your platform, you can use the docker image to use Concrete.

## Using PyPI

You can install Concrete from PyPI:

```shell
pip install -U pip wheel setuptools
pip install concrete-python
```

## Using Docker

You can also get the Concrete docker image:

```shell
docker pull zamafhe/concrete-python:v2.0.0
docker run --rm -it zamafhe/concrete-python:latest /bin/bash
```
