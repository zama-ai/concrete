# Installation

Concrete is natively supported on Linux and macOS from Python 3.8 to 3.11 inclusive. If you have Docker in your platform, you can use the docker image to use Concrete.

## Using PyPI

You can install Concrete from PyPI:

```shell
pip install -U pip wheel setuptools
pip install concrete-python
```

There are some optional features which can be enabled by installing the `full` version:

```shell
pip install -U pip wheel setuptools
pip install concrete-python[full]
```

{% hint style="info" %}
Full version depends on [pygraphviz](https://pygraphviz.github.io/), which needs [graphviz](https://graphviz.org/) to be installed in the operating system so please [install](https://pygraphviz.github.io/documentation/stable/install.html) the operating system dependencies before installing `concrete-python[full]`. 
{% endhint %}

{% hint style="info" %}
Installing `pygraphviz` on macOS can be problematic (see https://github.com/pygraphviz/pygraphviz/issues/11).

If you're using homebrew, you may try the following:
```shell
brew install graphviz
CFLAGS=-I$(brew --prefix graphviz)/include LDFLAGS=-L$(brew --prefix graphviz)/lib pip --no-cache-dir install pygraphviz
```
before running:
```shell
pip install concrete-python[full]
```
{% endhint %}

## Using Docker

You can also get the Concrete docker image (replace "v2.4.0" below by the correct version you want):

```shell
docker pull zamafhe/concrete-python:v2.4.0
docker run --rm -it zamafhe/concrete-python:latest /bin/bash
```

Docker is not supported on Apple Silicon.
