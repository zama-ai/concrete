# Installation

This document explains the steps to install **Concrete** into your project. 

**Concrete** is natively supported on Linux and macOS from Python 3.8 to 3.11 inclusive. If you have Docker in your platform, you can use the docker image to use **Concrete**.

## Using PyPI

Install **Concrete** from PyPI using the following commands:

```shell
pip install -U pip wheel setuptools
pip install concrete-python
```
{% hint style="info" %}
Not all versions are available on PyPI. If you need a version that is not on PyPI (including nightly releases), you can install it from our package index by adding `--extra-index-url https://pypi.zama.ai`.
{% endhint %}

To enable all the optional features, install the `full` version of **Concrete**:

```shell
pip install -U pip wheel setuptools
pip install concrete-python[full]
```

{% hint style="info" %}
Not all versions are available on PyPI. If you need a version that is not on PyPI (including nightly releases), you can install it from our package index by adding --index-url https://pypi.zama.ai/cpu.

In particular, wheels with **GPU support** are not on PyPI. You can install it from our package index by adding --index-url https://pypi.zama.ai/gpu, more information on GPU wheels [here](https://docs.zama.ai/concrete/execution-analysis/gpu_acceleration).
{% endhint %}

{% hint style="info" %}
The full version requires [pygraphviz](https://pygraphviz.github.io/), which depends on [graphviz](https://graphviz.org/). Make sure to [install](https://pygraphviz.github.io/documentation/stable/install.html) all the dependencies on your operating system before installing `concrete-python[full]`. 
{% endhint %}

{% hint style="info" %}
Installing `pygraphviz` on macOS can be problematic (see more details [here](https://github.com/pygraphviz/pygraphviz/issues/11)).

If you're using homebrew, you can try the following way:
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

You can also get the **Concrete** docker image. Replace `v2.4.0` below by the version you want to install:

```shell
docker pull zamafhe/concrete-python:v2.4.0
docker run --rm -it zamafhe/concrete-python:latest /bin/bash
```

Docker is not supported on Apple Silicon.
