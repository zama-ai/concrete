# Installation

**Concrete Python** is natively supported on Linux and macOS for Python 3.8 onwards.

## Using PyPI

You can install **Concrete Python** from PyPI:

```shell
pip install -U pip wheel setuptools
pip install concrete-python
```

{% hint style="warning" %}
Apple Silicon is not supported for the time being. We're working on bringing support for it, which should arrive soon.
{% endhint %}

## Using Docker

You can also get the **Concrete Python** docker image:

```shell
docker pull zamafhe/concrete-python:v1.0.0
```

### Starting a Jupyter server.

By default, the entry point of the **Concrete Python** docker image is a jupyter server that you can access from your browser:

```shell
docker run --rm -it -p 8888:8888 zamafhe/concrete-python:v1.0.0
```

To save notebooks on host, you can use a local volume:

```shell
docker run --rm -it -p 8888:8888 -v /path/to/notebooks:/data zamafhe/concrete-python:v1.0.0
```

### Starting a Bash session.

Alternatively, you can launch a Bash session:

```shell
docker run --rm -it zamafhe/concrete-python:v1.0.0 /bin/bash
```
