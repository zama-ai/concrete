# Installing

## Docker image

```{note}
The easiest way to install the framework is as a docker image. To get the image you need to login to ghcr.io with docker.
```

```{warning}
FIXME(Arthur): to check this is still valid
```

```shell
docker login ghcr.io
```

This command will ask for a username and a password. For username, just enter your GitHub username. For password, you should create a personal access token from [here](https://github.com/settings/tokens) selecting `read:packages` permission. Just paste the generated access token as your password, and you are good to go.

You can then either pull the latest docker image or a specific version:

```shell
docker pull ghcr.io/zama-ai/concretefhe:latest
# or
docker pull ghcr.io/zama-ai/concretefhe:v0.1.0
```

You can then use this image with the following command:

```shell
# Without local volume:
docker run --rm -it -p 8888:8888 ghcr.io/zama-ai/concretefhe:v0.1.0

# With local volume to save notebooks on host:
docker run --rm -it -p 8888:8888 -v /host/path:/data ghcr.io/zama-ai/concretefhe:v0.1.0
```

This will launch a **Concrete** enabled jupyter server in the docker, that you can access from your browser.

Alternatively you can just open a shell in the docker:

```shell
docker run --rm -it ghcr.io/zama-ai/concretefhe:v0.1.0 /bin/bash
```

## python package

```{warning}
FIXME(Arthur): explain how to install from pypi, when it is ready
```


