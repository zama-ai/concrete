# Installing

## Docker image

Currently the project is only available as a docker image. To get the image you need to login to ghcr.io with docker.

```shell
docker login ghcr.io
```

This command will ask for a username and a password. For username, just enter your GitHub username. For password, you should create a personal access token from [here](https://github.com/settings/tokens) selecting `read:packages` permission. Just paste the generated access token as your password, and you are good to go.

You can then either pull the latest docker image or a specific version:

```shell
docker pull ghcr.io/zama-ai/concretefhe-internal:latest
# or
docker pull ghcr.io/zama-ai/concretefhe-internal:v0.1.0
```

You can then use this image with the following command:

```shell
# Without local volume:
docker run --rm -it -p 8888:8888 ghcr.io/zama-ai/concretefhe-internal:v0.1.0

# With local volume to save notebooks on host:
docker run --rm -it -p 8888:8888 -v /host/path:/data ghcr.io/zama-ai/concretefhe-internal:v0.1.0
```

This will launch a concretefhe enabled jupyter server in the docker, that you can access from your browser.

Alternatively you can just open a shell in the docker:

```shell
docker run --rm -it ghcr.io/zama-ai/concretefhe-internal:v0.1.0 /bin/bash

root@e2d6c00e2f3d:/data# python3
Python 3.8.10 (default, Jun  2 2021, 10:49:15)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import concrete.numpy as hnp
>>> dir(hnp)
['ClearScalar', 'ClearTensor', 'CompilationArtifacts', 'CompilationConfiguration', 'EncryptedScalar', 'EncryptedTensor', 'Float', 'Float32', 'Float64', 'Integer', 'LookupTable', 'ScalarValue', 'SignedInteger', 'TensorValue', 'UnsignedInteger', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'compile', 'compile_numpy_function', 'compile_numpy_function_into_op_graph', 'draw_graph', 'get_printable_graph', 'np_dtypes_helpers', 'trace_numpy_function', 'tracing']
>>>
```




