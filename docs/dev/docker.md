# Docker Setup

## Installation

Before you start this section, go ahead and install Docker. You can follow [this](https://docs.docker.com/engine/install/) official guide if you need help.

## X forwarding

### Linux.

You can use xhost command:

```shell
xhost +localhost
```

### macOS.

To use X forwarding on Mac OS:

* Install XQuartz
* Open XQuartz.app application, make sure in the application parameters that `authorize network connections` are set (currently in the Security settings)
* Open a new terminal within XQuartz.app and type:

```shell
xhost +127.0.0.1
```

X server should be all set for Docker in the regular terminal.

## Building

You can use the dedicated target in the Makefile to build the docker image:

```shell
make docker_build
```

## Starting

You can use the dedicated target in the Makefile to start the docker session:

```shell
make docker_start
```
