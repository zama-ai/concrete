# Project Setup

{% hint style="info" %}
It is **strongly** recommended to use the development tool Docker. However, you are able to set the project up on a bare Linux or macOS as long as you have the required dependencies. You can see the required dependencies in `Dockerfile.dev` under `docker` directory.
{% endhint %}

## Installing `Python`

**Concrete-Numpy** is a `Python` library, so `Python` should be installed to develop it. `v3.8` and `v3.9` are, currently, the only supported versions.

You probably have Python already, but in case you don't, or in case you have an unsupported version, you can google `how to install python 3.8` and follow one of the results.

## Installing `Poetry`

`Poetry` is our package manager. It drastically simplifies dependency and environment management.

You can follow [this](https://python-poetry.org/docs/#installation) official guide to install it.

## Installing `make`

`make` is used to launch various commands such as formatting and testing.

On Linux, you can install `make` using the package manager of your distribution.

On macOS, you can install `gmake` via brew:

```shell
brew install make
```

{% hint style="info" %}
In the following sections, be sure to use the proper `make` tool for your system (i.e., `make`, `gmake`, etc).
{% endhint %}

## Cloning the repository

Now, it's time to get the source code of **Concrete-Numpy**.

Clone the git repository from GitHub using the protocol of your choice (ssh or https).

## Setting up the environment

Virtual environments are utilized to keep the project isolated from other `Python` projects in the system.

To create a new virtual environment and install dependencies, use the command:

```shell
make setup_env
```

## Activating the environment

To activate the newly created environment, use:

```shell
source .venv/bin/activate
```

## Syncing the environment

From time to time, new dependencies will be added to the project and old ones will be removed.mThe command below will make sure the project has the proper environment, so run it regularly.

```shell
make sync_env
```

## Troubleshooting

### In native setups.

If you are having issues in a native setup, you can try to re-create your environment like this:

```shell
deactivate
rm -rf .venv
make setup_env
source .venv/bin/activate
```

If the problem persists, you should consider using Docker. If you are working on a platform specific feature and Docker is not an option, you should create an issue so that we can take a look at your problem.

### In docker setups.

If you are having issues in a docker setup, you can try to re-build the docker image:

```shell
make docker_rebuild
make docker_start
```

If the problem persists, you should contact us for help.
