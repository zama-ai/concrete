```{warning}
FIXME(Arthur): to check what needs to be updated here
```

# Project Setup

```{note}  
You will need Zama's specific environment with zamalang module to have the project fully functional. It is currently only delivered via the docker image (see the [docker](./docker.md) guide).
```

## Installing Python v3.8

**concretefhe** is a `Python` library. So `Python` should be installed to develop **concretefhe**. `v3.8` is the only supported version.

You can follow [this](https://realpython.com/installing-python/) guide to install it (alternatively you can google `how to install python 3.8`).

## Installing Poetry

`Poetry` is our package manager. It simplifies dependency and environment management by a lot.

You can follow [this](https://python-poetry.org/docs/#installation) official guide to install it.

## Installing make

The dev tools use make to launch the various commands.

On Linux you can install make from your distribution's preferred package manager.

On Mac OS you can install a more recent version of make via brew:

```shell
# check for gmake
which gmake
# If you don't have it, it will error out, install gmake
brew install make
# recheck, now you should have gmake
which gmake
```

It is possible to install gmake as make, check this [StackOverflow post](https://stackoverflow.com/questions/38901894/how-can-i-install-a-newer-version-of-make-on-mac-os) for more infos.

On Windows check [this GitHub gist](https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make).

```{hint}
In the next sections, be sure to use the proper `make` tool for your system, `make`, `gmake` or other.
```

## Cloning repository

Now, it's time to get the source code of **concretefhe**. You can use the following command to do that.

```shell
git clone https://github.com/zama-ai/concretefhe-internal.git
```

## Setting up environment

We are going to make use of virtual environments. This helps to keep the project isolated from other `Python` projects in the system. The following commands will create a new virtual environment under the project directory and install dependencies to it.

```shell
cd concretefhe-internal
make setup_env
```

## Activating the environment

Finally, all we need to do is to activate the newly created environment using the following command.

### macOS or Linux

```shell
source .venv/bin/activate
```

### Windows

```shell
source .venv/Scripts/activate
```

## Leaving the environment

After your work is done you can simply run the following command to leave the environment.

```shell
deactivate
```

## Syncing environment with the latest changes

From time to time, new dependencies will be added to project or the old ones will be removed. The command below will make sure the project have proper environment. So run it regularly!

```shell
make sync_env
```

