# Getting Started

## Preparation

Before you can start improving `hdk` you need to set up your development environment! This section will show how you can do that.

### Installing Python v3.8

`hdk` is a `Python` library. So `Python` should be installed to develop `hdk`. `v3.8` is recommended because our CI also uses `v3.8`.

You can follow [this](https://realpython.com/installing-python/) guide to install it (alternatively you can google `how to install python 3.8`).

### Installing Poetry

`Poetry` is our package manager. It simplifies dependency and environment management by a lot.

You can follow [this](https://python-poetry.org/docs/#installation) official guide to install it.

### Cloning repository

Now, it's time to get the source code of `hdk`. You can use the following command to do that.

```shell
git clone https://github.com/zama-ai/hdk.git
```

### Setting up environment

We are going to make use of virtual environments. This helps to keep the project isolated from other `Python` projects in the system. The following commands will create a new virtual environment under the project directory and install dependencies to it.

```shell
cd hdk
make setup_env
```

### Activating the environment

Finally, all we need to do is to activate the newly created environment using the following command.

```shell
source .venv/bin/activate
```

### Leaving the environment

After your work is done you can simply run the following command to leave the environment.

```shell
deactivate
```

### Syncing environment with the latest changes

From time to time, new dependencies will be added to project or the old ones will be removed. The command below will make sure the project have proper environment. So run it regularly!

```shell
make sync_env
```

## Terminology

In this section we will go over some terms that we use throughout the project.

- intermediate representation
    - a data structure to represent a calculation
    - basically a computation graph where nodes are either inputs or operations on other nodes
- tracing
    - it is our technique to take directly a plain numpy function from a user and deduce its intermediate representation in a painless way for the user
- bounds
    - before intermediate representation is sent to the compiler, we need to know which node will output which type (e.g., uint3 vs uint5)
    - there are several ways to do this but the simplest one is to evaluate the intermediate representation with all combinations of inputs and remember the maximum and the minimum values for each node, which is what we call bounds, and bounds can be used to determine the appropriate type for each node

## Module Structure

In this section, we will discuss the module structure of hdk briefly. You are encouraged to check individual `.py` files to learn more!

- hdk
    - common: types and utilities that can be used by multiple frontends (e.g., numpy, torch)
      - bounds_measurement: utilities for determining bounds of intermediate representation
      - compilation: type definitions related to compilation (e.g., compilation config, compilation artifacts)
      - data_types: type definitions of typing information of intermediate representation
      - debugging: utilities for printing/displaying intermediate representation
      - extensions: utilities that provide special functionality to our users
      - representation: type definitions of intermediate representation
      - tracing: utilities for generic function tracing used during intermediate representation creation
    - hnumpy: numpy frontend of hdk

## Contributing

Now, you have a working environment, and you know what is where in the project. You are ready to contribute! Well, not so fast let's go over some other important things that you need to be careful about.

### Creating a new branch

We are using a consistent branch naming scheme, and you are expected to follow it as well. Here is the format and some examples.

```shell
git checkout -b {feat|fix|refactor|test|benchmark|doc|style|chore}/short-description_$issue_id
```

e.g.

```shell
git checkout -b feat/explicit-tlu_11
git checkout -b fix/tracing_indexing_42
```

### Before committing

Each commit to `hdk` should be comformant to the standards decided by the team. Conformance can be checked using the following commands.

```shell
make -k pcc
make pytest
```

### Commiting

We are using a consistent commit naming scheme, and you are expected to follow it as well. Here is the format and some examples.

```shell
git commit -m "{feat|fix|refactor|test|benchmark|doc|style|chore}{($location)}?: description of the change"
```

e.g.

```shell
git commit -m "feat: implement bounds checking"
git commit -m "feat(debugging): add an helper function to draw intermediate representation"
git commit -m "fix(tracing): fix a bug that crashed pytorch tracer"
```

To learn more about conventional commits, check [this](https://www.conventionalcommits.org/en/v1.0.0/) page.

### Before creating pull request

You should rebase on top of `main` branch before you create your pull request. This is to avoid merge commits and have a clean git log. After you commit your changes to your new branch, you can use the following commands to rebase.

```shell
# fetch the list of active remote branches
git fetch --all --prune

# checkout to main
git checkout main

# pull the latest changes to main (--ff-only is there to prevent accidental commits to main)
git pull --ff-only

# checkout back to your branch
git checkout $YOUR_BRANCH

# rebase on top of main branch
git rebase main

# push the latest version of the local branch to remote 
git push --force
```

You can learn more about rebasing in [here](https://git-scm.com/docs/git-rebase).

The last requirement before creating your PR is to make sure you get a hundred percent code coverage. You can verify this using the following command.

```shell
make pytest
make coverage
```

Note that this will compare the coverage with `origin/main`. If you want to set a custom base branch, you can specify `BB` environment variable like so `BB=$YOUR_BASE_BRANCH make coverage`.

If your coverage is below hundred percent, you should write more tests and then create the pull request. If you ignore this warning and create the PR, GitHub actions will fail and your PR will not be merged anyway.
