# Getting Started

## Preparation

### Installing Python v3.8

You can follow [this](https://realpython.com/installing-python/) guide.

### Installing Poertry

You can follow [this](https://python-poetry.org/docs/#installation) guide.

### Cloning repository

```shell
git clone https://github.com/zama-ai/hdk.git
```

### Setting up environment

```shell
cd hdk
make setup_env
```

### Activating the environment

```shell
source .venv/bin/activate
```

### Syncing environment with the latest changes

```shell
make sync_env
```

## Module Structure

- hdk
    - common: types and utilities that can be used by multiple frontends (e.g., numpy, torch)
      - bounds_measurement: utilities for determining bounds of intermediate representation
      - data_types: type definitions of typing information of intermediate representation
      - debugging: utilities for printing/displaying intermediate representation
      - representation: type definitions of intermediate representation
      - tracing: utilities for generic function tracing used during intermediate representation creation
    - hnumpy: numpy frontend of hdk

## Contributing

### Creating a new branch

```shell
git checkout -b (feat|fix|refactor|test|benchmark|doc|chore)/short-description
```

e.g.

```shell
git checkout -b feat/explicit-tlu
```

### Before committing

Each commit to `hdk` should be comformant to the standards decided by the team. Conformance can be checked using the following commands.

```shell
make -k pytest
make -k pcc
```

### Before creating pull request

Commits on the latest version of `main` branch should be rebased to your branch before your PR can be accepted. This is to avoid merge commits and have a clean git log. After you commit your changes to your new branch, you can use the following commands to rebase.

```shell
git checkout main
git pull
git checkout $YOUR_BRANCH
git rebase main
git push --force
```

You can learn more about rebasing in [here](https://git-scm.com/docs/git-rebase).

The last requirement before creating your PR is to make sure you get a hundred percent code coverage. You can verify this using the following command.

```shell
BB=$YOUR_BRANCH make coverage
```

If your coverage is below hundred percent, you should write more tests and then create the pull request.
