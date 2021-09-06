
# Contributing

There are two ways to contribute to `concretefhe`:
- you can open issues to report bugs, typos and suggest ideas
- you can ask to become an official contributor by emailing hello@zama.ai. Only approved contributors can send pull requests, so please make sure to get in touch before you do!

Let's go over some other important things that you need to be careful about.

## Creating a new branch

We are using a consistent branch naming scheme, and you are expected to follow it as well. Here is the format and some examples.

```shell
git checkout -b {feat|fix|refactor|test|benchmark|doc|style|chore}/short-description_$issue_id
```

e.g.

```shell
git checkout -b feat/explicit-tlu_11
git checkout -b fix/tracing_indexing_42
```

## Before committing

### Conformance

Each commit to `concretefhe` should be comformant to the standards decided by the team. Conformance can be checked using the following commands.

```shell
make pcc
make pytest
```

### pytest

Of course, tests must be passing as well.

```shell
make pytest
```

### Coverage

The last requirement is to make sure you get a hundred percent code coverage. You can verify this using the following command (after having done `make pytest`).

```shell
make coverage
```

Remark that only calling `make pytest` will give you information about the coverage, at the end of the execution, but the test will not return a failure if the coverage is not a hundred percent, as opposed to a call to `make coverage`.

Note that this will compare the coverage with `origin/main`. If you want to set a custom base branch, you can specify `BB` environment variable like so `BB=$YOUR_BASE_BRANCH make coverage`.

If your coverage is below hundred percent, you should write more tests and then create the pull request. If you ignore this warning and create the PR, GitHub actions will fail and your PR will not be merged anyway.

## Commiting

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

## Before creating pull request

We remind that only official contributors can send pull requests. To become such an official contributor, please email hello@zama.ai.

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
