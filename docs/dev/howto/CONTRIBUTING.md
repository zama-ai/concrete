
# Contributing

```{warning}
FIXME(alex): to see if something here needs some update
```

```{important}
There are two ways to contribute to **concretefhe**:
- you can open issues to report bugs, typos and suggest ideas
- you can ask to become an official contributor by emailing hello@zama.ai. Only approved contributors can send pull requests, so please make sure to get in touch before you do!
```

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

Each commit to **concretefhe**  should be conformant to the standards decided by the team. Conformance can be checked using the following command.

```shell
make pcc
```

### pytest

Of course, tests must be passing as well.

```shell
make pytest
```

### Coverage

The last requirement is to make sure you get a hundred percent code coverage. The `make pytest` command checks that by default and will fail with a coverage report at the end should some lines of your code not be executed during testing.

If your coverage is below hundred percent, you should write more tests and then create the pull request. If you ignore this warning and create the PR, GitHub actions will fail and your PR will not be merged anyway.

There may be cases where covering you code is not possible (exception that cannot be triggered in normal execution circumstances), in those cases you may be allowed to disable coverage for some specific lines. This should be the exception rather than the rule and reviewers will ask why some lines are not covered and if it appears they can be covered then the PR won't be accepted in that state.

## Commiting

We are using a consistent commit naming scheme, and you are expected to follow it as well (the CI will make sure you do). The accepted format can be printed to your terminal by running:

```shell
make show_scope
```

e.g.

```shell
git commit -m "feat: implement bounds checking"
git commit -m "feat(debugging): add an helper function to draw intermediate representation"
git commit -m "fix(tracing): fix a bug that crashed pytorch tracer"
```

To learn more about conventional commits, check [this](https://www.conventionalcommits.org/en/v1.0.0/) page. Remark that commit messages are checked in the comformance step, and rejected if they don't follow the rules.

## Before creating pull request

```{important}
We remind that only official contributors can send pull requests. To become such an official contributor, please email hello@zama.ai.
```

You should rebase on top of `main` branch before you create your pull request. We don't allow merge commits so rebasing on `main` before pushing gives you the best chance of avoiding having to rewrite parts of your PR later if some conflicts arise with other PRs being merged. After you commit your changes to your new branch, you can use the following commands to rebase:

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

# If there are conflicts during the rebase, resolve them
# and continue the rebase with the following command
git rebase --continue

# push the latest version of the local branch to remote
git push --force
```

You can learn more about rebasing in [here](https://git-scm.com/docs/git-rebase).
