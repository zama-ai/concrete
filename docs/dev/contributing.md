# Contribute

There are three ways to contribute to Concrete:

- You can open issues to report bugs and typos and to suggest ideas.
- You can become an official contributor, but you need to sign our Contributor License Agreement (CLA) on your first contribution. Our CLA-bot will guide you through the process when you will open a Pull Request on Github.
- You can also provide new tutorials or use-cases, showing what can be done with the library. The more examples we have, the better and clearer it is for the other users.

## Requirements

- TODO
- Python v3.8+

## Project setup

TODO

## Before committing

### Concrete Python

#### Testing

Concrete Python requires 100% code coverage, and it's checked in GitHub actions. Without it, PRs cannot be merged.

It's a good idea to run the tests with coverage locally using:

```shell
make pytest-multi
```

#### Linting

If the python frontend is modified, it's a good idea to run the pre-commit checks using:

```shell
make pcc
```

If pcc results in formatting errors, it's a good idea to run conformance using:

```shell
make conformance
```

Other issues need to be resolved manually.

## Committing

Concrete uses a consistent commit naming scheme, and you are expected to follow it as well. The accepted format can be seen in the [GitHub workflow](https://github.com/zama-ai/concrete/blob/main/.github/workflows/block_merge.yml) that checks it.

Here are some good messages:

```shell
git commit -m "feat(frontend-python): add identity extension"
git commit -m "fix(compiler): fix tiling"
```

To learn more about conventional commits, check [this page](https://www.conventionalcommits.org/en/v1.0.0/).

## Rebasing

You should rebase on top of the repository's `main` branch before you create your pull request. Merge commits are not allowed, so rebasing on `main` before pushing gives you the best chance of to avoid rewriting parts of your PR later if conflicts arise with other PRs being merged. After you commit changes to your forked repository, you can use the following commands to rebase your main branch with Concrete's:

```shell
# Add the Concrete repository as remote, named "upstream" 
git remote add upstream git@github.com:zama-ai/concrete.git

# Fetch all last branches and changes from Concrete
git fetch upstream

# Checkout to your local main branch
git checkout main

# Rebase on top of main
git rebase upstream/main

# If there are conflicts during the rebase, resolve them
# and continue the rebase with the following command
git rebase --continue

# Push the latest version of your local main to your remote forked repository
git push --force origin main
```

You can learn more about rebasing [here](https://git-scm.com/docs/git-rebase).

## Creating pull request

You can now open a pull-request [in the Concrete repository](https://github.com/zama-ai/concrete/pulls). For more details on how to do so from a forked repository, please read GitHub's [official documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) on the subject.
