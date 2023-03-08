# Contribute

{% hint style="info" %}
There are two ways to contribute to **Concrete-Numpy** or to **Concrete** tools in general:

* You can open issues to report bugs and typos and to suggest ideas.
* You can ask to become an official contributor by emailing hello@zama.ai. Only approved contributors can send pull requests (PRs), so please make sure to get in touch before you do!
{% endhint %}

Now, let's go over some other important items that you need to know.

## Creating a new branch

We are using a consistent branch naming scheme, and you are expected to follow it as well. Here is the format:

```shell
git checkout -b {feat|fix|refactor|test|benchmark|doc|style|chore}/short-description
```

...and here are some examples:

```shell
git checkout -b feat/direct-tlu
git checkout -b fix/tracing-indexing
```

## Before committing

### Conformance.

Each commit to **Concrete-Numpy** should conform to the standards decided by the team. Conformance can be checked using the following command:

```shell
make pcc
```

### Testing.

On top of conformance, all tests must pass with 100% code coverage across the codebase:

```shell
make pytest
```

{% hint style="info" %}
There may be cases where covering 100% of the code is not possible (e.g., exceptions that cannot be triggered in normal execution circumstances). In those cases, you may be allowed to disable coverage for some specific lines. This should be the exception rather than the rule. Reviewers may ask why some lines are not covered and, if it appears they can be covered, then the PR won't be accepted in that state.
{% endhint %}

## Committing

We are using a consistent commit naming scheme, and you are expected to follow it as well. Again, here is the accepted format:

```shell
make show_scope
```

...and some examples:

```shell
git commit -m "feat: implement bounds checking"
git commit -m "feat(debugging): add an helper function to print intermediate representation"
git commit -m "fix(tracing): fix a bug that crashed pytorch tracer"
```

To learn more about conventional commits, check [this](https://www.conventionalcommits.org/en/v1.0.0/) page.

## Before creating a pull request

{% hint style="info" %}
We remind you that only official contributors can send pull requests. To become an official contributor, please email hello@zama.ai.
{% endhint %}

You should rebase on top of the `main` branch before you create your pull request. We don't allow merge commits, so rebasing on `main` before pushing gives you the best chance of avoiding rewriting parts of your PR later if conflicts arise with other PRs being merged. After you commit your changes to your new branch, you can use the following commands to rebase:

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

You can learn more about rebasing [here](https://git-scm.com/docs/git-rebase).
