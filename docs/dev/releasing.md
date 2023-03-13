# Release process

## Release candidate cycle

Throughout the quarter, many release candidates are released. Those candidates are released in a private package repository. At the end of the quarter, we take the latest release candidate, and release it in PyPI without `rcX` tag.

## Release flow

* Checkout to the commit that you want to include in the release (everything before this commit and this commit will be in the release)
* Run `make release`
* Wait for CI to complete
* Checkout to `chore/version` branch
* Run `VERSION=a.b.c-rcX make set_version` with appropriate version
* Push the branch to origin
* Create a PR to merge it to main
* Wait for CI to finish and get approval in the meantime
* Merge the version update to main
