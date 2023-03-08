---
name: Release
about: Issue template to prepare a release step by step.
title: "Release vX.Y.Z (or vX.Y.Z-rc?)"
---
<!-- Make sure to set the proper version in the issue template -->
Please check all steps if it was either done/already done, at the end of a release all check boxes must have been checked.

Release check-list:
<!-- Note that some of these steps will be automated in the future -->
If it was not already done:
- [ ] Choose the version number, e.g. `vX.Y.Z` (can be `vX.Y.Z-rc?` for Release Candidates) following semantic versioning: https://semver.org/
- [ ] Update the project version to `X.Y.Z` (or `X.Y.Z-rc?`) by running:

```bash
VERSION=X.Y.Z make set_version
# or
VERSION=X.Y.Z-rc? make set_version
```

Then:
- [ ] For non RC releases: check the release milestone issues, cut out what can't be completed in time and change the milestones for these issues
- [ ] Checkout the commit for release
- [ ] Call `make release`, which creates a signed tag (requires GPG keys setup, see [here](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification)) and pushes it
- [ ] Wait for the release workflow to finish and check everything went well.

To continue the release cycle:
- [ ] Choose the version number for next release, e.g. `vA.B.C` (can be `vA.B.C-rc?` for Release Candidates) following semantic versioning: https://semver.org/
- [ ] Update the project version to `A.B.C` (or `A.B.C-rc?`) by running:

```bash
VERSION=A.B.C make set_version
# or
VERSION=A.B.C-rc? make set_version
```

All done!
