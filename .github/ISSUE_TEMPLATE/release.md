---
name: Release
about: Issue template to prepare a release step by step.
title: "Release vX.Y.Z (or vX.Y.Zrc?)"
---
<!-- Make sure to set the proper version in the issue template -->
Please check all steps if it was either done/already done, at the end of a release all check boxes must have been checked.

Release check-list:
<!-- Note that some of these steps will be automated in the future -->
If it was not already done:
- [ ] Choose the version number, e.g. `vX.Y.Z` (can be `vX.Y.Zrc?` for Release Candidates) following semantic versioning: https://semver.org/
- [ ] Update the version in pyproject.toml to `X.Y.Z` (or `X.Y.Zrc?`)

Then:
- [ ] For non RC releases: check the release milestone issues, cut out what can't be completed in time and change the milestones for these issues
- [ ] Checkout the commit for release, create a signed tag (requires GPG keys setup, see [here](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification)) with the version name (careful for RC) `git tag -s -a -m "vX.Y.Z release" vX.Y.Z`, (or `vX.Y.Zrc?`) push it to GitHub with `git push origin refs/tags/vX.Y.Z` (or `vX.Y.Zrc?`)
- [ ] Wait for the release workflow to finish and get the image url from the notification or the logs
- [ ] See [here](https://docs.github.com/en/github/administering-a-repository/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) to create the release in GitHub using the existing tag, add the pull link copied from the step before \(`ghcr.io/zama-ai/concretefhe:vX.Y.Z`\) (or `vX.Y.Zrc?`) for the uploaded docker image. Mark release as pre-release for an `rc` version. See template below:

This is the release markdown template you should copy and update:
```
**Docker Image:** ghcr.io/zama-ai/concretefhe:vX.Y.Z
**Documentation:** https://docs.zama.ai/concrete
```

To continue the release cycle:
- [ ] Choose the version number for next release, e.g. `vA.B.C` (can be `vA.B.Crc?` for Release Candidates) following semantic versioning: https://semver.org/
- [ ] Update the version in pyproject.toml to `A.B.C` (or `A.B.Crc?`)

All done!
