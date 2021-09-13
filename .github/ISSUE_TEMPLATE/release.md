---
name: Release
about: Issue template to prepare a release step by step.
title: "Release vX.Y.Z (or vX.Y.Zrc?)"
---
<!-- Make sure to set the proper version in the issue template -->
Release check-list:
<!-- Note that some of these steps will be automated in the future -->
- [ ] Choose the version number, e.g. `vX.Y.Z` (can be `vX.Y.Zrc?` for Release Candidates) following semantic versioning: https://semver.org/
- [ ] Update the version in pyproject.toml to `X.Y.Z` (or `X.Y.Zrc?`)
- [ ] Check the release milestone issues, cut out what can't be completed in time
- [ ] Checkout the commit for release, create a signed tag (requires GPG keys setup, see [here](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification)) with the version name (careful for RC) `git tag -s -a -m "vX.Y.Z release" vX.Y.Z`, (or `vX.Y.Zrc?`) push it to GitHub with `git push origin refs/tags/vX.Y.Z` (or `vX.Y.Zrc?`)
- [ ] Run sanity checks inside the dev docker: `make docker_build_and_start`, `make pcc` and `make pytest && make coverage`
- [ ] On the build machine with docker installed, run in your OS terminal in the project dir: `make release_docker`
- [ ] Re-tag the image with `docker tag concretefhe:latest ghcr.io/zama-ai/concretefhe:vX.Y.Z` (or `vX.Y.Zrc?`)
- [ ] `docker login ghcr.io`, input your username and GitHub Personal Access Token (PAT). If not already done add `write:packages` to your PAT
- [ ] Push the release image `docker push ghcr.io/zama-ai/concretefhe:vX.Y.Z` (or `vX.Y.Zrc?`)
- [ ] See [here](https://docs.github.com/en/github/administering-a-repository/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) to create the release in GitHub using the existing tag, add the pull link \(`ghcr.io/zama-ai/concretefhe:vX.Y.Z`\) (or `vX.Y.Zrc?`) for the uploaded docker image

All done!
