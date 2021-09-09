---
name: Release
about: Issue template to prepare a release step by step.
title: "Release vX.Y.Z"
---
<!-- Make sure to set the proper version in the issue template -->
Release check-list:
<!-- Note that some of these steps will be automated in the future -->
- [ ] Check the release milestone issues, cut out what can't be completed in time
- [ ] Choose the version number, e.g. `vX.Y.Z` following semantic versioning: https://semver.org/
- [ ] Update the version in pyproject.toml to `X.Y.Z`
- [ ] Checkout the commit for release, create a signed tag with the version name `git tag -s -a -m "vX.Y.Z release" vX.Y.Z`, push it to GitHub with `git push origin refs/tags/vX.Y.Z`
- [ ] Run sanity checks inside the dev docker: `make pcc` and `make pytest && make coverage`
- [ ] On the build machine with docker installed, run in your OS terminal in the project dir: `make release_docker`
- [ ] Re-tag the image with `docker tag concretefhe-release:latest ghcr.io/zama-ai/concretefhe-release:vX.Y.Z`
- [ ] `docker login ghcr.io`, input your username and GitHub Personal Access Token (PAT). If not already done add `write:packages` to your PAT
- [ ] Push the release image `docker push ghcr.io/zama-ai/concretefhe-release:vX.Y.Z`
- [ ] See [here](https://docs.github.com/en/github/administering-a-repository/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) to create the release in GitHub using the existing tag, add the pull link \(`ghcr.io/zama-ai/concretefhe-release:vX.Y.Z`\) for the uploaded docker image

All done!
