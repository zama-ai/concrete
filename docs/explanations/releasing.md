# Releasing Concrete

This document explains how Zama people can release a new version of Concrete.


## The process

### Create the release branch if needed

All releases should be done on a release branch: our release branches are named `release/MAJOR.MINOR.x` (eg, `release/2.7.x`):
- either you create a new version, then you need to create the new release branch (eg, the previous release was 2.6.x and now we release 2.7.0)

```bash
git branch release/MAJOR.MINOR.x
```

- or you create a dot release: in this case you should cherry-pick commits on the branch of the release you want to fix (eg, the previous release was 2.7.0 and now we release 2.7.1).

The `release/MAJOR.MINOR.x` branch will be the branch from where all releases `vMAJOR.MINOR.*` will be done, and from where the gitbook documentation is built `https://docs.zama.ai/concrete/v/MAJOR.MINOR`.

### Create a new draft release

Each push on the release branch will start all tests of Concrete. When you are happy with the state of the release branch, you need to update the API documentation:

```bash
./ci/scripts/make_apidocs.sh
```

If you miss it, the release worflow will stops on the `release-checks` steps on `concrete_python_release.yml`. Don't forget to push the updated API docs in the branch.

Then you just need to tag.

```bash
git tag vMAJOR.MINOR.REVISION
git push origin vMAJOR.MINOR.REVISION
```

This new tag push will start the release workflow: the workflow builds all release artifacts then create a new draft release on GitHub which you can find at `https://github.com/zama-ai/concrete/releases/tag/vMAJOR.MINOR.REVISION`.

You should edit the changelog and the release documentation, then make it reviewed by the product marketing team.

### Create a new official release

When the new release documentation has been reviewed, you may save the release as a non draft release, then publish wheels on pypi using the `https://github.com/zama-ai/concrete/actions/workflows/push_wheels_to_public_pypi.yml` workflow, by setting the version number as `MAJOR.MINOR.VERSION`.

### Artifacts to check

Follow the summary checklist:

- [ ] Create release branch `git branch release/MAJOR.MINOR.x` or pull `git pull release/MAJOR.MINOR.x`.
- [ ] Cherry pick commits for the new release `git branch release/MAJOR.MINOR.x`.
- [ ] Update documentation `./ci/scripts/make_apidoc.sh`
- [ ] (if diff) `git commit -m "doc(frontend-python): Update API documentation for vMAJOR.MINOR.x"`
- [ ] Update release branch `git push origin release/MAJOR.MINOR.x`
- [ ] Tag the new version `git tag vMAJOR.MINOR.REVISION && git push origin vMAJOR.MINOR.REVISION`
- [ ] Wait the end of the `https://github.com/zama-ai/concrete/actions/workflows/concrete_python_release.yml` build
- [ ] Edit and publish the Github release note `https://github.com/zama-ai/concrete/releases/tag/vMAJOR.MINOR.REVISION`
- [ ] Then publish wheels on public PyPI using manually the workflow `https://github.com/zama-ai/concrete/actions/workflows/push_wheels_to_public_pypi.yml` 

At the end, check all the artifacts:

- [ ] Release branch `https://github.com/zama-ai/concrete/tree/release/MAJOR.MINOR.x`,
- [ ] Github release note `https://github.com/zama-ai/concrete/releases/tag/vMAJOR.MINOR.REVISION` (publicly released),
- [ ] Public PyPy wheels `https://pypi.org/project/concrete-python/MAJOR.MINOR.REVISION/`
- [ ] Documentation up-to-date `https://docs.zama.ai/concrete`
- [ ] Zama PyPi CPU wheels `https://pypi.zama.ai/cpu/concrete-python/index.html`
- [ ] Zama PyPi GPU wheels `https://pypi.zama.ai/gpu/concrete-python/index.html`
- [ ] Docker images `https://hub.docker.com/r/zamafhe/concrete-python/tags`
