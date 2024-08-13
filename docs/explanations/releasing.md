# Releasing Concrete

This document explains how Zama people can release a new version of Concrete.


## The process

### Create the release branch (if new minor)
All releases should be done on a release branch, releases branch are named `release/MAJOR.MINOR.x`:
- else you create a REVISION release in this case you should cherry-pick commits on the branch of the release you want to fix,
- else you create a new version so you need to create the new release branch.

```bash
git branch release/MAJOR.MINOR.x
```

Now this branch will be the branch from where all releases `vMAJOR.MINOR.*` will be done, and from where the gitbook documentation is built `https://docs.zama.ai/concrete/v/MAJOR.MINOR`

### Create a new draft release

If you want to create a new release `v.MAJOR.MINOR.REVISION`, you should update the `release/MAJOR.MINOR.x`. Each push on the release branch will start all tests of concrete. When you are happy with the state of the release branch, take care to update the API documentation,

```bash
./ci/scripts/make_apidocs.sh
```

If you miss it the release worflow will stops on the `release-checks` steps on `concrete_python_release.yml`

then you just need to tag.

```bash
git tag vMAJOR.MINOR.REVISION
git push origin vMAJOR.MINOR.REVISION
```

This new tag push will start the release workflows, the workflow build all release artifatcs then create a new draft release on github you can find here

```bash
https://github.com/zama-ai/concrete/releases/tag/vMAJOR.MINOR.REVISION
```

You should edit the changelog and the release documentation, then make it review.

### Create a new official release

When the new release documentation has been reviewed, you may save the release as a non draft release, then publish wheels on pypi using the `https://github.com/zama-ai/concrete/actions/workflows/push_wheels_to_public_pypi.yml` workflow, by setting the version number as `MAJOR.MINOR.VERSION`.

### Artifacts to check

If you follow the summary checklist
- [ ] Create release branch `git branch release/MAJOR.MINOR.x` or pull `git pull release/MAJOR.MINOR.x`.
- [ ] Cherry pick commits for the new release `git branch release/MAJOR.MINOR.x`.
- [ ] Update documentation `./ci/scripts/make_apidoc.sh`
- [ ] (id diff) `git commit -m "doc(frontend-python): Update API documentation for vMAJOR.MINOR.x"`
- [ ] Update release branch `git push origin release/MAJOR.MINOR.x`
- [ ] Tag the new version `git tag vMAJOR.MINOR.REVISION && git push origin vMAJOR.MINOR.REVISION`
- [ ] Wait the end of the `https://github.com/zama-ai/concrete/actions/workflows/concrete_python_release.yml`
- [ ] Edit and publish the Github release note `https://github.com/zama-ai/concrete/releases/tag/v2.MAJOR.MINOR`
- [ ] Then publish wheels on public PyPI using manually the workflow `https://github.com/zama-ai/concrete/actions/workflows/push_wheels_to_public_pypi.yml` 


you must have all this artifacts

- [ ] Release branch `https://github.com/zama-ai/concrete/tree/release/MAJOR.MINOR.x`,
- [ ] Github release note `https://github.com/zama-ai/concrete/releases/tag/v2.MAJOR.MINOR` (publicly released),
- [ ] Public PyPy wheels `https://pypi.org/project/concrete-python/2.MAJOR.MINOR/`
- [ ] Documentation up-to-date `https://docs.zama.ai/concrete`
- [ ] Zama PyPi CPU wheels `https://pypi.zama.ai/cpu/concrete-python/index.html`
- [ ] Zama PyPi GPU wheels `https://pypi.zama.ai/gpu/concrete-python/index.html`
- [ ] Docker images `https://hub.docker.com/r/zamafhe/concrete-python/tags`
