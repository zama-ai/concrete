```{warning}
FIXME(arthur): to update with the new workflow
```

# Creating A Release On GitHub

## Release Candidate cycle

Before settling for a final release, we go through a Release Candidate (RC) cycle. The idea is that once the code base and documentations look ready for a release you create an RC Release by opening an issue with the release template [here](https://github.com/zama-ai/concretefhe-internal/issues/new?assignees=&labels=&template=release.md), starting with version `vX.Y.Zrc1` and then with versions `vX.Y.Zrc2`, `vX.Y.Zrc3`...

## Proper release

Once the last RC is deemed ready, open an issue with the release template using the last RC version from which you remove the `rc?` part (i.e. `v12.67.19` if your last RC version was `v12.67.19-rc4`) on [github](https://github.com/zama-ai/concretefhe-internal/issues/new?assignees=&labels=&template=release.md).
