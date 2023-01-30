"""
Simple script to check if a given version is the latest version of Concrete Numpy.
"""

import sys
from typing import List

import requests  # type: ignore
from semver import VersionInfo


def is_latest(new_version: VersionInfo, existing_versions: List[VersionInfo]):
    """
    Get if `new_version` is the latest version among `existing_versions`.
    """

    if new_version.prerelease:
        return False

    for existing_version in existing_versions:
        if existing_version.prerelease:
            continue

        if existing_version > new_version:
            return False

    return True


def main():
    """
    Run the script.
    """

    info = requests.get("https://api.github.com/repos/zama-ai/concrete-numpy/releases").json()

    new_version = VersionInfo.parse(sys.argv[1])
    existing_versions = [VersionInfo.parse(releases["name"][1:]) for releases in info]

    print(is_latest(new_version, existing_versions))


if __name__ == "__main__":
    main()
