"""
Simple script to check if a given version is a pre-release version.
"""

import sys

from semver import VersionInfo


def is_prerelease(version: VersionInfo):
    """
    Get if `version` is a pre-release version.
    """

    return version.prerelease is not None


def main():
    """
    Run the script.
    """

    version = VersionInfo.parse(sys.argv[1])
    print(str(is_prerelease(version)).lower())


if __name__ == "__main__":
    main()
