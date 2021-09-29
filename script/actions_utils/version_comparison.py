# -*- coding: utf-8 -*-

"""Helper script for github actions to compare versions"""
import argparse
import re
import sys


def main(args):
    """Entry point"""
    print(args, file=sys.stderr)
    semver_matcher = re.compile(r"^(v)?([\d.]+)(rc\d+)?$")
    # Keep versions that are not release candidate
    all_versions = [
        tuple(map(int, match.group(2).split(".")))
        for version in args.existing_versions
        if (match := semver_matcher.match(version)) is not None and match.group(3) is None
    ]

    nv_match = semver_matcher.match(args.new_version)
    new_version = (
        tuple(map(int, nv_match.group(2).split(".")))
        if nv_match is not None and nv_match.group(3) is None
        else None
    )

    all_versions.append(new_version)

    nv_is_rc = new_version is None
    nv_is_latest = not nv_is_rc and max(all_versions) == new_version
    print(str(nv_is_latest).lower())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Compare new version to previous versions and determine if it's the latest",
        allow_abbrev=False,
    )

    parser.add_argument("--new-version", type=str, required=True, help="The new version to compare")
    parser.add_argument(
        "--existing-versions",
        type=str,
        nargs="+",
        required=True,
        help="The list of existing versions",
    )

    cli_args = parser.parse_args()

    main(cli_args)
